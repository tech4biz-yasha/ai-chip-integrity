#!/usr/bin/env python3
"""
screen.py  -  AI Chip Integrity Suite, screener v0.1.0

Runs one fixed matrix multiply many times on a single device and checks every
result two ways:

  1. Reference check. Every output element must sit inside a proven
     rounding-error bound around a float64 answer computed on the CPU
     (Higham's componentwise bound for inner products, plus output rounding).
  2. Repeat check. Every run must be bit-for-bit identical to the first run
     on the same device.

A run that fails a check while the device reports no error is counted as a
silent error. The final verdict separates three cases:

  silent-data-corruption   some runs differ from the first run (intermittent)
  outside-error-bound      all runs agree but sit outside the proven bound
  nondeterministic-kernel  many runs differ but all stay inside the bound, so
                           the repeat check cannot be used on this device/dtype

--inject N flips one random bit in N chosen runs (never the first run) to prove
the checks catch corruption. In that mode the verdict reports detection.

Output: OCP Test & Validation JSON, one object per line, written to --out.
A short summary is printed to the terminal.
"""

import argparse
import os
import platform
import random
import socket
import sys
import time

# must be set before CUDA initialises so cuBLAS runs deterministically
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import ocptv.output as tv  # noqa: E402

VERSION = "0.1.0"

DTYPES = {
    "fp32": (torch.float32, torch.int32, 32, 2.0 ** -24),
    "fp16": (torch.float16, torch.int16, 16, 2.0 ** -11),
    "bf16": (torch.bfloat16, torch.int16, 16, 2.0 ** -8),
}
U_ACC = 2.0 ** -24          # accumulation assumed in FP32 or better (enforced on CUDA below)
NONDET_SHARE = 0.10         # at least this share of differing runs (and at least NONDET_MIN runs)
NONDET_MIN = 3              # reads as a non-deterministic kernel rather than a rare fault
_FAULT_HOOK = None          # tests only: callable(run_index, cpu_tensor) that corrupts an output in place


class FileWriter(tv.Writer):
    """Writes each OCP JSON object as one line of a file."""

    def __init__(self, path):
        self._f = open(path, "w", encoding="utf-8")

    def write(self, buffer):
        self._f.write(buffer.rstrip("\n") + "\n")
        self._f.flush()

    def close(self):
        self._f.close()


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_label(dev):
    if dev.type == "cuda":
        return torch.cuda.get_device_name(dev)
    if dev.type == "mps":
        return "Apple GPU (MPS) " + platform.machine()
    return "CPU " + (platform.processor() or platform.machine())


def lock_down_math(dev):
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_float32_matmul_precision("highest")
    if dev.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False


def make_case(n, dtype, seed):
    """Inputs rounded to the test dtype, plus the float64 reference and its bound."""
    g = torch.Generator().manual_seed(seed)
    a = (torch.rand(n, n, generator=g, dtype=torch.float64) * 2 - 1).to(dtype)
    b = (torch.rand(n, n, generator=g, dtype=torch.float64) * 2 - 1).to(dtype)
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    ref = a64 @ b64
    abs_ab = a64.abs() @ b64.abs()
    return a, b, ref, abs_ab


def error_bound(ref, abs_ab, k, u_out):
    """|C - R| <= g*|A||B| + u_out*(|R| + g*|A||B|), with g = k*u/(1-k*u)."""
    g = k * U_ACC / (1 - k * U_ACC)
    acc = g * abs_ab
    return acc + u_out * (ref.abs() + acc)


def flip_bit(c, int_view, bits, rng):
    """Flip one random bit of one random element (CPU tensor, in place)."""
    flat = c.view(int_view).view(-1)
    idx = rng.randrange(flat.numel())
    bit = rng.randrange(bits)
    mask = torch.tensor(1 << bit if bit < bits - 1 else -(1 << bit), dtype=int_view)
    flat[idx] = flat[idx] ^ mask
    return idx, bit


def screen_dtype(step, dev, name, n, iters, seed, inject, rng, hw):
    dtype, int_view, bits, u_out = DTYPES[name]
    a, b, ref, abs_ab = make_case(n, dtype, seed)
    bound = error_bound(ref, abs_ab, n, u_out)
    a_dev, b_dev = a.to(dev), b.to(dev)

    inject_runs = set(rng.sample(range(1, iters), min(inject, iters - 1))) if inject else set()
    first_bits = None
    ref_bad, rep_bad = [], []
    injected, caught = 0, 0
    worst_ratio, worst_abs, worst_nonfinite = 0.0, 0.0, 0
    t0 = time.time()

    for i in range(iters):
        c = (a_dev @ b_dev).to("cpu").contiguous()
        if _FAULT_HOOK is not None:
            _FAULT_HOOK(i, c)
        if i in inject_runs:
            flip_bit(c, int_view, bits, rng)
            injected += 1

        c64 = c.to(torch.float64)
        finite = torch.isfinite(c64)
        err = (c64 - ref).abs()
        nonfinite = int((~finite).sum())
        worst_nonfinite = max(worst_nonfinite, nonfinite)
        if bool(finite.any()):
            worst_ratio = max(worst_ratio, float((err[finite] / bound[finite]).max()))
            worst_abs = max(worst_abs, float(err[finite].max()))
        ref_fail = nonfinite > 0 or bool((err[finite] > bound[finite]).any())

        bits_now = c.view(int_view).clone()
        if first_bits is None:
            first_bits = bits_now
            rep_fail = False
        else:
            rep_fail = not torch.equal(bits_now, first_bits)

        if ref_fail:
            ref_bad.append(i)
        if rep_fail:
            rep_bad.append(i)
        if i in inject_runs and (ref_fail or rep_fail):
            caught += 1

    secs = time.time() - t0

    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    step.add_measurement(name="runs", value=iters, hardware_info=hw)
    step.add_measurement(name="matrix_size", value=n, hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    step.add_measurement(name="reference_check_failed_runs", value=len(ref_bad),
                         validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="repeat_check_failed_runs", value=len(rep_bad),
                         validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="worst_error_over_bound_ratio", value=worst_ratio, hardware_info=hw)
    step.add_measurement(name="worst_abs_error", value=worst_abs, hardware_info=hw)
    step.add_measurement(name="worst_nonfinite_values", value=worst_nonfinite, hardware_info=hw)

    if inject:
        step.add_measurement(name="injected_runs", value=injected, hardware_info=hw)
        step.add_measurement(name="injected_runs_detected", value=caught,
                             validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=injected)],
                             hardware_info=hw)
        clean_bad = (set(ref_bad) | set(rep_bad)) - inject_runs
        ok = caught == injected and not clean_bad
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg = f"{caught}/{injected} injected runs detected; {len(clean_bad)} unexpected failing runs"
    elif not ref_bad and not rep_bad:
        ok, verdict, msg = True, "no-silent-errors", f"{iters} runs, all inside bound and bit-identical"
    elif rep_bad and not ref_bad and len(rep_bad) >= max(NONDET_MIN, NONDET_SHARE * iters):
        ok, verdict = False, "nondeterministic-kernel"
        msg = f"{len(rep_bad)}/{iters} runs differ but all stay inside bound; repeat check unusable here"
    elif rep_bad:
        ok, verdict = False, "silent-data-corruption"
        msg = f"runs differing from first run: {rep_bad[:20]}; outside bound: {ref_bad[:20]}"
    else:
        ok, verdict = False, "outside-error-bound"
        msg = (f"all runs agree but {len(ref_bad)} runs sit outside the proven bound "
               f"(a faulty unit, or the device accumulating below FP32)")

    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL,
                       verdict=verdict, message=msg, hardware_info=hw)
    return ok, verdict, msg, secs


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite screener")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--size", type=int, default=1024, help="square matrix size")
    p.add_argument("--iters", type=int, default=50, help="runs per dtype")
    p.add_argument("--dtypes", default="fp32,fp16", help="comma list from fp32,fp16,bf16")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="bit flips to inject per dtype (self-test)")
    p.add_argument("--out", default="results.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    bad = [d for d in names if d not in DTYPES]
    if bad or args.iters < 2 or args.size < 2:
        p.error(f"check --dtypes {bad}, --iters >= 2, --size >= 2")

    dev = pick_device(args.device)
    lock_down_math(dev)
    rng = random.Random(args.seed)

    writer = FileWriter(args.out)
    tv.config(writer=writer)

    run = tv.TestRun(
        name="ai-chip-integrity-screen",
        version=VERSION,
        parameters={"device": str(dev), "size": args.size, "iters": args.iters,
                    "dtypes": ",".join(names), "seed": args.seed, "inject": args.inject},
    )
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=device_label(dev))
    dut.add_software_info(name="torch", version=torch.__version__)
    dut.add_software_info(name="python", version=platform.python_version())

    print(f"Device: {device_label(dev)} ({dev})  size={args.size}  runs={args.iters}  inject={args.inject}")
    all_ok = True
    run.start(dut=dut)
    try:
        for name in names:
            step = run.add_step(f"gemm_{name}")
            step.start()
            try:
                ok, verdict, msg, secs = screen_dtype(step, dev, name, args.size, args.iters,
                                                     args.seed, args.inject, rng, hw)
                step.end(status=tv.TestStatus.COMPLETE)
            except (RuntimeError, TypeError) as e:
                step.add_error(symptom="dtype-not-supported", message=str(e)[:500])
                step.end(status=tv.TestStatus.SKIP)
                print(f"  {name:5s}  SKIPPED  {str(e)[:120]}")
                continue
            all_ok &= ok
            print(f"  {name:5s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
    finally:
        run.end(status=tv.TestStatus.COMPLETE,
                result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()

    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
