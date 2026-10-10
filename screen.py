#!/usr/bin/env python3
"""
screen.py  -  AI Chip Integrity Suite, screener v0.1.0

Runs fixed matrix multiplies (several shapes and precisions, including exact
int8) many times on a single device and checks every result two ways:

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

import arith  # noqa: E402

VERSION = "0.3.0"

FP8 = getattr(torch, "float8_e4m3fn", None)   # E4M3, the FP8 format inference engines run on

DTYPES = {
    "fp32": (torch.float32, torch.float32, torch.int32, 32, 2.0 ** -24),
    "fp16": (torch.float16, torch.float16, torch.int16, 16, 2.0 ** -11),
    "bf16": (torch.bfloat16, torch.bfloat16, torch.int16, 16, 2.0 ** -8),
    "int8": (torch.int8, torch.int32, torch.int32, 32, None),   # exact: any difference is a fault
    # FP8 tensor cores (Ada, Hopper, Blackwell) with FP32 output. Inputs come from {-1, 0, 1}, so every product is
    # exact and every partial sum is an integer no larger than K in size: with K up to 4096 it fits in 13
    # significant bits. Hopper's FP8 tensor cores are reported to keep 14 bits when they add (DeepSeek-V3
    # technical report, 2024), so the answer is exact even with fast accumulation, and any difference is a
    # fault, or an accumulator narrower than 13 bits.
    "fp8": (FP8, torch.float32, torch.int32, 32, None),
    "fp8fast": (FP8, torch.float32, torch.int32, 32, None),      # same inputs, fast (reduced-precision) accumulation
}
FP8_EXACT_K = 4096          # largest K for which the FP8 partial sums fit in 13 significant bits
DEFAULT_SHAPES = "1024x1024x1024,4096x4096x4096,32x4096x11008"   # MxKxN; last one is an LLM decode shape
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


def make_case(m, k, n, dtype, seed):
    """Inputs rounded to the test dtype, plus the exact (float64 or int64) reference and |A|@|B|."""
    g = torch.Generator().manual_seed(seed)
    if dtype is None:
        raise RuntimeError("this PyTorch build has no float8_e4m3fn type")
    if dtype == FP8:
        a = torch.randint(-1, 2, (m, k), generator=g, dtype=torch.int8)
        b = torch.randint(-1, 2, (k, n), generator=g, dtype=torch.int8)
        ref = (a.to(torch.int64) @ b.to(torch.int64)).to(torch.float64)   # exact
        return a.to(torch.float32).to(FP8), b.to(torch.float32).to(FP8), ref, torch.zeros_like(ref)
    if dtype == torch.int8:
        a = torch.randint(-16, 16, (m, k), generator=g, dtype=torch.int8)
        b = torch.randint(-16, 16, (k, n), generator=g, dtype=torch.int8)
        ref = (a.to(torch.int64) @ b.to(torch.int64)).to(torch.float64)   # exact, |value| < 2^31
        return a, b, ref, torch.zeros_like(ref)
    a = (torch.rand(m, k, generator=g, dtype=torch.float64) * 2 - 1).to(dtype)
    b = (torch.rand(k, n, generator=g, dtype=torch.float64) * 2 - 1).to(dtype)
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return a, b, a64 @ b64, a64.abs() @ b64.abs()


def flush_products(a, b):
    """sum_k |a_ik b_kj| over products with a subnormal input: the most a flushing device can lose."""
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    return arith.flush_term(a64, a.dtype) @ b64.abs() + a64.abs() @ arith.flush_term(b64, b.dtype)


def parse_shapes(text):
    out = []
    for item in text.split(","):
        item = item.strip().lower()
        if not item:
            continue
        parts = item.split("x")
        if len(parts) == 1:
            parts = parts * 3
        if len(parts) != 3 or not all(p.isdigit() and int(p) >= 2 for p in parts):
            raise ValueError(f"bad shape {item!r}: use MxKxN, e.g. 1024x1024x1024")
        out.append(tuple(int(p) for p in parts))
    return out


def matmul(a, b, fast_accum=False):
    if a.dtype == torch.int8:
        return torch._int_mm(a, b)
    if FP8 is not None and a.dtype == FP8:
        one = torch.ones((), dtype=torch.float32, device=a.device)
        r = torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.float32, use_fast_accum=fast_accum)
        return r[0] if isinstance(r, tuple) else r       # PyTorch before 2.4 also returned amax
    return a @ b


def accumulation_factor(name, k, dev_type):
    """g in the bound below, from the arithmetic model in arith.py.

    fp32 on CUDA (TF32 off) and on CPU runs on IEEE FP32 fused multiply-add: gamma(k, 2^-24).
    Everything else may run on matrix units, which can truncate and add in blocks: gamma(2k, 2^-23),
    plus the product allowance for FP32 inputs on devices where we cannot rule out emulation."""
    if name == "fp32" and dev_type in ("cuda", "cpu"):
        return arith.ieee_dot(k)
    return arith.matrix_dot(k, fp32_inputs=(name == "fp32"))


def error_bound(ref, abs_ab, k, u_out, g=None, flush_ab=None, out_floor=0.0):
    """|C - R| <= acc + u_out*(|R| + acc) + floor, acc = g*|A||B| (+ products with a subnormal input).
    Integers: exact (zero)."""
    if u_out is None:
        return torch.zeros_like(ref)
    if g is None:
        g = arith.ieee_dot(k)
    acc = g * abs_ab
    if flush_ab is not None:
        acc = acc + flush_ab
    return acc + u_out * (ref.abs() + acc) + out_floor


def flip_bit(c, int_view, bits, rng):
    """Flip one random bit of one random element (CPU tensor, in place)."""
    flat = c.view(int_view).view(-1)
    idx = rng.randrange(flat.numel())
    bit = rng.randrange(bits)
    mask = torch.tensor(1 << bit if bit < bits - 1 else -(1 << bit), dtype=int_view)
    flat[idx] = flat[idx] ^ mask
    return idx, bit


def screen_case(step, dev, name, shape, iters, seed, inject, rng, hw):
    dtype, out_dtype, int_view, bits, u_out = DTYPES[name]
    m, k, n = shape
    a, b, ref, abs_ab = make_case(m, k, n, dtype, seed)
    exact = u_out is None
    g = None if exact else accumulation_factor(name, k, dev.type)
    bound = error_bound(ref, abs_ab, k, u_out, g=g,
                        flush_ab=None if exact else flush_products(a, b),
                        out_floor=0.0 if exact else arith.subnormal_floor(out_dtype))
    if dtype == FP8 and k > FP8_EXACT_K:
        raise RuntimeError(f"FP8 exact check needs K <= {FP8_EXACT_K}; got {k}")
    a_dev, b_dev = a.to(dev), b.to(dev)
    if dtype == FP8:
        b_dev = b_dev.t().contiguous().t()                 # the FP8 kernels take B column-major
    fast = name == "fp8fast"

    inject_runs = set(rng.sample(range(1, iters), min(inject, iters - 1))) if inject else set()
    first_bits = None
    ref_bad, rep_bad = [], []
    injected, caught = 0, 0
    worst_ratio, worst_abs, worst_nonfinite = 0.0, 0.0, 0
    t0 = time.time()

    for i in range(iters):
        c = matmul(a_dev, b_dev, fast).to("cpu").contiguous()
        if c.dtype != out_dtype:
            raise RuntimeError(f"device returned {c.dtype}, expected {out_dtype}")
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
            worst_abs = max(worst_abs, float(err[finite].max()))
            if not exact:
                worst_ratio = max(worst_ratio, float((err[finite] / bound[finite]).max()))
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
    step.add_measurement(name="shape_mkn", value=f"{m}x{k}x{n}", hardware_info=hw)
    step.add_measurement(name="exact_check", value=exact, hardware_info=hw)
    if not exact:
        step.add_measurement(name="accumulation_factor", value=g, hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    step.add_measurement(name="reference_check_failed_runs", value=len(ref_bad),
                         validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="repeat_check_failed_runs", value=len(rep_bad),
                         validators=None if inject else eq0, hardware_info=hw)
    if not exact:
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
        ok, verdict = True, "no-silent-errors"
        msg = f"{iters} runs, all {'exact' if exact else 'inside bound'} and bit-identical"
    elif rep_bad and not ref_bad and len(rep_bad) >= max(NONDET_MIN, NONDET_SHARE * iters):
        ok, verdict = False, "nondeterministic-kernel"
        msg = f"{len(rep_bad)}/{iters} runs differ but all stay inside bound; repeat check unusable here"
    elif rep_bad:
        ok, verdict = False, "silent-data-corruption"
        msg = f"runs differing from first run: {rep_bad[:20]}; outside bound: {ref_bad[:20]}"
    else:
        ok, verdict = False, "outside-error-bound"
        msg = (f"all runs agree but {len(ref_bad)} runs sit outside the proven bound "
               + ("(integer result differs from the exact answer)" if exact
                  else "(a faulty unit, the device accumulating below FP32, or TF32 used for fp32)"))
        if dtype == FP8:
            msg += "; for FP8 the answer is exact unless an accumulator holds fewer than 13 significant bits"

    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL,
                       verdict=verdict, message=msg, hardware_info=hw)
    return ok, verdict, msg, secs


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite screener")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--shapes", default=None, help=f"comma list of MxKxN (default {DEFAULT_SHAPES})")
    p.add_argument("--size", type=int, default=None, help="shortcut: one square NxNxN shape")
    p.add_argument("--iters", type=int, default=50, help="runs per shape and precision")
    p.add_argument("--dtypes", default="fp32,fp16,bf16,int8,fp8,fp8fast", help="comma list from fp32,fp16,bf16,int8,fp8,fp8fast")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="bit flips to inject per step (self-test)")
    p.add_argument("--out", default="results.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    bad = [d for d in names if d not in DTYPES]
    if bad or args.iters < 2:
        p.error(f"check --dtypes {bad}, --iters >= 2")
    try:
        shapes = parse_shapes(args.shapes or (f"{args.size}" if args.size else DEFAULT_SHAPES))
    except ValueError as e:
        p.error(str(e))

    dev = pick_device(args.device)
    lock_down_math(dev)
    rng = random.Random(args.seed)

    writer = FileWriter(args.out)
    tv.config(writer=writer)

    run = tv.TestRun(
        name="ai-chip-integrity-screen",
        version=VERSION,
        parameters={"device": str(dev), "shapes": ",".join("x".join(map(str, sh)) for sh in shapes),
                    "iters": args.iters, "dtypes": ",".join(names), "seed": args.seed,
                    "inject": args.inject, "bound_model": arith.MODEL_ID},
    )
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=device_label(dev))
    dut.add_software_info(name="torch", version=torch.__version__)
    dut.add_software_info(name="python", version=platform.python_version())

    print(f"Device: {device_label(dev)} ({dev})  runs={args.iters}  inject={args.inject}")
    all_ok = True
    run.start(dut=dut)
    try:
        for shape in shapes:
            label = "x".join(map(str, shape))
            for name in names:
                step = run.add_step(f"gemm_{name}_{label}")
                step.start()
                try:
                    ok, verdict, msg, secs = screen_case(step, dev, name, shape, args.iters,
                                                         args.seed, args.inject, rng, hw)
                    step.end(status=tv.TestStatus.COMPLETE)
                except (RuntimeError, TypeError, NotImplementedError) as e:
                    step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                    step.end(status=tv.TestStatus.SKIP)
                    print(f"  {label:>16s} {name:7s}  SKIPPED  {str(e)[:100]}")
                    continue
                all_ok &= ok
                print(f"  {label:>16s} {name:7s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
    finally:
        run.end(status=tv.TestStatus.COMPLETE,
                result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()

    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
