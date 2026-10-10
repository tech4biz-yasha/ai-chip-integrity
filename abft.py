#!/usr/bin/env python3
"""
abft.py  -  AI Chip Integrity Suite, checksum-protected matrix multiply (probe 6) v0.1.0

Huang and Abraham's algorithm-based fault tolerance (IEEE Transactions on Computers, 1984): append to A a row
holding its column sums and to B a column holding its row sums, and one multiply returns C together with C's row
sums and column sums, computed by the same hardware in the same pass. A wrong element then breaks exactly one row
check and one column check, which also says where it is. No reference answer is needed, so the same check can run
inside production work at the cost of one extra row and column, and it is the design that later becomes a block in
silicon.

The weak point of ABFT in floating point is the threshold: rounding makes a correct row sum differ from its
checksum, and a threshold picked by hand either raises false alarms or hides real faults. Here every threshold is
derived, element by element, from the same arithmetic model as the compute probe (arith.py): the bound of every
element in the row, the bound of the checksum element, the float64 host sum, and the exact rounding error of the
checksum vectors, which the host knows and removes. A correct device can never cross it.

Each run is also checked against the float64 reference and bit for bit against run 0, as in the compute probe, and
the probe counts runs where the reference check failed but the checksums held: ABFT's blind spots, measured.

--inject N adds, in N runs, an error four times the checksum threshold to one random element of C after it is
copied back. The checksums must flag exactly that row and that column, so the self-test proves location as well as
detection. Output: OCP Test and Validation JSON to --out.
"""

import argparse
import math
import platform
import random
import socket
import sys
import time

import torch
import ocptv.output as tv

import arith
import counters
import screen

VERSION = "0.1.0"
DTYPES = ["fp32", "fp16", "bf16"]
DEFAULT_SHAPES = "1024x1024x1024,32x4096x11008"
F64 = 2.0 ** -53


def checksum_scale(vec_abs, other_abs, dtype, out_dtype):
    """Power-of-two exponent s so that 2^-s times a checksum vector stays well inside the input type and its
    products stay well inside the output type and an FP32 accumulator. A checksum is a sum of m or n values, so
    without this it overflows whenever the values are anywhere near the top of their range."""
    top_out = 0.5 * min(torch.finfo(out_dtype).max, torch.finfo(torch.float32).max)
    top_in = 0.5 * torch.finfo(dtype).max
    need = max(float(vec_abs.max()) / top_in, float(other_abs.max()) / top_out, 1.0)
    return max(0, math.ceil(math.log2(need)))


def augment(a, b):
    """A with a column-checksum row and B with a row-checksum column, each scaled by a power of two to stay in
    range and rounded to the input type, plus the exact rounding error of each checksum vector, the float64
    error of computing it, and the two scale exponents."""
    dtype, out_dtype = a.dtype, screen.DTYPES[{torch.float32: "fp32", torch.float16: "fp16", torch.bfloat16: "bf16"}[a.dtype]][1]
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    m, n = a.shape[0], b.shape[1]
    col_sum = a64.sum(dim=0, keepdim=True)
    row_sum = b64.sum(dim=1, keepdim=True)
    sc = checksum_scale(col_sum.abs(), col_sum.abs() @ b64.abs(), dtype, out_dtype)
    sr = checksum_scale(row_sum.abs(), a64.abs() @ row_sum.abs(), dtype, out_dtype)
    col, row = col_sum * 2.0 ** -sc, row_sum * 2.0 ** -sr                       # exact: powers of two
    col_r, row_r = col.to(dtype), row.to(dtype)
    eps_col = arith.gamma(m, F64) * a64.abs().sum(dim=0, keepdim=True) * 2.0 ** -sc
    eps_row = arith.gamma(n, F64) * b64.abs().sum(dim=1, keepdim=True) * 2.0 ** -sr
    return (torch.cat((a, col_r), dim=0), torch.cat((b, row_r), dim=1),
            col_r.to(torch.float64) - col, row_r.to(torch.float64) - row, eps_col, eps_row, sc, sr)


def thresholds(a, b, ac, br, d_col, d_row, eps_col, eps_row, name, k, dev_type, sc=0, sr=0):
    """Per-row and per-column limits on |sum of computed C - computed checksum - known offset|, plus the element
    bound of the augmented product, the offsets, and the float64 reference of the C block."""
    dtype, out_dtype, _, _, u_out = screen.DTYPES[name]
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    ac64, br64 = ac.to(torch.float64), br.to(torch.float64)
    ref = ac64 @ br64
    beta = screen.error_bound(ref, ac64.abs() @ br64.abs(), k, u_out, g=screen.accumulation_factor(name, k, dev_type),
                              flush_ab=screen.flush_products(ac, br), out_floor=arith.subnormal_floor(out_dtype))
    m, n = a.shape[0], b.shape[1]
    c_abs = ref[:m, :n].abs() + beta[:m, :n]                          # most any computed C element can be
    # exact relations: 2^-sr sum_j C_ij - r_i = -(A d_row)_i and 2^-sc sum_i C_ij - c_j = -(d_col B)_j
    off_row = -(a64 @ d_row).squeeze(1)
    off_col = -(d_col @ b64).squeeze(0)
    slack_row = arith.gamma(k, F64) * (a64.abs() @ d_row.abs()).squeeze(1) + (a64.abs() @ eps_row).squeeze(1)
    slack_col = arith.gamma(k, F64) * (d_col.abs() @ b64.abs()).squeeze(0) + (eps_col @ b64.abs()).squeeze(0)
    thr_row = ((beta[:m, :n].sum(dim=1) + arith.gamma(n, F64) * c_abs.sum(dim=1)) * 2.0 ** -sr
               + beta[:m, n] + slack_row) * (1 + 4 * F64)
    thr_col = ((beta[:m, :n].sum(dim=0) + arith.gamma(m, F64) * c_abs.sum(dim=0)) * 2.0 ** -sc
               + beta[m, :n] + slack_col) * (1 + 4 * F64)
    return thr_row, thr_col, off_row, off_col, beta, ref


def check(cf, m, n, thr_row, thr_col, off_row, off_col, sc=0, sr=0):
    """Row and column residuals of one computed product. Returns flagged rows, flagged columns and the worst
    residual-to-threshold ratios."""
    c64 = cf.to(torch.float64)
    res_row = (c64[:m, :n].sum(dim=1) * 2.0 ** -sr - c64[:m, n] - off_row).abs()
    res_col = (c64[:m, :n].sum(dim=0) * 2.0 ** -sc - c64[m, :n] - off_col).abs()
    finite = torch.isfinite(c64).all()
    rows = torch.nonzero(~(res_row <= thr_row)).flatten().tolist()     # NaN counts as flagged
    cols = torch.nonzero(~(res_col <= thr_col)).flatten().tolist()
    worst_r = float((res_row / thr_row).nan_to_num(float("inf")).max())
    worst_c = float((res_col / thr_col).nan_to_num(float("inf")).max())
    return rows, cols, worst_r, worst_c, bool(finite)


def abft_case(step, dev, name, shape, iters, seed, inject, rng, hw, case=None):
    dtype, out_dtype, int_view, bits, u_out = screen.DTYPES[name]
    m, k, n = shape
    a, b, _, _ = case if case is not None else screen.make_case(m, k, n, dtype, seed)
    ac, br, d_col, d_row, eps_col, eps_row, sc, sr = augment(a, b)
    thr_row, thr_col, off_row, off_col, beta, ref = thresholds(a, b, ac, br, d_col, d_row, eps_col, eps_row, name, k,
                                                               dev.type, sc, sr)
    a_dev, b_dev = ac.to(dev), br.to(dev)

    inject_runs = set(rng.sample(range(1, iters), min(inject, iters - 1))) if inject else set()
    first_bits = None
    abft_bad, ref_bad, rep_bad, missed = [], [], [], []
    located, injected = 0, 0
    first_location = ""
    worst_r = worst_c = 0.0
    t0 = time.time()
    for i in range(iters):
        cf = screen.matmul(a_dev, b_dev).to("cpu").contiguous()
        if cf.dtype != out_dtype:
            raise RuntimeError(f"device returned {cf.dtype}, expected {out_dtype}")
        if screen._FAULT_HOOK is not None:
            screen._FAULT_HOOK(i, cf)
        target = None
        if i in inject_runs:
            ti, tj = rng.randrange(m), rng.randrange(n)
            delta = 4.0 * max(float(thr_row[ti]) * 2.0 ** sr, float(thr_col[tj]) * 2.0 ** sc)
            cf[ti, tj] = (cf[ti, tj].to(torch.float64) + delta).to(out_dtype)
            target = (ti, tj)
            injected += 1
        rows, cols, wr, wc, finite = check(cf, m, n, thr_row, thr_col, off_row, off_col, sc, sr)
        worst_r, worst_c = max(worst_r, wr), max(worst_c, wc)
        c64 = cf.to(torch.float64)
        ref_fail = (not finite) or bool(((c64 - ref).abs() > beta).any())
        abft_fail = bool(rows or cols) or not finite
        bits_now = cf.view(int_view).clone()
        rep_fail = first_bits is not None and not torch.equal(bits_now, first_bits)
        if first_bits is None:
            first_bits = bits_now
        if abft_fail:
            abft_bad.append(i)
            if not first_location and target is None:
                first_location = f"rows {rows[:5]}, columns {cols[:5]}"
        if ref_fail:
            ref_bad.append(i)
        if rep_fail:
            rep_bad.append(i)
        if ref_fail and not abft_fail and target is None:
            missed.append(i)
        if target is not None and rows == [target[0]] and cols == [target[1]]:
            located += 1
    secs = time.time() - t0

    typical = float(ref[:m, :n].abs().median())
    thr_med = float(thr_row.median()) * 2.0 ** sr               # in the units of C
    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    step.add_measurement(name="runs", value=iters, hardware_info=hw)
    step.add_measurement(name="shape_mkn", value=f"{m}x{k}x{n}", hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    step.add_measurement(name="flop_overhead", value=round((m + 1) * (n + 1) / (m * n) - 1, 6), hardware_info=hw)
    step.add_measurement(name="checksum_scale_power_of_two", value=max(sc, sr), hardware_info=hw)
    step.add_measurement(name="row_threshold_median", value=thr_med, hardware_info=hw)
    step.add_measurement(name="typical_value_median", value=typical, hardware_info=hw)
    step.add_measurement(name="threshold_over_typical_value", value=thr_med / typical if typical else 0.0, hardware_info=hw)
    step.add_measurement(name="worst_row_residual_over_threshold", value=worst_r, hardware_info=hw)
    step.add_measurement(name="worst_column_residual_over_threshold", value=worst_c, hardware_info=hw)
    step.add_measurement(name="checksum_check_failed_runs", value=len(abft_bad), validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="reference_check_failed_runs", value=len(ref_bad), validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="repeat_check_failed_runs", value=len(rep_bad), validators=None if inject else eq0, hardware_info=hw)
    step.add_measurement(name="checksum_missed_runs", value=len(missed), hardware_info=hw)
    if inject:
        step.add_measurement(name="injected_runs", value=injected, hardware_info=hw)
        step.add_measurement(name="injected_runs_located", value=located,
                             validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=injected)], hardware_info=hw)
        unexpected = (set(abft_bad) | set(ref_bad) | set(rep_bad)) - inject_runs
        ok = located == injected and not unexpected
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg = f"{located}/{injected} injected errors located to their exact row and column; {len(unexpected)} unexpected failing runs"
    elif not abft_bad and not ref_bad and not rep_bad:
        ok, verdict = True, "no-silent-errors"
        msg = f"{iters} runs, every checksum consistent, inside the reference bound and bit-identical"
    elif rep_bad and not abft_bad and not ref_bad and len(rep_bad) >= max(screen.NONDET_MIN, screen.NONDET_SHARE * iters):
        ok, verdict = False, "nondeterministic-kernel"
        msg = f"{len(rep_bad)}/{iters} runs differ but every checksum and bound holds; repeat check unusable here"
    elif rep_bad:
        ok, verdict = False, "silent-data-corruption"
        msg = f"runs differing from run 0: {rep_bad[:20]}; checksum failures: {abft_bad[:20]} {first_location}".rstrip()
    else:
        ok, verdict = False, "outside-error-bound"
        msg = f"all runs agree but checksums or the reference fail: {first_location or 'reference only'}"
    if missed and not inject:
        msg += f"; {len(missed)} runs failed the reference check with consistent checksums (ABFT blind spot)"
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg, hardware_info=hw)
    return ok, verdict, msg, secs


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite checksum-protected matrix multiply")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma list of MxKxN")
    p.add_argument("--iters", type=int, default=20, help="runs per shape and precision")
    p.add_argument("--dtypes", default=",".join(DTYPES), help="comma list from fp32,fp16,bf16")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="errors to inject per step (self-test)")
    p.add_argument("--out", default="abft.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)
    try:
        shapes = screen.parse_shapes(args.shapes)
    except ValueError as e:
        p.error(str(e))
    names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if any(d not in DTYPES for d in names) or args.iters < 2:
        p.error("check --dtypes (fp32, fp16, bf16) and --iters >= 2")

    dev = screen.pick_device(args.device)
    screen.lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = screen.FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(name="ai-chip-integrity-abft", version=VERSION,
                     parameters={"device": str(dev), "shapes": ",".join("x".join(map(str, s)) for s in shapes),
                                 "iters": args.iters, "dtypes": ",".join(names), "seed": args.seed,
                                 "inject": args.inject, "bound_model": arith.MODEL_ID})
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=screen.device_label(dev))
    dut.add_software_info(name="torch", version=torch.__version__)
    dut.add_software_info(name="python", version=platform.python_version())

    print(f"Device: {screen.device_label(dev)} ({dev})  runs={args.iters}  inject={args.inject}")
    all_ok = True
    run.start(dut=dut)
    before = counters.snapshot(dev)
    try:
        for shape in shapes:
            label = "x".join(map(str, shape))
            for name in names:
                step = run.add_step(f"abft_{name}_{label}")
                step.start()
                try:
                    ok, verdict, msg, secs = abft_case(step, dev, name, shape, args.iters, args.seed, args.inject, rng, hw)
                    step.end(status=tv.TestStatus.COMPLETE)
                except (RuntimeError, TypeError, NotImplementedError) as e:
                    step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                    step.end(status=tv.TestStatus.SKIP)
                    print(f"  {label:>16s} {name:5s}  SKIPPED  {str(e)[:100]}")
                    continue
                all_ok &= ok
                print(f"  {label:>16s} {name:5s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
    finally:
        try:
            all_ok &= counters.record(run, hw, before, counters.snapshot(dev), probe_ok=all_ok, injected=args.inject > 0)
        except Exception as e:
            print(f"  error counters could not be recorded: {e}")
        run.end(status=tv.TestStatus.COMPLETE, result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()
    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
