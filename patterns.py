#!/usr/bin/env python3
"""
patterns.py  -  AI Chip Integrity Suite, data pattern probe v0.1.0

A fault that only fires on certain bit patterns never shows up on uniform random inputs. This probe runs the
compute probe's matrix multiply, with the same two checks and the same proven or exact reference, on inputs
chosen to drive specific parts of the arithmetic:

  wide       random exponents across many binades: the alignment shifters and exponent logic
  mantissa   every mantissa bit set: the longest carry chains in multipliers and adders
  cancel     A = [Y Y] and B = [X; -X]: the exact answer is zero, so only rounding may remain
  alternate  checkerboard signs, so every product in a dot product has the same sign and sums grow largest
  sparse     90% exact zeros: zero handling and skipping
  subnormal  A below the smallest normal number of its type, B large enough that every product is normal: the
             subnormal input path. The probe first measures whether the device flushes subnormal inputs to
             zero, then judges it against the exact model that matches, with no allowance for the other
  near_max   scaled so the largest possible partial sum is a quarter to a half of the output type's maximum
  extremes   int8 only: -128, -127, 0 and 127, the corners of the integer multiplier

Every input set is also scaled by a power of two, when needed, so that no partial sum in any order can reach
the output type's range. FP8 keeps its inputs in {-1, 0, 1} so its answers stay exact, so it runs alternate
and sparse only.

--inject N flips one random bit in N runs of every step, as in the compute probe. Output: OCP Test and
Validation JSON to --out.
"""

import argparse
import math
import platform
import random
import socket
import sys

import torch
import ocptv.output as tv

import arith
import counters
import screen

VERSION = "0.1.1"
FLOAT_PATTERNS = ["wide", "mantissa", "cancel", "alternate", "sparse", "subnormal", "near_max"]
APPLIES = {
    "fp32": FLOAT_PATTERNS, "fp16": FLOAT_PATTERNS, "bf16": FLOAT_PATTERNS,
    "int8": ["extremes", "alternate", "sparse"],
    "fp8": ["alternate", "sparse"], "fp8fast": ["alternate", "sparse"],
}
ALL_PATTERNS = FLOAT_PATTERNS + ["extremes"]
SIGNIFICAND_BITS = {torch.float32: 24, torch.float16: 11, torch.bfloat16: 8}
WIDE_EXPONENTS = {torch.float32: (-40, 40), torch.float16: (-7, 7), torch.bfloat16: (-40, 40)}
SUBNORMAL_LIFT = {torch.float32: 100, torch.float16: 10, torch.bfloat16: 100}   # B's exponent: products stay normal
DEFAULT_SHAPES = "1024x1024x1024,32x4096x11008"


def flushes_subnormal_inputs(dev, name):
    """Does this device and precision flush subnormal matmul inputs to zero? One small product decides it."""
    dtype = screen.DTYPES[name][0]
    a = torch.full((16, 16), torch.finfo(dtype).tiny / 4, dtype=torch.float64).to(dtype)
    b = torch.full((16, 16), 2.0 ** SUBNORMAL_LIFT[dtype], dtype=torch.float64).to(dtype)
    c = screen.matmul(a.to(dev), b.to(dev)).to("cpu").to(torch.float64)
    return bool(float(c.abs().max()) == 0.0)


def _u(g, shape):
    return torch.rand(*shape, generator=g, dtype=torch.float64)


def _signs(g, shape):
    return torch.where(_u(g, shape) < 0.5, -1.0, 1.0).to(torch.float64)


def _checker(rows, cols):
    i = torch.arange(rows)[:, None]
    j = torch.arange(cols)[None, :]
    return torch.where((i + j) % 2 == 0, 1.0, -1.0).to(torch.float64)


def range_limit(dtype, out_dtype):
    """Largest partial sum allowed: half of what the output type and an FP32 accumulator can hold."""
    top = min(torch.finfo(out_dtype).max, torch.finfo(torch.float32).max)
    return 0.5 * top


def fit_range(a, b, dtype, out_dtype, grow):
    """Scale A and B by powers of two so the largest possible partial sum, max(|A| @ |B|), sits between a
    quarter and a half of the output type's maximum when growing (near_max), or at most half of it otherwise,
    so no partial sum in any order can overflow. Powers of two keep normal values exact, and the reference is
    always taken from the rounded result."""
    limit = range_limit(dtype, out_dtype)
    s = float((a.abs() @ b.abs()).max())
    if s == 0.0:
        return a, b, 0
    shift = math.floor(math.log2(limit / s))
    if not grow:
        shift = min(0, shift)
    if shift == 0:
        return a, b, 0
    sa = shift // 2 + shift % 2
    return a * 2.0 ** sa, b * 2.0 ** (shift - sa), shift


def float_inputs(pattern, m, k, n, dtype, out_dtype, g):
    if pattern == "wide":
        lo, hi = WIDE_EXPONENTS[dtype]

        def gen(shape):
            e = torch.randint(lo, hi + 1, shape, generator=g).to(torch.float64)
            return _signs(g, shape) * (1.0 + _u(g, shape)) * torch.pow(2.0, e)
        a, b = gen((m, k)), gen((k, n))
    elif pattern == "mantissa":
        top = 2.0 - 2.0 ** (1 - SIGNIFICAND_BITS[dtype])          # 1.11...1 in binary

        def gen(shape):
            e = torch.randint(-2, 3, shape, generator=g).to(torch.float64)
            return _signs(g, shape) * top * torch.pow(2.0, e)
        a, b = gen((m, k)), gen((k, n))
    elif pattern == "cancel":
        if k % 2:
            raise RuntimeError("the cancel pattern needs an even K")
        y = _u(g, (m, k // 2)) * 2 - 1
        x = _u(g, (k // 2, n)) * 2 - 1
        a, b = torch.cat((y, y), dim=1), torch.cat((x, -x), dim=0)
    elif pattern == "alternate":
        a = _checker(m, k) * (0.5 + 0.5 * _u(g, (m, k)))
        b = _checker(k, n) * (0.5 + 0.5 * _u(g, (k, n)))
    elif pattern == "sparse":
        a = (_u(g, (m, k)) * 2 - 1) * (_u(g, (m, k)) >= 0.9)
        b = (_u(g, (k, n)) * 2 - 1) * (_u(g, (k, n)) >= 0.9)
    elif pattern == "subnormal":
        tiny = torch.finfo(dtype).tiny
        a = _signs(g, (m, k)) * (0.25 + 0.5 * _u(g, (m, k))) * tiny          # all subnormal, none near zero
        b = _signs(g, (k, n)) * (1.0 + _u(g, (k, n))) * 2.0 ** SUBNORMAL_LIFT[dtype]
    elif pattern == "near_max":
        a = _signs(g, (m, k)) * (0.5 + 0.5 * _u(g, (m, k)))
        b = _signs(g, (k, n)) * (0.5 + 0.5 * _u(g, (k, n)))
    else:
        raise ValueError(pattern)
    a, b = a.to(dtype).to(torch.float64), b.to(dtype).to(torch.float64)
    a, b, shift = fit_range(a, b, dtype, out_dtype, grow=(pattern == "near_max"))
    return a.to(dtype), b.to(dtype), shift


def int8_inputs(pattern, m, k, n, g):
    if pattern == "extremes":
        choices = torch.tensor([-128, -127, 0, 127], dtype=torch.int8)
        a = choices[torch.randint(0, 4, (m, k), generator=g)]
        b = choices[torch.randint(0, 4, (k, n), generator=g)]
    elif pattern == "alternate":
        a = (_checker(m, k) * torch.randint(1, 17, (m, k), generator=g)).to(torch.int8)
        b = (_checker(k, n) * torch.randint(1, 17, (k, n), generator=g)).to(torch.int8)
    elif pattern == "sparse":
        a = torch.randint(-16, 16, (m, k), generator=g, dtype=torch.int8) * (_u(g, (m, k)) >= 0.9)
        b = torch.randint(-16, 16, (k, n), generator=g, dtype=torch.int8) * (_u(g, (k, n)) >= 0.9)
    else:
        raise ValueError(pattern)
    return a.to(torch.int8), b.to(torch.int8)


def fp8_inputs(pattern, m, k, n, g):
    if pattern == "alternate":
        a, b = _checker(m, k), _checker(k, n)                      # every sum is +K or -K: still exact
    elif pattern == "sparse":
        a = torch.randint(-1, 2, (m, k), generator=g).to(torch.float64) * (_u(g, (m, k)) >= 0.9)
        b = torch.randint(-1, 2, (k, n), generator=g).to(torch.float64) * (_u(g, (k, n)) >= 0.9)
    else:
        raise ValueError(pattern)
    return a.to(torch.float32).to(screen.FP8), b.to(torch.float32).to(screen.FP8)


def make_pattern_case(name, pattern, m, k, n, seed, flushed=False):
    """Inputs for one dtype and pattern, plus the reference and |A|@|B| in the form screen_case expects.
    Returns (case, shift) where shift is the power of two the inputs were scaled by. With flushed=True the
    reference is built from A with its subnormal values set to zero, as a flushing device computes it."""
    dtype, out_dtype = screen.DTYPES[name][0], screen.DTYPES[name][1]
    if pattern not in APPLIES[name]:
        raise RuntimeError(f"pattern {pattern} does not apply to {name}")
    g = torch.Generator().manual_seed(seed * 7919 + ALL_PATTERNS.index(pattern))
    if dtype is None:
        raise RuntimeError("this PyTorch build has no float8_e4m3fn type")
    if dtype == torch.int8 or dtype == screen.FP8:
        a, b = int8_inputs(pattern, m, k, n, g) if dtype == torch.int8 else fp8_inputs(pattern, m, k, n, g)
        ref = (a.to(torch.float32).to(torch.int64) @ b.to(torch.float32).to(torch.int64)).to(torch.float64)
        return (a, b, ref, torch.zeros_like(ref)), 0
    a, b, shift = float_inputs(pattern, m, k, n, dtype, out_dtype, g)
    a64, b64 = a.to(torch.float64), b.to(torch.float64)
    if flushed:
        a64 = torch.where(a64.abs() < torch.finfo(dtype).tiny, torch.zeros_like(a64), a64)
    return (a, b, a64 @ b64, a64.abs() @ b64.abs()), shift


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite data pattern probe")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--shapes", default=DEFAULT_SHAPES, help="comma list of MxKxN")
    p.add_argument("--iters", type=int, default=20, help="runs per shape, precision and pattern")
    p.add_argument("--dtypes", default="fp32,fp16,bf16,int8,fp8,fp8fast", help="comma list from fp32,fp16,bf16,int8,fp8,fp8fast")
    p.add_argument("--patterns", default=",".join(ALL_PATTERNS), help="comma list from " + ",".join(ALL_PATTERNS))
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="bit flips to inject per step (self-test)")
    p.add_argument("--out", default="patterns.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    try:
        shapes = screen.parse_shapes(args.shapes)
    except ValueError as e:
        p.error(str(e))
    names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    pats = [x.strip() for x in args.patterns.split(",") if x.strip()]
    if any(d not in screen.DTYPES for d in names) or any(x not in ALL_PATTERNS for x in pats) or args.iters < 2:
        p.error("check --dtypes, --patterns and --iters >= 2")

    dev = screen.pick_device(args.device)
    screen.lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = screen.FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(name="ai-chip-integrity-patterns", version=VERSION,
                     parameters={"device": str(dev), "shapes": ",".join("x".join(map(str, s)) for s in shapes),
                                 "iters": args.iters, "dtypes": ",".join(names), "patterns": ",".join(pats),
                                 "seed": args.seed, "inject": args.inject, "bound_model": arith.MODEL_ID})
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
                for pattern in [x for x in pats if x in APPLIES[name]]:
                    step = run.add_step(f"gemm_{name}_{pattern}_{label}")
                    step.start()
                    try:
                        subnormal = pattern == "subnormal"
                        flushed = flushes_subnormal_inputs(dev, name) if subnormal else False
                        case, shift = make_pattern_case(name, pattern, *shape, args.seed, flushed=flushed)
                        step.add_measurement(name="pattern", value=pattern, hardware_info=hw)
                        step.add_measurement(name="input_scale_power_of_two", value=shift, hardware_info=hw)
                        if subnormal:
                            step.add_measurement(name="subnormal_inputs_flushed", value=flushed, hardware_info=hw)
                        ok, verdict, msg, secs = screen.screen_case(step, dev, name, shape, args.iters, args.seed,
                                                                    args.inject, rng, hw, case=case,
                                                                    allow_flush=not subnormal)
                        if subnormal:
                            msg += "; device flushes subnormal inputs" if flushed else "; device keeps subnormal inputs"
                        step.end(status=tv.TestStatus.COMPLETE)
                    except (RuntimeError, TypeError, NotImplementedError) as e:
                        step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                        step.end(status=tv.TestStatus.SKIP)
                        print(f"  {label:>16s} {name:7s} {pattern:9s}  SKIPPED  {str(e)[:90]}")
                        continue
                    all_ok &= ok
                    print(f"  {label:>16s} {name:7s} {pattern:9s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
    finally:
        try:
            all_ok &= counters.record(run, hw, before, counters.snapshot(dev), probe_ok=all_ok, injected=args.inject > 0)
        except Exception as e:                       # the cross-check must never cost the run its result file
            print(f"  error counters could not be recorded: {e}")
        run.end(status=tv.TestStatus.COMPLETE, result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()
    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
