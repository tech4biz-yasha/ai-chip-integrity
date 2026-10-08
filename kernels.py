#!/usr/bin/env python3
"""
kernels.py  -  AI Chip Integrity Suite, transformer kernel probe v0.1.0

The matrix multiply probe exercises the multiply-accumulate units. Real model
layers also run exponentials, divisions, square roots, error functions and
reductions, which live on different silicon. This probe runs four kernels
that make up a transformer block and checks every output element against a
float64 reference with a componentwise rounding-error bound, plus the same
bit-for-bit repeat check as the matrix probe.

  softmax     rows of logits:            exp, sum-reduce, divide
  layernorm   rows of activations:       mean, variance, rsqrt, scale, shift
  gelu        elementwise (erf form):    erf, multiply
  attention   softmax(Q K^T / sqrt(d)) V with the device's own attention kernel

Bounds are derived from standard rounding-error analysis (Higham) for the
operations in each kernel, under the arithmetic model in arith.py, whose id
every result file records: FP32 scalar arithmetic rounds to nearest, matrix
units accumulate faithfully in FP32 (truncation allowed), exp, divide, sqrt
and rsqrt are within eight FP32 ulps, and erf within 2^-20 absolute. Fused attention is assumed
to keep logits and softmax in FP32; on CUDA, fp16 and bf16 attention may only
run on fused kernels so the unfused path cannot stand in silently.
A device that breaks the model shows up as outside-error-bound on every run,
and the message says so.

--inject N flips one random bit in N chosen runs (never the first) to prove
the checks catch corruption. Output: OCP Test & Validation JSON to --out.
"""

import argparse
import math
import os
import platform
import random
import socket
import sys
import time

# must be set before CUDA initialises so cuBLAS runs deterministically
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import ocptv.output as tv  # noqa: E402

import arith  # noqa: E402

from screen import FileWriter, pick_device, device_label, lock_down_math, flip_bit, NONDET_SHARE, NONDET_MIN  # noqa: E402

VERSION = "0.1.0"
U_RN, U_TC, EPS_FN = arith.U_RN, arith.U_TC, arith.EPS_FN
DTYPES = {
    "fp32": (torch.float32, torch.int32, 32),
    "fp16": (torch.float16, torch.int16, 16),
    "bf16": (torch.bfloat16, torch.int16, 16),
}
KERNELS = ["softmax", "layernorm", "gelu", "attention"]
MIN_KEY_BLOCK = 16          # fused attention rescales its running sums at most once per block of this many keys
_FAULT_HOOK = None          # tests only: callable(run_index, cpu_tensor) that corrupts an output in place


def u_out(dtype):
    return arith.OUT[dtype][0]


def finish(bound, ref, dtype):
    """Output rounding to the test type (round to nearest) and the subnormal floor."""
    return bound + u_out(dtype) * (ref.abs() + bound) + arith.subnormal_floor(dtype)


# ---------------------------------------------------------------- cases and bounds
# Each bound is the forward error of the kernel under the model in arith.py: scalar FP32 arithmetic
# rounds to nearest, exp/divide/sqrt/rsqrt are within EPS_FN relative and erf within EPS_FN absolute,
# matrix units are faithful.

def case_softmax(rows, cols, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, cols, generator=g, dtype=torch.float64) * 3.0).to(dtype)
    x64 = x.to(torch.float64)
    m = x64.max(dim=1, keepdim=True).values
    e = torch.exp(x64 - m)
    s = e / e.sum(dim=1, keepdim=True)
    return (x,), s, finish(softmax_error(x64, s, cols), s, dtype)


def softmax_error(x64, s, n):
    """exp(x_i - max) / sum_j exp(x_j - max), computed in FP32."""
    m = x64.max(dim=1, keepdim=True).values
    r = EPS_FN + U_RN * (x64 - m).abs() + U_RN               # each exponential: function error, argument rounding
    rmax = r.max(dim=1, keepdim=True).values
    den = rmax + arith.ieee_dot(n) * (1.0 + rmax)            # the row sum
    rho = den / (1.0 - den)                                  # its reciprocal
    rel = (1.0 + r) * (1.0 + rho) * (1.0 + EPS_FN + U_RN) - 1.0   # divide, or reciprocal then multiply
    return s * rel


def case_layernorm(rows, cols, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, cols, generator=g, dtype=torch.float64) * 1.5 + 2.0).to(dtype)
    w = (torch.rand(cols, generator=g, dtype=torch.float64) + 0.5).to(dtype)
    b = (torch.randn(cols, generator=g, dtype=torch.float64) * 0.5).to(dtype)
    x64, w64, b64 = x.to(torch.float64), w.to(torch.float64), b.to(torch.float64)
    eps = 1e-5
    mu = x64.mean(dim=1, keepdim=True)
    d = x64 - mu
    var = (d * d).mean(dim=1, keepdim=True)
    rinv = 1.0 / torch.sqrt(var + eps)
    y = d * rinv * w64 + b64
    return (x, w, b, eps), y, finish(layernorm_error(x64, w64, b64, d, var, rinv, y, cols, eps), y, dtype)


def layernorm_error(x64, w64, b64, d, var, rinv, y, n, eps):
    """(x - mean) * rsqrt(var + eps) * w + b. The variance term covers two-pass, Welford and E[x^2] - mean^2."""
    gn = arith.ieee_dot(n + 3)
    e_mu = gn * x64.abs().mean(dim=1, keepdim=True)
    e_d = e_mu + U_RN * (d.abs() + e_mu)
    e_var = ((2.0 * d.abs() * e_d + e_d * e_d).mean(dim=1, keepdim=True) * (1.0 + gn)
             + gn * ((x64 * x64).mean(dim=1, keepdim=True) + var + eps))
    v = var + eps
    e_r = e_var / (2.0 * (v - e_var).clamp_min(1e-300)) + U_RN + EPS_FN     # add eps, then rsqrt
    t = d * rinv
    e_t = e_d * rinv * (1.0 + e_r) + t.abs() * e_r + U_RN * t.abs() * (1.0 + e_r)
    e_y = w64.abs() * e_t + 2.0 * U_RN * ((w64 * t).abs() + b64.abs())
    return e_y * (1.0 + 4.0 * U_RN)


def case_gelu(rows, cols, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    x = (torch.randn(rows, cols, generator=g, dtype=torch.float64) * 3.0).to(dtype)
    x64 = x.to(torch.float64)
    a = x64 / math.sqrt(2.0)
    erf = torch.erf(a)
    y = 0.5 * x64 * (1.0 + erf)
    return (x,), y, finish(gelu_error(x64, a, erf, y), y, dtype)


def gelu_error(x64, a, erf, y):
    """0.5 * x * (1 + erf(x / sqrt 2)): the argument (a multiply by a rounded constant, or a divide), erf,
    1 + erf, and the product. erf' is largest at the smaller end of the perturbed argument."""
    da = EPS_FN + U_RN                                         # relative error of the argument
    slope = (2.0 / math.sqrt(math.pi)) * torch.exp(-(a.abs() * (1.0 - 2.0 * da)) ** 2)
    # erf is allowed EPS_FN absolute, not relative: |erf| <= 1, so this also covers a relatively accurate erf,
    # and it admits the absolutely accurate approximations vector libraries use (PyTorch's ARM CPU path
    # uses Abramowitz and Stegun 7.1.26, whose relative error near zero reaches 100%)
    e_erf = slope * a.abs() * da + EPS_FN + U_RN * (1.0 + erf).abs()
    return (0.5 * x64.abs()) * e_erf * (1.0 + 2.0 * U_RN) + 2.0 * U_RN * y.abs()


def case_attention(heads, L, dim, dtype, seed):
    g = torch.Generator().manual_seed(seed)
    q, k, v = (torch.randn(heads, L, dim, generator=g, dtype=torch.float64).to(dtype) for _ in range(3))
    q64, k64, v64 = q.to(torch.float64), k.to(torch.float64), v.to(torch.float64)
    scale = 1.0 / math.sqrt(dim)
    sc = (q64 @ k64.transpose(1, 2)) * scale
    m = sc.max(dim=2, keepdim=True).values
    e = torch.exp(sc - m)
    p = e / e.sum(dim=2, keepdim=True)
    o = p @ v64
    return (q, k, v), o, finish(attention_error(q64, k64, v64, sc, p, o, L, dim, scale, dtype), o, dtype)


def attention_error(q64, k64, v64, sc, p, o, L, dim, scale, dtype):
    """softmax(Q K^T * scale) V, as a fused kernel computes it.

    Logits come from a matrix-unit dot product (plus up to four roundings for scale factors) and
    are shifted and scaled again on the way into exp or exp2. Online softmax rescales the running
    sums at most once per block of MIN_KEY_BLOCK keys. P is held in the input precision for the
    second product, which also runs on a matrix unit. Logits and softmax stay in FP32."""
    fp32 = dtype == torch.float32
    u_in, min_normal, half_sub = arith.OUT[dtype]
    nb = math.ceil(L / MIN_KEY_BLOCK)
    abs_qk = (q64.abs() @ k64.abs().transpose(1, 2)) * scale
    m = sc.max(dim=2, keepdim=True).values
    lo = sc.min(dim=2, keepdim=True).values
    e_logit = arith.matrix_dot(dim + 4, fp32_inputs=fp32) * abs_qk + 3.0 * U_RN * (sc.abs() + m.abs())
    rel_logits = torch.expm1(2.0 * e_logit.max(dim=2, keepdim=True).values)   # softmax moves at most this much
    r = (nb + 1) * (EPS_FN + 2.0 * U_RN) + 2.0 * U_RN * (m - lo)              # exp of each weight and each rescale
    den = r + arith.ieee_dot(L + nb) * (1.0 + r)                             # the row sum of weights
    rho = den / (1.0 - den)
    rel_w = (1.0 + rel_logits) * (1.0 + r) * (1.0 + rho) * (1.0 + EPS_FN + U_RN) * (1.0 + u_in) - 1.0
    e_p = p * rel_w + torch.where(p < min_normal, torch.full_like(p, half_sub), torch.zeros_like(p))
    e_o = e_p @ v64.abs() + arith.matrix_dot(L + nb, fp32_inputs=fp32) * ((p + e_p) @ v64.abs())
    return e_o * (1.0 + 2.0 * U_RN)


def run_kernel(name, args):
    if name == "softmax":
        return torch.softmax(args[0], dim=1)
    if name == "layernorm":
        x, w, b, eps = args
        return F.layer_norm(x, (x.shape[1],), weight=w, bias=b, eps=eps)
    if name == "gelu":
        return F.gelu(args[0])
    if name == "attention":
        return attention(*args)
    raise ValueError(name)


def attention(q, k, v):
    """On CUDA, fp16 and bf16 may only use fused kernels, whose logits and softmax stay in FP32 as the bound
    assumes; the unfused math path rounds logits to the input precision, so it is not allowed to stand in
    silently. fp32 may also use the math path (plain FP32 with TF32 off). Elsewhere the default kernel runs."""
    if q.device.type == "cuda":
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
        except ImportError:                       # torch < 2.3: no backend control
            return F.scaled_dot_product_attention(q, k, v)
        allowed = [SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            allowed.append(SDPBackend.CUDNN_ATTENTION)
        if q.dtype == torch.float32:
            allowed.append(SDPBackend.MATH)
        with sdpa_kernel(allowed):
            return F.scaled_dot_product_attention(q, k, v)
    return F.scaled_dot_product_attention(q, k, v)


def make_case(name, dtype, seed, size):
    rows, cols = size, size
    if name == "softmax":
        return case_softmax(rows, cols, dtype, seed)
    if name == "layernorm":
        return case_layernorm(rows, cols, dtype, seed)
    if name == "gelu":
        return case_gelu(rows, cols, dtype, seed)
    if name == "attention":
        return case_attention(8, size // 4, 128, dtype, seed)
    raise ValueError(name)


# ---------------------------------------------------------------- probe

def probe(step, dev, kernel, dname, size, iters, seed, inject, rng, hw):
    dtype, int_view, bits = DTYPES[dname]
    args, ref, bound = make_case(kernel, dtype, seed, size)
    dev_args = tuple(a.to(dev) if torch.is_tensor(a) else a for a in args)
    if kernel == "layernorm":
        dev_args = (dev_args[0], dev_args[1], dev_args[2], args[3])

    inject_runs = set(rng.sample(range(1, iters), min(inject, iters - 1))) if inject else set()
    first_bits = None
    ref_bad, rep_bad = [], []
    injected, caught = 0, 0
    worst_ratio, worst_abs, worst_nonfinite = 0.0, 0.0, 0
    t0 = time.time()

    for i in range(iters):
        c = run_kernel(kernel, dev_args).to("cpu").contiguous()
        if c.dtype != dtype:
            raise RuntimeError(f"device returned {c.dtype}, expected {dtype}")
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
            worst_ratio = max(worst_ratio, float((err[finite] / bound[finite].clamp_min(1e-300)).max()))
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
    step.add_measurement(name="size", value=size, hardware_info=hw)
    step.add_measurement(name="elements", value=int(ref.numel()), hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    step.add_measurement(name="assumed_fn_rel_error", value=EPS_FN, hardware_info=hw)
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
        msg = (f"all runs agree but {len(ref_bad)} runs sit outside the bound (a faulty unit, or the device breaking "
               f"the arithmetic model: intermediates below FP32, TF32 used for fp32, exp/divide/rsqrt worse than "
               f"{arith.ulps(EPS_FN):.0f} ulps, erf worse than 2^-20 absolute, or an approximate kernel such as tanh-GELU)")

    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL,
                       verdict=verdict, message=msg, hardware_info=hw)
    return ok, verdict, msg, secs


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite transformer kernel probe")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--size", type=int, default=4096, help="rows and columns for softmax/layernorm/gelu; attention uses 8 heads x size/4 tokens x 128")
    p.add_argument("--iters", type=int, default=25, help="runs per kernel and precision")
    p.add_argument("--kernels", default=",".join(KERNELS), help="comma list from softmax,layernorm,gelu,attention")
    p.add_argument("--dtypes", default="fp32,fp16,bf16", help="comma list from fp32,fp16,bf16")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="bit flips to inject per step (self-test)")
    p.add_argument("--out", default="kernels.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    kernels = [k.strip() for k in args.kernels.split(",") if k.strip()]
    names = [d.strip() for d in args.dtypes.split(",") if d.strip()]
    if any(k not in KERNELS for k in kernels) or any(d not in DTYPES for d in names) or args.iters < 2 or args.size < 16:
        p.error("check --kernels, --dtypes, --iters >= 2, --size >= 16")

    dev = pick_device(args.device)
    lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(name="ai-chip-integrity-kernels", version=VERSION,
                     parameters={"device": str(dev), "size": args.size, "iters": args.iters,
                                 "kernels": ",".join(kernels), "dtypes": ",".join(names), "seed": args.seed,
                                 "inject": args.inject, "assumed_fn_rel_error": EPS_FN,
                                 "bound_model": arith.MODEL_ID,
                                 "attention_kernels": ("fused only for fp16/bf16 (flash, efficient, cudnn); math also "
                                                       "for fp32" if dev.type == "cuda" else "device default")})
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=device_label(dev))
    dut.add_software_info(name="torch", version=torch.__version__)
    dut.add_software_info(name="python", version=platform.python_version())

    print(f"Device: {device_label(dev)} ({dev})  size={args.size}  runs={args.iters}  inject={args.inject}")
    all_ok = True
    run.start(dut=dut)
    try:
        for kernel in kernels:
            for name in names:
                step = run.add_step(f"{kernel}_{name}")
                step.start()
                try:
                    ok, verdict, msg, secs = probe(step, dev, kernel, name, args.size, args.iters,
                                                   args.seed, args.inject, rng, hw)
                    step.end(status=tv.TestStatus.COMPLETE)
                except (RuntimeError, TypeError, NotImplementedError) as e:
                    step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                    step.end(status=tv.TestStatus.SKIP)
                    print(f"  {kernel:10s} {name:5s}  SKIPPED  {str(e)[:100]}")
                    continue
                all_ok &= ok
                print(f"  {kernel:10s} {name:5s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
    finally:
        run.end(status=tv.TestStatus.COMPLETE,
                result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()
    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
