"""
The bounds rest on arith.py. These tests build the arithmetic the model claims to cover, including
the worst parts of it, and check the bound holds. Run from the repo root:  python -m pytest -q
"""

import math
import pathlib
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import arith  # noqa: E402
import kernels  # noqa: E402


def truncate(x, ulp):
    """Round x toward zero to a multiple of ulp (exact in Python floats for these magnitudes)."""
    return math.trunc(x / ulp) * ulp


def fp32_ulp(x):
    return 2.0 ** (math.frexp(abs(x))[1] - 24) if x else 2.0 ** -149


def tensor_core_dot(terms, block):
    """A matrix-unit dot product as measured on NVIDIA tensor cores: each step aligns the running sum and
    `block` exact products to the largest exponent, truncates what falls off a 24-bit window, adds exactly,
    then truncates the sum to FP32. No guard bits, round toward zero: the worst case the model claims."""
    acc = 0.0
    for i in range(0, len(terms), block):
        vals = [acc] + terms[i:i + block]
        top = max(abs(v) for v in vals)
        if top == 0.0:
            continue
        window = fp32_ulp(top)
        acc = math.fsum(truncate(v, window) for v in vals)              # exact sum of the aligned values
        acc = truncate(acc, fp32_ulp(acc))                               # then truncated to FP32
    return acc


@pytest.mark.parametrize("block", [1, 2, 4, 8, 16, 32])
def test_matrix_bound_covers_truncating_blocks(block):
    rng = random.Random(block)
    n = 512
    for trial in range(40):
        # fp16 inputs: products exact in float64; mixed magnitudes and signs maximise alignment loss
        a = [float(torch.tensor(rng.uniform(-1, 1) * 2.0 ** rng.randint(-6, 0), dtype=torch.float16)) for _ in range(n)]
        b = [float(torch.tensor(rng.uniform(-1, 1) * 2.0 ** rng.randint(-6, 0), dtype=torch.float16)) for _ in range(n)]
        terms = [x * y for x, y in zip(a, b)]
        if trial % 2:
            terms = [abs(t) for t in terms]              # same sign: the sum grows, truncation always bites
        exact = math.fsum(terms)
        err = abs(tensor_core_dot(terms, block) - exact)
        assert err <= arith.matrix_dot(n) * math.fsum(abs(t) for t in terms), (block, trial)


def test_truncation_breaks_the_old_bound_and_not_the_new_one():
    """Why v0.2.1 changed the fp16/bf16 bound. Running sum near 1.0, then many products just under two
    ulps: truncation drops almost an ulp every step, twice what round-to-nearest gamma(n, 2^-24) allows."""
    x = float(torch.tensor(2.0 ** -12 * (2.0 - 2.0 ** -10), dtype=torch.float16))   # exact in fp16
    n = 512
    terms = [1.0] + [x * x] * (n - 1)                                                # exact products
    exact = math.fsum(terms)
    err = abs(tensor_core_dot(terms, 1) - exact)
    total = math.fsum(abs(t) for t in terms)
    assert err > arith.ieee_dot(n) * total * 1.9            # old bound broken by almost 2x
    assert err <= arith.matrix_dot(n) * total               # new bound holds


def flash_attention(q, k, v, block):
    """FlashAttention-2 forward in FP32, emulated op by op: online softmax over key blocks, running max and
    rescale, exp, P rounded to the input precision for the PV product, final divide."""
    h, L, d = q.shape
    scale = torch.tensor(1.0 / math.sqrt(d), dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    m = torch.full((h, L, 1), -math.inf)
    l = torch.zeros(h, L, 1)
    acc = torch.zeros(h, L, d)
    for j in range(0, L, block):
        s = (qf @ kf[:, j:j + block].transpose(1, 2)) * scale
        m_new = torch.maximum(m, s.max(dim=2, keepdim=True).values)
        alpha = torch.exp(m - m_new)
        p = torch.exp(s - m_new)
        l = alpha * l + p.sum(dim=2, keepdim=True)
        acc = alpha * acc + p.to(q.dtype).float() @ vf[:, j:j + block]
        m = m_new
    return (acc / l).to(q.dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [16, 64])
def test_attention_bound_covers_flash_style_kernel(dtype, block):
    (q, k, v), ref, bound = kernels.case_attention(4, 256, 64, dtype, seed=11)
    out = flash_attention(q, k, v, block).to(torch.float64)
    ratio = float(((out - ref).abs() / bound).max())
    assert ratio <= 1.0, ratio


def _fma32(a, b, c):
    return (a.astype(np.float64) * b.astype(np.float64) + c.astype(np.float64)).astype(np.float32)


def arm_erf(x):
    """PyTorch 2.8's aarch64 Vectorized<float>::erf, op by op in float32: Abramowitz and Stegun 7.1.26,
    t = 1/(1 + p|x|), a degree-5 polynomial in t, times exp(-x^2), subtracted from 1."""
    f = np.float32
    p, p1, p2, p3, p4, p5 = (f(c) for c in (0.3275911, 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429))
    one = np.ones_like(x)
    t = (one / _fma32(np.full_like(x, p), np.abs(x), one)).astype(f)
    r = np.full_like(x, p5)
    for c in (p4, p3, p2, p1):
        r = _fma32(r, t, np.full_like(x, c))
    e = np.exp(-(x * x).astype(f)).astype(f)
    return np.copysign(_fma32((t * -e).astype(f), r, one), x)


def arm_gelu(x):
    """PyTorch's CPU vectorized_gelu: x * 0.5 * (1 + erf(x * sqrt(1/2))), in float32."""
    f = np.float32
    a = (x * f(math.sqrt(0.5))).astype(f)
    return ((x * f(0.5)).astype(f) * (f(1.0) + arm_erf(a)).astype(f)).astype(f)


def test_arm_erf_is_absolutely_but_not_relatively_accurate():
    """Why erf gets an absolute allowance: near zero this erf is wrong by up to 100% relative."""
    x = np.concatenate([np.linspace(-6, 6, 200001), np.geomspace(1e-9, 1e-2, 20001)]).astype(np.float32)
    true = torch.erf(torch.from_numpy(x.astype(np.float64))).numpy()
    err = np.abs(arm_erf(x).astype(np.float64) - true)
    assert err.max() < 0.6 * arith.EPS_FN
    near_zero = (np.abs(x) < 1e-4) & (true != 0)
    assert (err[near_zero] / np.abs(true[near_zero])).max() > 0.5


@pytest.mark.parametrize("seed", [0, 1, 1234])
def test_gelu_bound_covers_arm_cpu_erf(seed):
    (x,), ref, bound = kernels.case_gelu(256, 256, torch.float32, seed)
    y = torch.from_numpy(arm_gelu(x.numpy())).to(torch.float64)
    assert float(((y - ref).abs() / bound).max()) <= 1.0


def narrow_accumulator_dot(terms, bits):
    """Sum with an accumulator that keeps only `bits` significant bits and truncates, one term at a time."""
    acc = 0.0
    for t in terms:
        s = acc + t
        if s:
            ulp = 2.0 ** (math.frexp(abs(s))[1] - bits)
            s = truncate(s, ulp)
        acc = s
    return acc


def test_fp8_ternary_sums_need_13_bits_and_no_more():
    """Why the FP8 check is exact: sums of K <= 4096 terms from {-1, 0, 1} are integers no larger than 4096,
    exact in a 13-bit accumulator. A narrower accumulator does lose them, so the check would catch one."""
    rng = random.Random(5)
    worst13, lost_narrow = 0.0, 0
    for trial in range(6):
        terms = [float(rng.choice((1, 1, 1, 0, -1))) for _ in range(4096)]   # biased, so the sum grows large
        exact = math.fsum(terms)
        worst13 = max(worst13, abs(narrow_accumulator_dot(terms, 13) - exact))
        lost_narrow += narrow_accumulator_dot(terms, 9) != exact
    assert worst13 == 0.0
    assert lost_narrow > 0


def test_model_id_is_recorded_constants():
    assert "2^-23" in arith.MODEL_ID and "2^-20" in arith.MODEL_ID
    assert arith.ulps(arith.EPS_FN) == 8.0
