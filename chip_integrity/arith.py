"""
arith.py  -  the arithmetic model behind every bound in the AI Chip Integrity Suite.

Every reference bound in the suite is derived from these assumptions and nothing else, so an
element outside a bound means the device broke at least one of them. The model id below is
written into every result file.

  IEEE    Scalar FP32 add, subtract, multiply and fused multiply-add round to nearest, as IEEE 754
          requires, in any order. Unit roundoff U_RN = 2^-24. A sum or dot product of n terms is
          then within gamma(n, U_RN) of the sum of the absolute terms (Higham, Theorem 3.1).

  MATRIX  Matrix units (NVIDIA tensor cores and their equivalents) form products of fp16 and bf16
          inputs exactly and accumulate in FP32 faithfully: a step that adds m values to a running
          sum, whatever the block width m, lands within one FP32 ulp of the largest of them per
          value. That covers round to nearest and also the truncation measured on NVIDIA tensor
          cores (Fasi, Higham, Mikaitis and Pranesh, PeerJ CS 2021; Valpey and Pai, arXiv
          2502.15999). Summing the per-step errors gives at most 2n units of U_TC = 2^-23 for a
          dot product of n terms, so the factor is gamma(2n, U_TC). FP32 inputs on a matrix unit
          may be emulated (3xTF32, for example) and each product may carry EPS_MUL relative error.

  FN      exp, exp2, divide, square root and reciprocal square root are within EPS_FN of the true
          value, relative. EPS_FN = 2^-20 covers eight FP32 ulps anywhere in a binade (an ulp is at
          most 2^-23 of the value). erf, sin and cos are within EPS_FN absolute: their values are at
          most 1 in size, so this includes every relatively accurate implementation, and it also admits
          the absolutely accurate approximations vector libraries use (Abramowitz and Stegun 7.1.26 on
          PyTorch's ARM CPU path errs by up to 5.4e-7 absolute, but by up to 100% relative near zero).

  OUT     Conversion of an FP32 result to fp16 or bf16 rounds to nearest and keeps subnormal results,
          as IEEE 754 requires, so below the smallest normal number the error is at most half a
          subnormal spacing. (Subnormal inputs to the matrix multiply may be flushed; screen.py allows
          for that separately.)

  REF     The float64 reference is computed with rounding too: a dot product of n terms is within
          gamma(n, 2^-53) of its absolute sum, and that is added to every bound. Products too small for
          FP32's normal range may underflow: each can lose at most the smallest normal number, 2^-126.

A device that keeps intermediates below FP32, uses TF32 where FP32 was requested, or has a less
accurate exp or erf than EPS_FN, shows as outside-error-bound on every run, and the probes say so.
"""

import math

import torch

MODEL_ID = "arith-v1: ieee u=2^-24; matrix faithful gamma(2n) u=2^-23, fp32 products 2^-20; fn 2^-20 (erf, sin, cos absolute); out rn"

U_RN = 2.0 ** -24
U_TC = 2.0 ** -23
EPS_FN = 2.0 ** -20
EPS_MUL = 2.0 ** -20

# per output type: unit roundoff, smallest normal, half the subnormal spacing
OUT = {
    torch.float32: (2.0 ** -24, 2.0 ** -126, 2.0 ** -150),
    torch.float16: (2.0 ** -11, 2.0 ** -14, 2.0 ** -25),
    torch.bfloat16: (2.0 ** -8, 2.0 ** -126, 2.0 ** -134),
}


def gamma(n, u):
    """Higham's gamma_n = n*u / (1 - n*u)."""
    nu = n * u
    if nu >= 1.0:
        raise ValueError(f"gamma undefined for n={n}, u={u}")
    return nu / (1.0 - nu)


def ieee_dot(n):
    """Error factor for a sum or dot product of n terms on IEEE FP32 scalar units."""
    return gamma(n, U_RN)


def matrix_dot(n, fp32_inputs=False):
    """Error factor for a dot product of n terms on a matrix unit, any block width, truncating or not."""
    g = gamma(2 * n, U_TC)
    return g + EPS_MUL * (1.0 + g) if fp32_inputs else g


def reference_dot(n):
    """Rounding error factor of the float64 reference for a dot product of n terms."""
    return gamma(n, 2.0 ** -53)


def underflow_floor(n):
    """Most that n products and their running sum can lose to FP32 underflow, absolute."""
    return (n + 1) * 2.0 ** -126


def subnormal_floor(dtype):
    """Absolute error allowed on an output of this type below its smallest normal number."""
    return OUT[dtype][2]


def flush_term(x64, dtype):
    """|x| where x is subnormal in dtype, else 0: what a device that flushes subnormal inputs may lose."""
    return torch.where(x64.abs() < OUT[dtype][1], x64.abs(), torch.zeros_like(x64))


def ulps(rel):
    """A relative error expressed in FP32 ulps where an ulp is largest relative to the value (2^-23)."""
    return rel / U_TC if rel else 0.0


assert math.isclose(ulps(EPS_FN), 8.0)
