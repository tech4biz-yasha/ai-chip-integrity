# ai-chip-integrity

[![CI](https://github.com/tech4biz-yasha/ai-chip-integrity/actions/workflows/ci.yml/badge.svg)](https://github.com/tech4biz-yasha/ai-chip-integrity/actions/workflows/ci.yml) [![PyPI](https://img.shields.io/pypi/v/ai-chip-integrity)](https://pypi.org/project/ai-chip-integrity/) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23285189.svg)](https://doi.org/10.5281/zenodo.23285189)

An open tester for silent computation errors in AI chips.

It runs a fixed calculation on a GPU, NPU or CPU many times and proves, element by element, whether the chip returned the right answer every time. Results come out in the Open Compute Project Test and Validation format, so they drop straight into fleet tooling.

**Status: early.** Six probes so far, numbered as on the roadmap below: `screen.py` v0.3.2 (probe 1, matrix multiply across several shapes, four precisions and exact FP8), `memcheck.py` v0.1.1 (probe 2, a sweep of most of the device memory), `kernels.py` v0.2.2 (probe 3, softmax, layer norm, GELU, rotary embeddings and fused attention), `abft.py` v0.1.0 (probe 6, checksum-protected matrix multiply with proven thresholds), `patterns.py` v0.1.2 (probe 9, data patterns that drive the matrix multiply) and `counters.py` v0.1.0 (probe 11, the chip's own error counters, read around every run of the other five). One device at a time, four measured chips. Read the limits section before relying on a PASS.

## Quick start

```
pip install ai-chip-integrity
chip-integrity run --quick     # about a minute: every probe and its self-test at small sizes
chip-integrity run             # the full run: ten OCP result files in chip-integrity-results/
```

On an NVIDIA GPU host with Docker and the NVIDIA container toolkit, with nothing else installed:

```
docker run --rm --gpus all -v "$PWD:/results" ghcr.io/tech4biz-yasha/ai-chip-integrity:0.4.0
```

## Why this exists

A chip can pass its vendor diagnostics and still compute wrong answers with no error raised. ByteDance reported at OSDI 2026 that the standard practice of isolating faulty GPUs with synthetic microbenchmarks missed over 60% of defective devices, that defects often appear later through aging, and that the failures are data dependent and unit specific, so a device that passes a general stress test can fail on specific input data ([Zheng et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zheng)). A second ByteDance and Tsinghua study found 18 silent corruption incidents and 13 faulty GPUs across 35 million GPU hours of production training ([Lei et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/lei)).

Those detectors live inside one company's training stack. This project is the open, vendor neutral version: the same idea, runnable by anyone on any chip, with results in a shared format and a public table.

## Measured results

Every row comes from a run of this code with the result files in [`results/`](results/). Nothing here is modelled or estimated.

**Compute probe (`screen.py`)**

| Device | Date | Tool | Shapes (MxKxN) | Runs per step | fp32 | fp16 | bf16 | int8 | FP8 | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-10 | v0.3.2 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | PASS, exact | PASS, exact with full and fast accumulation | 90 of 90 | no-silent-errors |
| NVIDIA H200, PyTorch 2.8.0+cu128, rented pod | 2026-10-08 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | PASS, exact | not in v0.2.0 | 60 of 60 | no-silent-errors |
| NVIDIA A100-SXM4-80GB, PyTorch 2.8.0+cu128, rented pod | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | PASS, exact | not in v0.2.0 | 60 of 60 | no-silent-errors |
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | PASS, exact | not in v0.2.0 | 60 of 60 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | skipped, no int8 matmul on MPS | not in v0.2.0 | 45 of 45 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.1.0 | 1024x1024x1024 | 50 | PASS, 1.6 s | PASS, 1.1 s | not run | not run | not run | 10 of 10 | no-silent-errors |

**Kernel probe (`kernels.py`)**

| Device | Date | Tool | Kernels | Size | Runs per step | fp32 | fp16 | bf16 | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-10 | v0.2.2 | softmax, layer norm, GELU, RoPE, attention | 4096x4096; RoPE 4096 positions x 128; attention 8 heads x 1024 tokens x 128 | 25 | PASS | PASS | PASS | 45 of 45 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-10 | v0.2.0 | softmax, layer norm, GELU, RoPE, attention | 4096x4096; RoPE 4096 positions x 128; attention 8 heads x 1024 tokens x 128 | 25 | PASS | PASS | PASS | 45 of 45 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-08 | v0.1.0 | softmax, layer norm, GELU, attention | 4096x4096; attention 8 heads x 1024 tokens x 128 | 25 | PASS | PASS | PASS | 36 of 36 | no-silent-errors |

Notes on the Apple rows: in the v0.2.0 run all 375 results across 15 steps were inside the bound and bit-for-bit identical to their first run, including RoPE with angles up to 4095 radians; result files `results/mac_kernels_v020.jsonl` and `results/mac_kernels_v020_inject.jsonl`. The v0.1.0 run (300 results across 12 steps, all clean, 36 of 36 faults caught) is kept as `results/mac_kernels.jsonl` and `results/mac_kernels_inject.jsonl`; the site shows the newest version per device.

Notes on the H100 row: all 375 results across 15 steps were inside the bound and bit-for-bit identical to their first run. Attention ran on FlashAttention in fp16 and bf16 and on the memory-efficient kernel in fp32, each chosen and recorded by the probe. The 16-bit kernels sit closest to their bounds (up to 0.999 for fp16 softmax), as expected where the single rounding of the output dominates the bound; fp32 stays at or below 0.162. The GPU's own functions were well inside the 2^-20 allowance: exp within 1.5e-7 relative, erf within 6.1e-8 and sin and cos within 8.4e-8 absolute. Result files: `results/h100_kernels_v022.jsonl` and `results/h100_kernels_v022_inject.jsonl`.

**Checksum probe (`abft.py`)**

| Device | Date | Tool | Shapes (MxKxN) | Runs per step | fp32 | fp16 | bf16 | Blind-spot runs | Injected errors located | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-10 | v0.1.0 | 1024x1024x1024, 32x4096x11008 | 20 | PASS | PASS | PASS | 0 | 18 of 18 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-10 | v0.1.0 | 1024x1024x1024, 32x4096x11008 | 20 | PASS | PASS | PASS | 0 | 18 of 18 | no-silent-errors |

Notes on the Apple row: all 120 runs across 6 steps had every row and column checksum consistent, sat inside the reference bound and were bit-for-bit identical to their first run, and every injected error was located to its exact row and column. Result files: `results/mac_abft.jsonl` and `results/mac_abft_inject.jsonl`.

Notes on the H100 row: all 120 runs had every checksum consistent, and every injected error was located to its exact row and column. The proven threshold over a typical value was 2.3 for fp32, 9.8 for fp16 and 14.1 for bf16 at 1024x1024x1024, and 193, 780 and 827 at 32x4096x11008, so at the decode shape the checksums alone can only guarantee to catch errors hundreds of times larger than a typical output; the reference and repeat checks carry the rest. The largest residual seen used under 4% of its threshold (bf16) and about 0.004% (fp32), the margin a statistical threshold would trade for proof. The fp16 checksum at the decode shape was scaled by 2^-2 to stay in range. Result files: `results/h100_abft.jsonl` and `results/h100_abft_inject.jsonl`.

**Data pattern probe (`patterns.py`)**

| Device | Date | Tool | Patterns | Shapes (MxKxN) | Runs per step | fp32 | fp16 | bf16 | int8 | FP8 | Subnormal inputs | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-10 | v0.1.2 | wide, mantissa, cancel, alternate, sparse, subnormal, near_max, extremes | 1024x1024x1024, 32x4096x11008 | 20 | PASS | PASS | PASS | PASS, exact | PASS, exact with full and fast accumulation | kept in fp32, fp16 and bf16 | 168 of 168 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-10 | v0.1.0 | wide, mantissa, cancel, alternate, sparse, subnormal, near_max | 1024x1024x1024, 32x4096x11008 | 20 | PASS | PASS | PASS | skipped, no int8 matmul on MPS | skipped, no FP8 on MPS | flushed in fp32 and bf16, kept in fp16 | 126 of 126 | no-silent-errors |

Notes on the Apple row: all 840 results across 42 steps were inside the bound and bit-for-bit identical to their first run. The Apple GPU's matrix multiply flushes subnormal inputs to zero in fp32 and bf16 and keeps them in fp16; each precision was checked against the exact model that matches. Result files: `results/mac_patterns.jsonl` and `results/mac_patterns_inject.jsonl`.

Notes on the H100 row: all 1,120 results across 56 steps were inside the bound or exact, and bit-for-bit identical to their first run; the worst error over bound was 0.93 (bf16, sparse, 1024x1024x1024). The H100 keeps subnormal inputs in fp32, fp16 and bf16 at both shapes, measured at each step's own shape. The same multiply flushes them on the Apple GPU in fp32 and bf16, and the pod's own host CPU (Xeon Platinum 8480+) flushed bf16 subnormals in its AMX kernels while keeping them at 16x16, so identical inputs can legitimately give different answers on different chips. Result files: `results/h100_patterns.jsonl` and `results/h100_patterns_inject.jsonl`.

**Memory sweep (`memcheck.py`)**

| Device | Date | Memory tested | Patterns | Dwell | Bad words | Injected flips located | Verdict |
|---|---|---|---|---|---|---|---|
| NVIDIA H200, 139.8 GiB total, rented pod | 2026-10-08 | 111.25 GiB (29,863,444,480 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |
| NVIDIA A100-SXM4-80GB, 79.3 GiB total, rented pod | 2026-10-07 | 63.00 GiB (16,911,433,728 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |
| NVIDIA H100 80GB HBM3 (SXM), 79.2 GiB total, rented pod, v0.1.1 with error counters | 2026-10-10 | 62.75 GiB (16,844,324,864 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |
| NVIDIA H100 80GB HBM3 (SXM), 79.2 GiB total, rented pod, v0.1.0 | 2026-10-07 | 62.75 GiB (16,844,324,864 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |
| Apple GPU via MPS, MacBook Pro (arm64), 11.8 GiB recommended max | 2026-10-07 | 5.50 GiB (1,476,395,008 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |

The H200 sweep took 24.8 s for 111.25 GiB, the A100 sweep took 28.3 s, the H100 sweep took 21.6 s both clean and with injection (21.8 s and 21.7 s in the v0.1.1 run, with every error counter quiet), so about 17 GiB/s of write, read and compare at HBM speed; the Mac sweep took 69 s clean and 85 s with injection. Result files: `results/h200_memcheck.jsonl`, `results/h200_memcheck_inject.jsonl`, `results/a100_memcheck.jsonl`, `results/a100_memcheck_inject.jsonl`, `results/h100_memcheck.jsonl`, `results/h100_memcheck_inject.jsonl`, `results/h100_memcheck_v011.jsonl`, `results/h100_memcheck_v011_inject.jsonl`, `results/mac_memcheck.jsonl` and `results/mac_memcheck_inject.jsonl`.

The H200, A100 and first H100 rows were measured with v0.2.0, before the FP8 steps existed.

Notes on the H100 v0.3.2 row: all 900 results across 18 steps were inside the bound or exact, and bit-for-bit identical to their first run. FP8 was exact on every element with both full and fast accumulation at all three shapes, including K = 4096. The worst error over bound was 0.0053 for fp32, 0.21 for fp16 and 0.66 for bf16. Result files: `results/h100_v032.jsonl` and `results/h100_v032_inject.jsonl`.

**Error counters (`counters.py`), first real cross-check.** On the H100, every clean run of the five probes ended `counters-quiet`. ECC was on, the volatile corrected and uncorrected ECC counts read 0 before and after each run, no memory rows were remapped, and nothing was pending or failed. During the compute run the GPU went from 31 to 34 C, from 345 to 1980 MHz and from 69.7 to 118.9 W. Apple rows read `counters-unavailable`, as designed.

Notes on the H200 row: all 600 results across 12 steps were bit-for-bit identical to their first run and int8 was exact on every element. Result files: `results/h200_clean.jsonl` and `results/h200_inject.jsonl`.

Notes on the A100 row: all 600 results across 12 steps were bit-for-bit identical to their first run and int8 was exact on every element, the same as the H100. Result files: `results/a100_clean.jsonl` and `results/a100_inject.jsonl`.

Notes on the H100 v0.2.0 row: all 600 results across 12 steps were bit-for-bit identical to their first run, with TF32 and reduced-precision reductions disabled and cuBLAS set deterministic. int8 ran through the tensor cores and matched the exact int64 reference on every element. Wall time per step (56 to 59 s at 4096x4096x4096) is dominated by the CPU-side checking of 16.8 million elements per run, not by the GPU. Result files: `results/h100_clean.jsonl` and `results/h100_inject.jsonl`.

Notes on the Apple rows: in the v0.2.0 run all 450 results across 9 steps were bit-for-bit identical to their first run, so the matrix multiply on this chip is deterministic, and fp16 and bf16 stayed inside the FP32-accumulation error bound at every shape. The 4096x4096x4096 steps took 23 to 30 s each for 50 runs. The fault column is the self-test described below (software injected, not a property of the chip). Result files: `results/mac_v020.jsonl` and `results/mac_v020_inject.jsonl`.

## Probe 1: compute (`screen.py`)

Matrix multiplies `C = A @ B` with fixed inputs, repeated `--iters` times for every shape and precision. Three shapes run by default: 1024x1024x1024, 4096x4096x4096, and 32x4096x11008, which is the shape of an LLM decode step through a feed-forward layer (M tokens by K hidden by N intermediate). Float inputs are drawn uniformly from [-1, 1) with a fixed seed, then rounded to the test precision, so every precision starts from the same underlying matrices. int8 inputs are integers in [-16, 16), and FP8 inputs come from {-1, 0, 1}.

Every run is checked two ways.

**1. Reference check.** For float precisions the reference `R` is computed in float64 on the CPU from the rounded inputs, and each element of the device result must satisfy

```
|C - R|  <=  acc  +  u_out * ( |R| + acc )  +  floor

acc    = (g + g_ref) * (|A| @ |B|)  +  (K + 1) * 2^-126  +  products with a subnormal input
g      = n * u / (1 - n * u)        Higham's inner product bound
g_ref  = gamma(K, 2^-53)            the float64 reference's own rounding (from v0.3.1)
fp32   n = K,  u = 2^-24            IEEE FP32 fused multiply-add, TF32 off (CUDA and CPU)
fp16   n = 2K, u = 2^-23            matrix units: faithful accumulation, truncation allowed,
bf16                                any block width (also fp32 on other devices, plus 2^-20 per product)
u_out  = 2^-24, 2^-11 or 2^-8       output rounding (fp32, fp16, bf16)
floor  = half the subnormal spacing of the output type
```

The first term is the largest error the accumulation can produce under the arithmetic model in [`arith.py`](arith.py), plus the reference's own rounding and FP32 underflow; the second is the final rounding to the output type. NVIDIA tensor cores accumulate with truncation rather than round to nearest (Fasi, Higham, Mikaitis and Pranesh, PeerJ CS 2021; Valpey and Pai, arXiv 2502.15999), so for fp16 and bf16 the bound uses the truncation unit and allows for block accumulation. Up to v0.2.0 the bound used `n = K, u = 2^-24` for every precision, about four times tighter at fp16 and bf16; every published row passed that tighter bound, so it passes this one. Any element outside the bound, and any NaN or Inf, is a wrong answer, not a tolerance judgement.

For int8 the device returns int32 and the reference is computed exactly in int64, so the bound is zero: any difference at all is a wrong answer.

**FP8.** From v0.3.0 the probe also runs the FP8 (E4M3) tensor-core path that inference engines use, through `torch._scaled_mm` with FP32 output, once with full accumulation (`fp8`) and once with fast accumulation (`fp8fast`). The inputs come from {-1, 0, 1}, so every product is exact and every partial sum is an integer no larger than K: with K up to 4096 that fits in 13 significant bits. Hopper's FP8 tensor cores are reported to keep 14 bits when they add (DeepSeek-V3 technical report, 2024), so the exact answer is the only correct one, even with fast accumulation, and any difference is a fault or an accumulator narrower than 13 bits. It needs an Ada, Hopper or Blackwell GPU; on an A100 or an Apple GPU these steps are skipped and recorded. Shapes with K above 4096 are skipped for FP8. FP4 follows once a Blackwell card is measured.

**2. Repeat check.** Every run is compared bit for bit with the first run on the same device. A single flipped bit anywhere in the output fails this check, even when it is far too small to leave the reference bound.

**Verdict per precision**

| Verdict | Meaning |
|---|---|
| `no-silent-errors` | Every run inside the bound and bit-identical to run 0 |
| `silent-data-corruption` | Some runs differ from run 0 (intermittent fault), possibly also outside the bound |
| `outside-error-bound` | All runs agree with each other but sit outside the bound: a faulty unit, a device breaking the arithmetic model (accumulating below FP32, or TF32 used for fp32), or an int8 or FP8 result that is not exact |
| `nondeterministic-kernel` | At least 3 runs and at least 10% of runs differ from run 0, yet all stay inside the bound. The repeat check cannot be used on that device and precision |

**Self-test.** `--inject N` flips one random bit of one random output element in `N` runs (never run 0), after the result has been copied back from the device. It proves the checker catches corruption; it does not stress the chip. The verdict then becomes `injection-self-test-pass` only if every injected run was caught and no uninjected run failed.

## Probe 2: memory (`memcheck.py`)

The matrix multiply touches a few megabytes. The memory sweep fills as much of the device memory as it can (by default 80% of what is free on CUDA, 48% of the recommended maximum on Apple MPS, 2 GiB on CPU, or `--gb` to choose), writes a known value into every 32-bit word, waits a dwell time (default 2 s), reads everything back and counts every word that differs.

Six patterns run in order: all zeros, all ones, `0xAAAAAAAA`, `0x55555555`, an address hash (every word pseudo-random from its own address, so neighbouring cells never hold the same value) and the bitwise inverse of that hash. Together they drive every bit of every word to both 0 and 1 while its neighbours hold the opposite, which is what catches stuck bits, coupled cells and weak rows. The expected value is recomputed on the device at read time, so the comparison runs at device speed. For every failing pattern the first ten byte offsets and the OR of the bad bits are logged, so a stuck bit shows up as a single mask such as `0x00000020`.

**Verdict per pass**

| Verdict | Meaning |
|---|---|
| `no-memory-errors` | Every word read back correctly on every pattern |
| `memory-errors` | At least one word differed; the measurements say which patterns and the log says where |

**Self-test.** `--inject N` flips one random bit in N random words of device memory after each pattern is written. Unlike the compute probe's self-test, this corrupts real device memory, so it also exercises the read-back path. The verdict is `injection-self-test-pass` only when every injected flip is located and no other word fails.

## Probe 3: transformer kernels (`kernels.py`)

Matrix multiply exercises the multiply-accumulate units. A transformer block also runs exponentials, divisions, square roots and error functions, which run on different hardware, and faults are data dependent. This probe runs five kernels through the device's own implementations and checks every output element against a float64 reference on the CPU, with the same two checks as the compute probe.

| Kernel | Default input | Operations exercised |
|---|---|---|
| `softmax` | 4096 x 4096 logits, N(0, 3) | exp, sum reduction, divide |
| `layernorm` | 4096 x 4096 activations, N(2, 1.5), random weight and bias | mean, variance, reciprocal square root, scale, shift |
| `gelu` | 4096 x 4096, N(0, 3), erf form | erf, multiply |
| `rope` | 4096 positions x 128 (one head), base 10000, FP32 angles up to 4095 rad | sin and cos with large-angle argument reduction, multiply, add |
| `attention` | 8 heads x 1024 tokens x 128, `scaled_dot_product_attention` | QK^T, softmax, PV through the fused kernel |

Precisions: fp32, fp16, bf16. 25 runs per kernel and precision by default.

**Bounds.** Each kernel has a componentwise bound from standard rounding-error analysis (Higham), applied to every element:

1. softmax: the relative error of each exponential, `EPS_FN + u|x_i - max| + u`, carried through the sum (`gamma_n`) and the quotient.
2. layer norm: error of the mean, of each deviation, of the variance (sized to cover the two-pass, Welford and `E[x^2] - mean^2` methods), of the reciprocal square root, then of scale and shift.
3. GELU: argument scaling, erf, `1 + erf`, the product and the output rounding.
4. rope: cos and sin of the FP32 angles (2^-20 absolute), their cast to the test type, then two products and a sum, each rounded in the test type, as eager PyTorch does.
5. attention: the matrix-unit error of every logit from QK^T, propagated through softmax as `expm1(2 * max logit error)`; the exp of each weight and of every online-softmax rescale (at most one per block of 16 keys); the row sum; P rounded to the input precision, with an absolute allowance where P is subnormal; and the matrix-unit error of PV.

Every bound adds the output rounding of the test precision and an absolute floor of half the subnormal spacing of the output type, the most that rounding to nearest can lose below the smallest normal number.

**Stated assumptions**, the arithmetic model in [`arith.py`](arith.py), recorded in every result file:

1. Scalar FP32 arithmetic rounds to nearest; matrix units accumulate faithfully (truncation allowed).
2. exp, divide, square root and reciprocal square root are within `EPS_FN = 2^-20` relative error, eight FP32 ulps anywhere in a binade. erf, sin and cos are within `2^-20` absolute, which admits the absolutely accurate approximations vector libraries use (PyTorch's ARM CPU erf is Abramowitz and Stegun 7.1.26: up to 5.4e-7 absolute error, but up to 100% relative near zero).
3. Fused attention keeps logits and softmax in FP32 and holds P in the input precision. On CUDA, fp16 and bf16 attention may only use the fused kernels (flash, memory-efficient, cuDNN), so the unfused path, which rounds logits to the input precision, cannot stand in silently. The fused kernels take a batch dimension, so the probe passes (1, heads, tokens, dim), tries each kernel on its own in the order flash, cuDNN, memory-efficient (then the math path for fp32 only), runs the first that accepts the inputs, and records its name as `attention_backend`; if none accepts, the step is refused with every reason.

A device that breaks any of them, or that substitutes the tanh approximation for GELU, shows as `outside-error-bound` on every run and the message says so. To tell a faulty unit from a loose function library, every softmax, GELU and RoPE step also measures the device's own exp, erf, sin and cos on that step's inputs and records the worst error; when it exceeds the allowance, the verdict message says the library is outside the model. At the test default, the fp32 bounds sit within 3e-5 of the output scale for softmax, layer norm and GELU, and within 1e-3 for attention, whose bound carries worst-case matrix-unit accumulation through the softmax (`tests/test_kernels.py::test_bounds_are_tight_for_fp32`).

Verdicts and the `--inject N` self-test are the same as the compute probe.

## Probe 6: checksum-protected matrix multiply (`abft.py`)

Huang and Abraham's algorithm-based fault tolerance (IEEE Transactions on Computers, 1984). A is extended with a row holding its column sums and B with a column holding its row sums, so one multiply on the device returns C together with C's own row sums and column sums, computed by the same hardware in the same pass:

```
A' = [ A ; 2^-sc * (column sums of A) ]
B' = [ B , 2^-sr * (row sums of B) ]
A' @ B' = [ C , 2^-sr * (row sums of C) ; 2^-sc * (column sums of C) , . ]
```

A wrong element breaks exactly one row check and one column check, and their crossing is the element. No reference answer is needed, so the same check can run inside real work for the cost of one extra row and column (0.2% more arithmetic at 1024x1024, about 3% at the 32-row decode shape).

**Thresholds.** In floating point a correct row sum differs from its checksum by rounding, so the threshold decides everything. Each one is derived per row and per column from the arithmetic model in `arith.py`: the bounds of every element in the row and of the checksum element, the float64 host sum, and the rounding of the checksum vectors to the input type, which the host knows exactly and removes. A correct device cannot cross it. The powers of two `2^-sc` and `2^-sr` keep the checksums inside the input type, the output type and an FP32 accumulator, because a checksum is a sum of m or n values and would otherwise overflow near the top of the range; they are recorded as `checksum_scale_power_of_two`.

**What a proven threshold costs.** `threshold_over_typical_value` reports how large an error must be, compared with a typical element of C, before the checksums are guaranteed to see it. On CPU it is about 0.07 in fp32 at 256x256x256 and between 1.5 and 9 in bf16. Because the threshold adds up the worst-case rounding of every element in a row, it grows with the row length and the inner dimension: on the H100 it was 2.3 in fp32 at 1024x1024x1024 and 193 at 32x4096x11008. Smaller faults are left to the reference check and the bit-for-bit repeat check, which run on every run as in the compute probe.

**The blind spot, measured.** Four errors in a rectangle, +d, -d, -d, +d, cancel in every row and column sum, so checksums cannot see them. Every run the reference check fails while every checksum holds is counted as `checksum_missed_runs` and named in the verdict message, never passed.

**Self-test.** `--inject N` adds, in N runs, an error four times the threshold to one random element of C after it is copied back; the self-test passes only if the checksums flag exactly that row and that column every time. fp32, fp16 and bf16 are supported; int8 and FP8 need checksums wider than their own types and come later.

## Probe 9: data patterns (`patterns.py`)

Some faults only fire on particular bit patterns, and uniform random inputs almost never produce them. This probe runs the compute probe's matrix multiply, with the same two checks and the same proven or exact reference, on inputs built to reach specific parts of the arithmetic.

| Pattern | Inputs | What it reaches |
|---|---|---|
| `wide` | random exponents across about 80 binades (fp32, bf16) or 15 (fp16) | the shifters that align operands before they are added |
| `mantissa` | every mantissa bit set, at exponents -2 to 2 | the longest carry chains in multipliers and adders |
| `cancel` | A = [Y Y], B = [X; -X] | an exact answer of zero, so only rounding may remain |
| `alternate` | checkerboard signs, so every product in a dot product has the same sign | sums at their largest, the worst case for truncating accumulators |
| `sparse` | nine values in ten exactly zero | zero handling and zero skipping |
| `subnormal` | A below the smallest normal number of its type, B large enough that every product is normal | the subnormal input path |
| `near_max` | scaled so the largest possible partial sum is a quarter to a half of the output type's maximum | the largest exponents and the output conversion |
| `extremes` | int8 only: -128, -127, 0 and 127 | the corners of the integer multiplier |

Every input set is scaled by a power of two where needed, so that no partial sum, in any order, can overflow the output type or an FP32 accumulator. FP8 keeps its inputs in {-1, 0, 1} so its answers stay exact, so it runs `alternate` and `sparse` only; int8 runs `extremes`, `alternate` and `sparse`.

**Subnormals.** Flushing subnormal inputs to zero is allowed by the arithmetic model, but a bound wide enough to cover both behaviours would let real faults through. So before each `subnormal` step the probe multiplies a matrix of the step's own shape, A holding one subnormal value everywhere and B one power of two, which makes every product the same normal number and every sum exact. Each output then shows directly whether that element's subnormal inputs were kept or flushed. The shape matters because libraries choose kernels by shape: on an H100 pod's host CPU, the small-matrix path kept subnormals while the larger bf16 kernels (AMX and AVX-512 BF16 instructions, which ignore the flush setting) flushed them. The probe records `subnormal_input_policy` (`keeps`, `flushes` or `mixed`) and `subnormal_inputs_flushed`, then checks the device against the exact model that matches, with no allowance for the other. Only a kernel that keeps in some places and flushes in others (`mixed`) gets the wider bound, which also lets any product with a subnormal input be dropped. Tests in `tests/test_patterns.py` make the CPU flush inside the matmul only and confirm that judging it against the wrong model fails, emulate a library that picks its kernel by shape, and emulate a kernel that mixes, with and without injected flips.

The bound for every float step adds two terms the compute probe also carries from v0.3.1: the float64 reference's own rounding, `gamma(K, 2^-53) * (|A| @ |B|)`, and `(K + 1) * 2^-126` for products that may underflow FP32. Both are far below the other terms on ordinary data; they make the bound hold on the extreme inputs as well.

## Probe 11: error counters (`counters.py`)

Every run of the other four probes reads the chip's own error counters at its start and its end, and records them in one more step, `device_error_counters`. Then it asks the question a silent error raises: did the hardware notice?

| Verdict | Meaning |
|---|---|
| `counters-quiet` | No counter moved and the probe found nothing wrong |
| `silent-error-confirmed` | The probe found wrong answers and every counter stayed quiet: silent in the strict sense, the case ECC and the error logs cannot see |
| `hardware-noticed` | The probe found wrong answers and a counter moved too |
| `corrected-during-run` | A corrected-error counter moved but every answer was right. The hardware caught something; on a healthy chip that is worth watching. The run still passes |
| `uncorrected-during-run` | An uncorrected-error counter moved, or row remapping failed: a hardware fault even if the answers held. The run fails |
| `self-test-run` | Faults were planted in software, so no cross-check is drawn; the counters are still recorded |
| `counters-unavailable` | No NVIDIA management library, or a device without these counters (Apple GPU, CPU). Never fails a run |

On NVIDIA GPUs the counters come from NVML through the `nvidia-ml-py` package (installed on Linux by `requirements.txt`): volatile ECC counts since the driver loaded, corrected and uncorrected; remapped memory rows with their pending and failure flags (A100 and newer); and retired pages (older parts). Temperature, SM clock and power are recorded beside them for context. A counter the device does not support is left out of the file, never reported as zero. The device is matched to NVML by UUID, because CUDA and NVML can number devices differently.

## Run it

`chip-integrity run` executes every probe and then its self-test, each in its own process, and writes ten OCP result files. It runs exactly the ten probe commands listed further down, with the same self-test strength, so its files are the same as running them by hand.

Options for `chip-integrity run`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N`, passed to every probe |
| `--name` | `chip` | prefix of the result file names |
| `--out-dir` | `chip-integrity-results` | folder for the files; `chip-integrity-quick` with `--quick` |
| `--quick` | off | small sizes: a smoke test, not a publishable row |
| `--skip` | none | comma list of probes to leave out, from `screen`, `kernels`, `abft`, `patterns`, `memcheck` |

The exit code is 0 when every step passes and 1 when any fails; a summary names each step and its file. `chip-integrity screen` (and `memcheck`, `kernels`, `abft`, `patterns`) runs one probe with its own options, listed below, and `chip-integrity --version` prints the suite version and every probe's version.

From a clone of the repository, the probes also run as scripts, with no install beyond the requirements:

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python screen.py                 # compute probe; auto picks cuda, then mps, then cpu; 3 shapes x 4 precisions, plus FP8
python screen.py --inject 5      # compute self-test
python kernels.py                # kernel probe; softmax, layer norm, GELU, RoPE, attention x fp32, fp16, bf16
python kernels.py --inject 3     # kernel self-test
python abft.py                   # checksum probe; C with its own row and column checksums
python abft.py --inject 3        # checksum self-test: injected errors must be located
python patterns.py               # data pattern probe; 8 patterns through the matrix multiply
python patterns.py --inject 3    # data pattern self-test
python memcheck.py               # memory probe; 6 patterns over most of the device memory
python memcheck.py --inject 5    # memory self-test
python -m pytest -q              # 176 tests on CPU; 6 more validate the output against OCP's schema
```

Options for `memcheck.py`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--gb` | none | memory to test in GiB; overrides `--fraction` |
| `--fraction` | `0.8` | fraction of free device memory to test |
| `--chunk-mb` | `256` | allocation chunk size |
| `--dwell` | `2.0` | seconds between write and read-back |
| `--passes` | `1` | repeat the whole pattern set |
| `--seed` | `1234` | hash pattern seed |
| `--inject` | `0` | bit flips to inject per pattern (self-test, max 100) |
| `--out` | `memcheck.jsonl` | OCP output file |

Options for `screen.py`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--shapes` | `1024x1024x1024,4096x4096x4096,32x4096x11008` | comma list of `MxKxN` |
| `--size` | none | shortcut for one square `NxNxN` shape |
| `--iters` | `50` | runs per shape and precision |
| `--dtypes` | `fp32,fp16,bf16,int8,fp8,fp8fast` | comma list from `fp32`, `fp16`, `bf16`, `int8`, `fp8`, `fp8fast` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | bit flips to inject per step (self-test) |
| `--out` | `results.jsonl` | OCP output file |

Options for `kernels.py`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--size` | `4096` | rows and columns for softmax, layer norm and GELU, positions for RoPE (head dimension 128); attention uses 8 heads x size/4 tokens x 128 |
| `--iters` | `25` | runs per kernel and precision |
| `--kernels` | `softmax,layernorm,gelu,rope,attention` | comma list of kernels |
| `--dtypes` | `fp32,fp16,bf16` | comma list from `fp32`, `fp16`, `bf16` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | bit flips to inject per step (self-test) |
| `--out` | `kernels.jsonl` | OCP output file |

Options for `abft.py`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--shapes` | `1024x1024x1024,32x4096x11008` | comma list of `MxKxN` |
| `--iters` | `20` | runs per shape and precision |
| `--dtypes` | `fp32,fp16,bf16` | comma list from `fp32`, `fp16`, `bf16` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | errors to inject per step (self-test) |
| `--out` | `abft.jsonl` | OCP output file |

Options for `patterns.py`:

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--shapes` | `1024x1024x1024,32x4096x11008` | comma list of `MxKxN` |
| `--iters` | `20` | runs per shape, precision and pattern |
| `--dtypes` | `fp32,fp16,bf16,int8,fp8,fp8fast` | comma list of precisions |
| `--patterns` | all eight | comma list from `wide`, `mantissa`, `cancel`, `alternate`, `sparse`, `subnormal`, `near_max`, `extremes` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | bit flips to inject per step (self-test) |
| `--out` | `patterns.jsonl` | OCP output file |

Exit code is 0 for PASS and 1 for FAIL. A precision or shape the device does not support (for example int8 matmul on a device without it) is skipped and recorded as an error artifact.

On CUDA, `screen.py` and `kernels.py` turn off TF32 and reduced precision reductions, set `CUBLAS_WORKSPACE_CONFIG`, and ask PyTorch for deterministic algorithms, so the bound and the repeat check both hold. `kernels.py` also limits fp16 and bf16 attention to the fused kernels and records which one ran; that control needs PyTorch 2.3 or later, and on older versions the device default runs.

To run the schema test, clone [ocp-diag-core](https://github.com/opencomputeproject/ocp-diag-core) and set `OCP_SCHEMA_DIR` to its `json_spec/output` folder.

## Files

| Path | What it is |
|---|---|
| `chip_integrity/screen.py` | Probe 1, matrix multiply |
| `chip_integrity/kernels.py` | Probe 3, transformer kernels |
| `chip_integrity/abft.py` | Probe 6, checksum-protected matrix multiply |
| `chip_integrity/patterns.py` | Probe 9, data patterns through the matrix multiply |
| `chip_integrity/counters.py` | Probe 11, the chip's error counters, read around every run of the other probes |
| `chip_integrity/memcheck.py` | Probe 2, memory sweep |
| `chip_integrity/arith.py` | The arithmetic model every bound is derived from, with its id written into each result file |
| `chip_integrity/cli.py` | The `chip-integrity` command: `run` executes every probe and its self-test, each in its own process, and writes the ten result files; `chip-integrity <probe>` runs one probe with its own options |
| `screen.py`, `kernels.py`, `abft.py`, `patterns.py`, `memcheck.py` | Small wrappers so `python screen.py` and the other probe commands keep working from a clone |
| `pyproject.toml` | Package metadata for PyPI. The version lives in `chip_integrity/__init__.py` and must match `CITATION.cff` |
| `Dockerfile` | The `ghcr.io/tech4biz-yasha/ai-chip-integrity` image: PyTorch 2.8.0 with CUDA 12.8, the same build the H100 rows were measured with, and `chip-integrity run` as its default command |
| `.github/workflows/` | `ci.yml` runs every test, including the OCP schema checks, builds the package, and runs a quick pass inside the Docker image on every push; `release.yml` publishes the package to PyPI and the image to GitHub's container registry when a release is published |
| `build_site.py` | Builds `docs/index.html` (chipintegrity.org) from `docs/template.html` and every file in `results/`, so the published tables can never say more than the files do. One row per device and probe; a newer tool version replaces the older row. Run `python build_site.py` after adding result files |
| `results/` | Every published result file, in OCP format |
| `tests/` | The test suite below |

## Tests

`python -m pytest -q` runs 176 tests on the CPU in under a minute; 6 more validate every probe's output against the official OCP schema when `OCP_SCHEMA_DIR` is set. On CPUs whose matrix multiply ignores flush-to-zero, such as Apple silicon, the two flushing tests in `tests/test_patterns.py` skip themselves, giving 174 passed.

1. `tests/test_screen.py` (37): every verdict path, every bit position caught by the self-test, NaN handling, int8 and FP8 off-by-one, FP8 inputs exact and K above 4096 skipped, rectangular shapes, bound tightness.
2. `tests/test_kernels.py` (23): all kernels and precisions, the self-test on six seeds, intermittent and systematic faults, a one-ulp change, internal fp16 arithmetic and tanh-GELU both caught as outside the bound, NaN handling, fp32 bound tightness, a sin that is off by 1e-4 caught in RoPE, RoPE angles reaching every position, the device's own exp, erf, sin and cos measured, and the CUDA attention path forced onto the CPU: a batch dimension added with the same shape and values back, the kernel that ran recorded, fused kernels only for fp16 and bf16, and a loud refusal when no kernel fits.
3. `tests/test_memcheck.py` (8): a stuck bit reported with its offset, a dead row counted word by word, injected flips located, the patterns themselves.
4. `tests/test_arith.py` (19): the model checked against simulations of the hardware it claims to cover. A simulated tensor core that truncates, at block widths 1 to 32, stays inside the bound; a constructed case breaks the old v0.2.0 bound by almost 2x and stays inside the new one; a FlashAttention-2 style kernel emulated op by op stays inside the attention bound; PyTorch's ARM CPU erf and GELU, emulated op by op, stay inside the GELU bound; and FP8 sums of {-1, 0, 1} stay exact in a 13-bit accumulator while a 9-bit one loses them.
5. `tests/test_build_site.py` (6): a newer tool version replaces the older row for the same device on the site, kernel sizes render from the files, data pattern rows render with the flushing behaviour, a mixed policy is shown as such, checksum rows render with their located self-test count, and the attention kernel that ran is named.
6. `tests/test_patterns.py` (17): every pattern passes on CPU, each input set really has its pattern and stays clear of overflow, FP8 and int8 inputs stay exact, flushing is detected and judging a flushing device against the wrong model fails, the policy is measured at each step's own shape (a library that picks its kernel by shape), a kernel that keeps in some places and flushes in others passes with and without injected flips, a fault four times the bound is caught on the cancelling pattern, injected flips are caught on every pattern, and the output matches the OCP schema.
7. `tests/test_abft.py` (25): clean runs in every precision, injected errors located to their exact row and column on several seeds, a single error pointing at its element, a fault in a checksum flagging only its line, no false alarm on the same-sign, wide, near-maximum and cancelling patterns in every precision, checksums scaled to stay in range, the rectangle fault counted as a blind spot, NaN, and the OCP schema.
8. `tests/test_counters.py` (17): with a simulated NVIDIA management library, every cross-check verdict; unsupported counters left out rather than zeroed; no counters on CPU and Apple devices; corrected errors reported without failing a run; uncorrected errors failing it; a wrong answer with quiet counters confirmed silent; the site keeping the counter step out of the precision list; and the OCP schema.
9. `tests/test_readme.py` (17): this README carries every probe's version, every option and its default, every option of `chip-integrity run`, every result file, every script and test file, and the citation version and DOI, and the package version matches the citation, so neither can drift from the code without a test failing.
10. `tests/test_cli.py` (13): `chip-integrity run` executes exactly the ten probe commands in this README with the same file names and self-test strength, `--quick` only adds small sizes, `--skip` leaves whole probes out, a failing step fails the run without stopping the others, a quick run is never offered as a row, bad names and unknown probes are refused before anything runs, one probe runs with its own options, the old `python screen.py` paths still work, and a real quick run writes its files.

## Output format

All four probes write one JSON object per line following the [OCP Test and Validation output spec](https://github.com/opencomputeproject/ocp-diag-core/tree/main/json_spec), version 2.0, written through the official `ocptv` library:

- `testRunStart` with the parameters used (for `screen.py` and `kernels.py` these include `bound_model`, the id of the arithmetic model in `arith.py`), the host name as DUT id, the device as a hardware component, and the `torch` and `python` versions as software components
- one test step per shape and precision, named `gemm_<precision>_<MxKxN>` (for example `gemm_fp16_4096x4096x4096`), carrying the measurements `runs`, `shape_mkn`, `exact_check`, `accumulation_factor` (the `g` used in the bound, float precisions only), `seconds`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `worst_error_over_bound_ratio` (float precisions only), `worst_abs_error`, `worst_nonfinite_values`, and in self-test mode `injected_runs` and `injected_runs_detected`, with validators on the counts that must be zero or equal to the injected count
- a `diagnosis` per step with the verdict above
- `testRunEnd` with the overall PASS or FAIL

`kernels.py` writes one step per kernel and precision, named `<kernel>_<precision>` (for example `attention_bf16`), with `runs`, `attention_backend` (attention steps: `flash`, `cudnn`, `efficient`, `math` or `default`), `size`, `elements`, `seconds`, `assumed_fn_rel_error`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `worst_error_over_bound_ratio`, `worst_abs_error`, `worst_nonfinite_values`, `device_exp_rel_worst_error`, `device_erf_abs_worst_error` or `device_sincos_abs_worst_error` (softmax, GELU and RoPE steps), the self-test counts, and a diagnosis with the verdict. Its run parameters also record `assumed_fn_rel_error` and `attention_kernels` (which attention kernels were allowed).

`abft.py` writes one step per shape and precision, named `abft_<precision>_<MxKxN>`, with `runs`, `shape_mkn`, `seconds`, `flop_overhead`, `checksum_scale_power_of_two`, `row_threshold_median`, `typical_value_median`, `threshold_over_typical_value`, `worst_row_residual_over_threshold`, `worst_column_residual_over_threshold`, `checksum_check_failed_runs`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `checksum_missed_runs`, in self-test mode `injected_runs` and `injected_runs_located`, and a diagnosis with the verdict and, on a failure, the flagged rows and columns.

Every probe ends its run with a `device_error_counters` step from `counters.py`: `counters_available`, `ecc_enabled`, then for each counter the device supports `<counter>_before`, `<counter>_after` and `<counter>_delta` (`ecc_corrected_volatile`, `ecc_uncorrected_volatile`, `remapped_rows_corrected`, `remapped_rows_uncorrected`, `retired_pages_single_bit`, `retired_pages_double_bit`), `remap_pending`, `remap_failure`, `temperature_c`, `sm_clock_mhz` and `power_w` before and after, and a diagnosis with the cross-check verdict.

`patterns.py` writes one step per shape, precision and pattern, named `gemm_<precision>_<pattern>_<MxKxN>` (for example `gemm_bf16_near_max_1024x1024x1024`), with the compute probe's measurements plus `pattern`, `input_scale_power_of_two` and, for the subnormal pattern, `subnormal_input_policy` and `subnormal_inputs_flushed`.

`memcheck.py` writes one step per pass (`memory_sweep_pass1`, ...) with `bytes_tested`, `device_memory_bytes`, `coverage_fraction`, `dwell_seconds`, `seconds`, one `bad_words_<pattern>` per pattern (validated to be zero outside self-test), a warning log line per failing pattern with the first byte offsets and the bad-bit mask, and in self-test mode `injected_<pattern>` and `injected_detected_<pattern>`.

## Assumptions and limits

1. Every bound follows from the arithmetic model in `arith.py`: FP32 scalar arithmetic rounds to nearest, matrix units accumulate faithfully in FP32 (truncation allowed), exp, divide and square roots stay within eight ulps, and erf within 2^-20 absolute. TF32 is switched off on CUDA. A device that accumulates below FP32 or breaks the model in another way shows up as `outside-error-bound` on every run, and the message says so. That was not the case on any chip measured above.
2. These are six probes. A PASS means "no silent error in these matrix multiplies, these checksums, these data patterns, these transformer kernels and this memory sweep, in these runs, on this device today". It is not a certificate of a healthy chip. The OSDI 2026 results above show that defects depend on the data, the kernel, temperature and age; this version sweeps shapes, precisions, five transformer kernels and memory, but not the rest.
3. Runs take seconds, not hours, so thermal and aging effects are not exercised.
4. One device per run. Multi-device comparison is on the roadmap.
5. The error counters are NVIDIA's volatile counts since the driver last loaded; on other devices probe 11 records that none are available. In the compute and kernel probes the injected faults are applied to the output on the CPU side, so they test the checker, not the chip. The memory probe injects into device memory itself.

## Roadmap

Twelve probes, each aimed at one part of the chip. Six are built: 1, 2, 3, 6, 9 and 11.

1. **Matrix multiply** (`screen.py`). Done for fp32, fp16, bf16, int8 and FP8 (E4M3, exact, from v0.3.0). FP4 follows once a Blackwell card is measured.
2. **Memory pattern sweep** (`memcheck.py`). Done for device memory. On-chip SRAM (shared memory and L2) is next, NVIDIA GPUs only.
3. **Transformer kernels** (`kernels.py`). Done, with rotary embeddings (sin and cos) added in v0.2.0.
4. **Full model forward pass.** A small LLM with every layer checked against a CPU float64 reference. Each layer is fed the device's own input to that layer, so its bound stays as tight as a single kernel's; one bound carried through the whole model would be too loose to catch anything.
5. **Reductions, all-reduce and copy engines.** Sums across threads, blocks and then GPUs over NVLink and PCIe, checked with checksums that must agree at both ends, and host-to-device and device-to-device copies of known patterns.
6. **Checksum-protected matrix multiply** (`abft.py`, algorithm-based fault tolerance, Huang and Abraham 1984). Done for fp32, fp16 and bf16: row and column checksums computed in the same multiply detect and locate a wrong element, with proven thresholds and the blind spot measured. int8 and FP8 checksums come later.
7. **Hours under load.** The same probes repeated for hours at full power, with temperature and clocks logged next to every result.
8. **Clock and voltage margin sweep.** Lower the margin step by step where the driver allows it and record where each unit starts to fail. Likely needs bare-metal access, since container pods rarely allow clock control.
9. **Data pattern library** (`patterns.py`). Done for the matrix multiply: wide exponents, full mantissas, cancellation, alternating signs, sparsity, subnormals with flush detection, near-overflow and int8 extremes. Patterns for the transformer kernels are next.
10. **Fault injection inside the computation.** Flip bits in registers during the multiply (NVBit on NVIDIA) to measure how often a flip becomes a wrong answer. The basis for a space radiation column.
11. **ECC and error counters** (`counters.py`). Done: every run of every probe reads the chip's counters before and after and records whether the hardware noticed what the probe noticed. Its first real test was the H100 run on 10 October 2026, with every ECC and row-remap counter quiet.
12. **Recovery.** After a detected fault, reset and rerun on the same card to see whether the fault clears. Likely needs bare-metal access, since a GPU reset needs root on the host.

Around the probes: a fleet mode that runs every probe across many devices and collects the files, side by side runs against vendor diagnostics on the same device, and a rulebook for submitting rows to the public table.

## Contributing a row

Run `chip-integrity run --name NAME` on your device, or the ten probe commands in Run it, then open a pull request with the ten result files (`NAME_clean.jsonl`, `NAME_inject.jsonl`, `NAME_kernels.jsonl`, `NAME_kernels_inject.jsonl`, `NAME_abft.jsonl`, `NAME_abft_inject.jsonl`, `NAME_patterns.jsonl`, `NAME_patterns_inject.jsonl`, `NAME_memcheck.jsonl` and `NAME_memcheck_inject.jsonl`), the device name, driver version and PyTorch version. Rows are added only from attached result files.

## Citing

If you use this in research, please cite it as

```
Yasha Khandelwal. ai-chip-integrity: an open tester for silent computation errors in AI chips. Version 0.4.0, 2026.
https://doi.org/10.5281/zenodo.23285189
https://github.com/tech4biz-yasha/ai-chip-integrity
```

A `CITATION.cff` file is included.

## Author and license

Yasha Khandelwal, yasha.khandelwal@tech4biz.io

MIT License. See `LICENSE`.
