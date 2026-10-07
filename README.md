# ai-chip-integrity

An open tester for silent computation errors in AI chips.

It runs a fixed calculation on a GPU, NPU or CPU many times and proves, element by element, whether the chip returned the right answer every time. Results come out in the Open Compute Project Test and Validation format, so they drop straight into fleet tooling.

**Status: early.** Two probes so far: `screen.py` v0.2.0 (matrix multiply across several shapes and four precisions) and `memcheck.py` v0.1.0 (a sweep of the whole device memory). One device at a time, two measured chips. Read the limits section before relying on a PASS.

## Why this exists

A chip can pass its vendor diagnostics and still compute wrong answers with no error raised. ByteDance reported at OSDI 2026 that the standard practice of isolating faulty GPUs with synthetic microbenchmarks missed over 60% of defective devices, that defects often appear later through aging, and that the failures are data dependent and unit specific, so a device that passes a general stress test can fail on specific input data ([Zheng et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zheng)). A second ByteDance and Tsinghua study found 18 silent corruption incidents and 13 faulty GPUs across 35 million GPU hours of production training ([Lei et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/lei)).

Those detectors live inside one company's training stack. This project is the open, vendor neutral version: the same idea, runnable by anyone on any chip, with results in a shared format and a public table.

## Measured results

Every row comes from a run of this code with the result files in [`results/`](results/). Nothing here is modelled or estimated.

**Compute probe (`screen.py`)**

| Device | Date | Tool | Shapes (MxKxN) | Runs per step | fp32 | fp16 | bf16 | int8 | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA H100 80GB HBM3 (SXM), driver 580.126.09, PyTorch 2.8.0+cu128, rented pod | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | PASS, exact | 60 of 60 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | skipped, no int8 matmul on MPS | 45 of 45 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.1.0 | 1024x1024x1024 | 50 | PASS, 1.6 s | PASS, 1.1 s | not run | not run | 10 of 10 | no-silent-errors |

**Memory sweep (`memcheck.py` v0.1.0)**

| Device | Date | Memory tested | Patterns | Dwell | Bad words | Injected flips located | Verdict |
|---|---|---|---|---|---|---|---|
| Apple GPU via MPS, MacBook Pro (arm64), 11.8 GiB recommended max | 2026-10-07 | 5.50 GiB (1,476,395,008 words) | 6 | 2 s | 0 | 30 of 30 | no-memory-errors |

The sweep took 69 s clean and 85 s with injection. Result files: `results/mac_memcheck.jsonl` and `results/mac_memcheck_inject.jsonl`.

Notes on the H100 row: all 600 results across 12 steps were bit-for-bit identical to their first run, with TF32 and reduced-precision reductions disabled and cuBLAS set deterministic. int8 ran through the tensor cores and matched the exact int64 reference on every element. Wall time per step (56 to 59 s at 4096x4096x4096) is dominated by the CPU-side checking of 16.8 million elements per run, not by the GPU. Result files: `results/h100_clean.jsonl` and `results/h100_inject.jsonl`.

Notes on the Apple rows: in the v0.2.0 run all 450 results across 9 steps were bit-for-bit identical to their first run, so the matrix multiply on this chip is deterministic, and fp16 and bf16 stayed inside the FP32-accumulation error bound at every shape. The 4096x4096x4096 steps took 23 to 30 s each for 50 runs. The fault column is the self-test described below (software injected, not a property of the chip). Result files: `results/mac_v020.jsonl` and `results/mac_v020_inject.jsonl`.

## Probe 1: compute (`screen.py`)

Matrix multiplies `C = A @ B` with fixed inputs, repeated `--iters` times for every shape and precision. Three shapes run by default: 1024x1024x1024, 4096x4096x4096, and 32x4096x11008, which is the shape of an LLM decode step through a feed-forward layer (M tokens by K hidden by N intermediate). Float inputs are drawn uniformly from [-1, 1) with a fixed seed, then rounded to the test precision, so every precision starts from the same underlying matrices. int8 inputs are integers in [-16, 16).

Every run is checked two ways.

**1. Reference check.** For float precisions the reference `R` is computed in float64 on the CPU from the rounded inputs, and each element of the device result must satisfy

```
|C - R|  <=  g * (|A| @ |B|)  +  u_out * ( |R| + g * (|A| @ |B|) )

g      = n * u / (1 - n * u)        Higham's inner product bound for n terms
u      = 2^-24                      unit roundoff of the accumulator (FP32 assumed)
u_out  = 2^-24, 2^-11 or 2^-8       unit roundoff of the output type (fp32, fp16, bf16)
```

The first term is the largest error any rounding sequence of the dot product can produce (`n` here is the inner dimension K); the second is the final rounding to the output type. Any element outside the bound, and any NaN or Inf, is a wrong answer, not a tolerance judgement.

For int8 the device returns int32 and the reference is computed exactly in int64, so the bound is zero: any difference at all is a wrong answer.

**2. Repeat check.** Every run is compared bit for bit with the first run on the same device. A single flipped bit anywhere in the output fails this check, even when it is far too small to leave the reference bound.

**Verdict per precision**

| Verdict | Meaning |
|---|---|
| `no-silent-errors` | Every run inside the bound and bit-identical to run 0 |
| `silent-data-corruption` | Some runs differ from run 0 (intermittent fault), possibly also outside the bound |
| `outside-error-bound` | All runs agree with each other but sit outside the proven bound: a faulty unit, a device accumulating below FP32, or an int8 result that is not exact |
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

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python screen.py                 # compute probe; auto picks cuda, then mps, then cpu; 3 shapes x 4 precisions
python screen.py --inject 5      # compute self-test
python memcheck.py               # memory probe; 6 patterns over most of the device memory
python memcheck.py --inject 5    # memory self-test
python -m pytest -q              # 39 tests on CPU; 2 more validate the output against OCP's schema
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
| `--dtypes` | `fp32,fp16,bf16,int8` | comma list from `fp32`, `fp16`, `bf16`, `int8` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | bit flips to inject per step (self-test) |
| `--out` | `results.jsonl` | OCP output file |

Exit code is 0 for PASS and 1 for FAIL. A precision or shape the device does not support (for example int8 matmul on a device without it) is skipped and recorded as an error artifact.

On CUDA the script turns off TF32 and reduced precision reductions, sets `CUBLAS_WORKSPACE_CONFIG`, and asks PyTorch for deterministic algorithms, so the bound and the repeat check both hold.

To run the schema test, clone [ocp-diag-core](https://github.com/opencomputeproject/ocp-diag-core) and set `OCP_SCHEMA_DIR` to its `json_spec/output` folder.

## Output format

Both probes write one JSON object per line following the [OCP Test and Validation output spec](https://github.com/opencomputeproject/ocp-diag-core/tree/main/json_spec), version 2.0, written through the official `ocptv` library:

- `testRunStart` with the parameters used, the host name as DUT id, the device as a hardware component, and the `torch` and `python` versions as software components
- one test step per shape and precision, named `gemm_<precision>_<MxKxN>` (for example `gemm_fp16_4096x4096x4096`), carrying the measurements `runs`, `shape_mkn`, `exact_check`, `seconds`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `worst_error_over_bound_ratio` (float precisions only), `worst_abs_error`, `worst_nonfinite_values`, and in self-test mode `injected_runs` and `injected_runs_detected`, with validators on the counts that must be zero or equal to the injected count
- a `diagnosis` per step with the verdict above
- `testRunEnd` with the overall PASS or FAIL

`memcheck.py` writes one step per pass (`memory_sweep_pass1`, ...) with `bytes_tested`, `device_memory_bytes`, `coverage_fraction`, `dwell_seconds`, `seconds`, one `bad_words_<pattern>` per pattern (validated to be zero outside self-test), a warning log line per failing pattern with the first byte offsets and the bad-bit mask, and in self-test mode `injected_<pattern>` and `injected_detected_<pattern>`.

## Assumptions and limits

1. The bound assumes the device accumulates in FP32 or better. That is enforced on CUDA by the flags above. On other devices a lower accumulation precision would show up as `outside-error-bound` on every run, and the message says so. It was not the case on the Apple GPU measured above.
2. These are two probes. A PASS means "no silent error in these matrix multiplies and this memory sweep, in these runs, on this device today". It is not a certificate of a healthy chip. The OSDI 2026 results above show that defects depend on the data, the kernel, temperature and age; this version sweeps shapes, precisions and memory but not the rest.
3. Runs take seconds, not hours, so thermal and aging effects are not exercised.
4. One device per run. Multi-device comparison is on the roadmap.
5. The injected faults are applied to the output on the CPU side. They test the checker, not the chip.

## Roadmap

1. Real model layers and full inference as workloads, since faults are data dependent.
2. More input patterns, and fp8 where the hardware has it.
3. Long runs under load so temperature and aging are part of the test.
4. Fault injection inside the computation on the device, so a chip's sensitivity to bit flips can be measured (the basis for a space radiation column).
5. Side by side runs against vendor diagnostics on the same device.
6. A public table with one row per chip, precision and age, and a rulebook for submitting rows.

## Contributing a row

Run `python screen.py`, `python screen.py --inject 5`, `python memcheck.py` and `python memcheck.py --inject 5` on your device, then open a pull request with the four result files, the device name, driver version and PyTorch version. Rows are added only from attached result files.

## Citing

If you use this in research, please cite it as

```
Yasha Khandelwal. ai-chip-integrity: an open tester for silent computation errors in AI chips. Version 0.2.0, 2026.
https://github.com/tech4biz-yasha/ai-chip-integrity
```

A `CITATION.cff` file is included.

## Author and license

Yasha Khandelwal, yasha.khandelwal@tech4biz.io

MIT License. See `LICENSE`.
