# ai-chip-integrity

An open tester for silent computation errors in AI chips.

It runs a fixed calculation on a GPU, NPU or CPU many times and proves, element by element, whether the chip returned the right answer every time. Results come out in the Open Compute Project Test and Validation format, so they drop straight into fleet tooling.

**Status: v0.2.0, early.** One operation (matrix multiply) across several shapes and four precisions, one device at a time, one measured row. Read the limits section before relying on a PASS.

## Why this exists

A chip can pass its vendor diagnostics and still compute wrong answers with no error raised. ByteDance reported at OSDI 2026 that the standard practice of isolating faulty GPUs with synthetic microbenchmarks missed over 60% of defective devices, that defects often appear later through aging, and that the failures are data dependent and unit specific, so a device that passes a general stress test can fail on specific input data ([Zheng et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zheng)). A second ByteDance and Tsinghua study found 18 silent corruption incidents and 13 faulty GPUs across 35 million GPU hours of production training ([Lei et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/lei)).

Those detectors live inside one company's training stack. This project is the open, vendor neutral version: the same idea, runnable by anyone on any chip, with results in a shared format and a public table.

## Measured results

Every row comes from a run of this code with the result files in [`results/`](results/). Nothing here is modelled or estimated.

| Device | Date | Tool | Shapes (MxKxN) | Runs per step | fp32 | fp16 | bf16 | int8 | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.2.0 | 1024x1024x1024, 4096x4096x4096, 32x4096x11008 | 50 | PASS | PASS | PASS | skipped, no int8 matmul on MPS | 45 of 45 | no-silent-errors |
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | v0.1.0 | 1024x1024x1024 | 50 | PASS, 1.6 s | PASS, 1.1 s | not run | not run | 10 of 10 | no-silent-errors |

Notes on the Apple rows: in the v0.2.0 run all 450 results across 9 steps were bit-for-bit identical to their first run, so the matrix multiply on this chip is deterministic, and fp16 and bf16 stayed inside the FP32-accumulation error bound at every shape. The 4096x4096x4096 steps took 23 to 30 s each for 50 runs. The fault column is the self-test described below (software injected, not a property of the chip). Result files: `results/mac_v020.jsonl` and `results/mac_v020_inject.jsonl`.

## What it measures

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

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python screen.py                 # auto picks cuda, then mps, then cpu; 3 shapes x 4 precisions
python screen.py --inject 5      # self-test
python -m pytest -q              # 32 tests on CPU; a 33rd validates the output against OCP's schema
```

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

`results.jsonl` holds one JSON object per line following the [OCP Test and Validation output spec](https://github.com/opencomputeproject/ocp-diag-core/tree/main/json_spec), version 2.0, written through the official `ocptv` library:

- `testRunStart` with the parameters used, the host name as DUT id, the device as a hardware component, and the `torch` and `python` versions as software components
- one test step per shape and precision, named `gemm_<precision>_<MxKxN>` (for example `gemm_fp16_4096x4096x4096`), carrying the measurements `runs`, `shape_mkn`, `exact_check`, `seconds`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `worst_error_over_bound_ratio` (float precisions only), `worst_abs_error`, `worst_nonfinite_values`, and in self-test mode `injected_runs` and `injected_runs_detected`, with validators on the counts that must be zero or equal to the injected count
- a `diagnosis` per step with the verdict above
- `testRunEnd` with the overall PASS or FAIL

## Assumptions and limits

1. The bound assumes the device accumulates in FP32 or better. That is enforced on CUDA by the flags above. On other devices a lower accumulation precision would show up as `outside-error-bound` on every run, and the message says so. It was not the case on the Apple GPU measured above.
2. This is one operation. A PASS means "no silent error in these matrix multiplies, in these runs, on this device today". It is not a certificate of a healthy chip. The OSDI 2026 results above show that defects depend on the data, the kernel, temperature and age; this version sweeps shapes and precisions but not the rest.
3. Runs take seconds, not hours, so thermal and aging effects are not exercised.
4. One device per run. Multi-device comparison is on the roadmap.
5. The injected faults are applied to the output on the CPU side. They test the checker, not the chip.

## Roadmap

1. Real model layers and full inference as workloads, since faults are data dependent.
2. More input patterns, fp8 where the hardware has it, and a full memory sweep.
3. Long runs under load so temperature and aging are part of the test.
4. Fault injection inside the computation on the device, so a chip's sensitivity to bit flips can be measured (the basis for a space radiation column).
5. Side by side runs against vendor diagnostics on the same device.
6. A public table with one row per chip, precision and age, and a rulebook for submitting rows.

## Contributing a row

Run `python screen.py` and `python screen.py --inject 5` on your device, then open a pull request with the two `results.jsonl` files, the device name, driver version and PyTorch version. Rows are added only from attached result files.

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
