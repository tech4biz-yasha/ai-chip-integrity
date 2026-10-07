# ai-chip-integrity

An open tester for silent computation errors in AI chips.

It runs a fixed calculation on a GPU, NPU or CPU many times and proves, element by element, whether the chip returned the right answer every time. Results come out in the Open Compute Project Test and Validation format, so they drop straight into fleet tooling.

**Status: v0.1.0, early.** One operation (matrix multiply), one device at a time, one measured row. Read the limits section before relying on a PASS.

## Why this exists

A chip can pass its vendor diagnostics and still compute wrong answers with no error raised. ByteDance reported at OSDI 2026 that the standard practice of isolating faulty GPUs with synthetic microbenchmarks missed over 60% of defective devices, that defects often appear later through aging, and that the failures are data dependent and unit specific, so a device that passes a general stress test can fail on specific input data ([Zheng et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/zheng)). A second ByteDance and Tsinghua study found 18 silent corruption incidents and 13 faulty GPUs across 35 million GPU hours of production training ([Lei et al., OSDI '26](https://www.usenix.org/conference/osdi26/presentation/lei)).

Those detectors live inside one company's training stack. This project is the open, vendor neutral version: the same idea, runnable by anyone on any chip, with results in a shared format and a public table.

## Measured results

Every row comes from a run of this code with the result file attached. Nothing here is modelled or estimated.

| Device | Date | Matrix | Runs per precision | fp32 | fp16 | Injected faults caught | Verdict |
|---|---|---|---|---|---|---|---|
| Apple GPU via MPS, MacBook Pro (arm64) | 2026-10-07 | 1024 x 1024 | 50 | PASS, 1.6 s | PASS, 1.1 s | 10 of 10 | no-silent-errors |

Notes on the first row: all 100 runs were bit-for-bit identical to the first run, so the matrix multiply on this chip is deterministic, and fp16 stayed inside the FP32-accumulation error bound. The fault column is the self-test described below (software injected, not a property of the chip).

## What it measures

One square matrix multiply `C = A @ B` with fixed inputs, repeated `--iters` times per precision. Inputs are drawn uniformly from [-1, 1) with a fixed seed, then rounded to the test precision, so every precision starts from the same underlying matrices.

Every run is checked two ways.

**1. Reference check.** The reference `R` is computed in float64 on the CPU from the rounded inputs. Each element of the device result must satisfy

```
|C - R|  <=  g * (|A| @ |B|)  +  u_out * ( |R| + g * (|A| @ |B|) )

g      = n * u / (1 - n * u)        Higham's inner product bound for n terms
u      = 2^-24                      unit roundoff of the accumulator (FP32 assumed)
u_out  = 2^-24, 2^-11 or 2^-8       unit roundoff of the output type (fp32, fp16, bf16)
```

The first term is the largest error any rounding sequence of the dot product can produce; the second is the final rounding to the output type. Any element outside the bound, and any NaN or Inf, is a wrong answer, not a tolerance judgement.

**2. Repeat check.** Every run is compared bit for bit with the first run on the same device. A single flipped bit anywhere in the output fails this check, even when it is far too small to leave the reference bound.

**Verdict per precision**

| Verdict | Meaning |
|---|---|
| `no-silent-errors` | Every run inside the bound and bit-identical to run 0 |
| `silent-data-corruption` | Some runs differ from run 0 (intermittent fault), possibly also outside the bound |
| `outside-error-bound` | All runs agree with each other but sit outside the proven bound: a faulty unit, or a device accumulating below FP32 |
| `nondeterministic-kernel` | At least 3 runs and at least 10% of runs differ from run 0, yet all stay inside the bound. The repeat check cannot be used on that device and precision |

**Self-test.** `--inject N` flips one random bit of one random output element in `N` runs (never run 0), after the result has been copied back from the device. It proves the checker catches corruption; it does not stress the chip. The verdict then becomes `injection-self-test-pass` only if every injected run was caught and no uninjected run failed.

## Run it

```
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python screen.py                 # auto picks cuda, then mps, then cpu
python screen.py --inject 5      # self-test
python -m pytest -q              # 28 tests on CPU; a 29th validates the output against OCP's schema
```

| Option | Default | Meaning |
|---|---|---|
| `--device` | `auto` | `cpu`, `mps`, `cuda` or `cuda:N` |
| `--size` | `1024` | square matrix size `n` |
| `--iters` | `50` | runs per precision |
| `--dtypes` | `fp32,fp16` | comma list from `fp32`, `fp16`, `bf16` |
| `--seed` | `1234` | input seed |
| `--inject` | `0` | bit flips to inject per precision (self-test) |
| `--out` | `results.jsonl` | OCP output file |

Exit code is 0 for PASS and 1 for FAIL. A precision the device does not support is skipped and recorded as an error artifact.

On CUDA the script turns off TF32 and reduced precision reductions, sets `CUBLAS_WORKSPACE_CONFIG`, and asks PyTorch for deterministic algorithms, so the bound and the repeat check both hold.

To run the schema test, clone [ocp-diag-core](https://github.com/opencomputeproject/ocp-diag-core) and set `OCP_SCHEMA_DIR` to its `json_spec/output` folder.

## Output format

`results.jsonl` holds one JSON object per line following the [OCP Test and Validation output spec](https://github.com/opencomputeproject/ocp-diag-core/tree/main/json_spec), version 2.0, written through the official `ocptv` library:

- `testRunStart` with the parameters used, the host name as DUT id, the device as a hardware component, and the `torch` and `python` versions as software components
- one test step per precision (`gemm_fp32`, `gemm_fp16`, `gemm_bf16`) carrying the measurements `runs`, `matrix_size`, `seconds`, `reference_check_failed_runs`, `repeat_check_failed_runs`, `worst_error_over_bound_ratio`, `worst_abs_error`, `worst_nonfinite_values`, and in self-test mode `injected_runs` and `injected_runs_detected`, with validators on the counts that must be zero or equal to the injected count
- a `diagnosis` per step with the verdict above
- `testRunEnd` with the overall PASS or FAIL

## Assumptions and limits

1. The bound assumes the device accumulates in FP32 or better. That is enforced on CUDA by the flags above. On other devices a lower accumulation precision would show up as `outside-error-bound` on every run, and the message says so. It was not the case on the Apple GPU measured above.
2. This is one operation. A PASS means "no silent error in this matrix multiply, in these runs, on this device today". It is not a certificate of a healthy chip. The OSDI 2026 results above show that defects depend on the data, the kernel, temperature and age, none of which this version sweeps.
3. Runs take seconds, not hours, so thermal and aging effects are not exercised.
4. One device per run. Multi-device comparison is on the roadmap.
5. The injected faults are applied to the output on the CPU side. They test the checker, not the chip.

## Roadmap

1. Real model layers and full inference as workloads, since faults are data dependent.
2. Sweeps across sizes, input patterns and precisions.
3. Long runs under load so temperature and aging are part of the test.
4. Fault injection inside the computation on the device, so a chip's sensitivity to bit flips can be measured (the basis for a space radiation column).
5. Side by side runs against vendor diagnostics on the same device.
6. A public table with one row per chip, precision and age, and a rulebook for submitting rows.

## Contributing a row

Run `python screen.py` and `python screen.py --inject 5` on your device, then open a pull request with the two `results.jsonl` files, the device name, driver version and PyTorch version. Rows are added only from attached result files.

## Citing

If you use this in research, please cite it as

```
Yasha Khandelwal. ai-chip-integrity: an open tester for silent computation errors in AI chips. Version 0.1.0, 2026.
https://github.com/tech4biz-yasha/ai-chip-integrity
```

A `CITATION.cff` file is included.

## Author and license

Yasha Khandelwal, yasha.khandelwal@tech4biz.io

MIT License. See `LICENSE`.
