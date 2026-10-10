"""
Tests for patterns.py. Run from the repo root:  python -m pytest -q
All tests run on the CPU.
"""

import json
import math
import os
import pathlib
import random
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import arith  # noqa: E402
import patterns  # noqa: E402
import screen  # noqa: E402

SMALL = "64x64x64,16x128x48"


def strict_load(path):
    def no_constants(tok):
        raise ValueError(f"non-standard JSON constant {tok}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=no_constants) for line in f if line.strip()]


def by_step(objs):
    names, diag, meas = {}, {}, {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        sid = sa["testStepId"]
        if "testStepStart" in sa:
            names[sid] = sa["testStepStart"]["name"]
        if "diagnosis" in sa:
            diag[names[sid]] = sa["diagnosis"]
        if "measurement" in sa:
            meas.setdefault(names[sid], {})[sa["measurement"]["name"]] = sa["measurement"]["value"]
    return diag, meas


@pytest.fixture(autouse=True)
def clear_hook():
    screen._FAULT_HOOK = None
    yield
    screen._FAULT_HOOK = None
    torch.set_flush_denormal(False)


def test_every_pattern_passes_on_cpu(tmp_path):
    out = tmp_path / "p.jsonl"
    rc = patterns.main(["--device", "cpu", "--shapes", SMALL, "--iters", "3", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 0
    expected = 2 * sum(len(v) for v in patterns.APPLIES.values())
    assert len(diag) == expected
    for name, d in diag.items():
        assert d["verdict"] == "no-silent-errors", name


@pytest.mark.parametrize("dt", ["fp32", "fp16", "bf16"])
def test_float_patterns_have_their_shape_and_stay_in_range(dt):
    dtype, out_dtype = screen.DTYPES[dt][0], screen.DTYPES[dt][1]
    top = min(torch.finfo(out_dtype).max, torch.finfo(torch.float32).max)
    for pattern in patterns.FLOAT_PATTERNS:
        (a, b, ref, abs_ab), shift = patterns.make_pattern_case(dt, pattern, 32, 256, 24, seed=9)
        a64, b64 = a.to(torch.float64), b.to(torch.float64)
        assert float(abs_ab.max()) <= top / 2, pattern                 # no partial sum can overflow, any order
        if pattern == "near_max":
            assert float(abs_ab.max()) > top / 4
        if pattern == "mantissa":
            v = a64[a64 != 0].abs()
            frac = v / torch.pow(2.0, torch.floor(torch.log2(v)))
            assert torch.all(frac == 2.0 - 2.0 ** (1 - patterns.SIGNIFICAND_BITS[dtype]))
        if pattern == "cancel":
            assert float(ref.abs().max()) <= arith.reference_dot(256) * float(abs_ab.max())
        if pattern == "alternate":
            assert torch.all(a64 * patterns._checker(32, 256) > 0)
        if pattern == "sparse":
            assert float((a64 == 0).double().mean()) > 0.85
        if pattern == "subnormal":
            tiny = torch.finfo(dtype).tiny
            assert torch.all((a64.abs() < tiny) & (a64 != 0))
            assert float(a64.abs().min() * b64.abs().min()) >= torch.finfo(torch.float32).tiny  # products normal


def test_integer_and_fp8_patterns_keep_exact_values():
    (a, b, _, _), _ = patterns.make_pattern_case("int8", "extremes", 16, 64, 16, seed=2)
    assert set(a.unique().tolist()) <= {-128, -127, 0, 127}
    for pattern in ("alternate", "sparse"):
        (a, b, ref, abs_ab), _ = patterns.make_pattern_case("fp8", pattern, 16, 4096, 16, seed=2)
        assert set(a.float().unique().tolist()) <= {-1.0, 0.0, 1.0}
        assert float(ref.abs().max()) <= 4096 and float(abs_ab.abs().max()) == 0.0


def test_patterns_that_do_not_apply_are_refused():
    with pytest.raises(RuntimeError):
        patterns.make_pattern_case("fp8", "wide", 16, 64, 16, seed=1)


@pytest.fixture
def flushing_device(monkeypatch):
    """Make the CPU behave like a device that flushes subnormal inputs, inside the matmul only, so the host still
    builds the inputs and the reference with subnormals intact."""
    if not torch.set_flush_denormal(True):
        pytest.skip("this CPU cannot flush subnormals")
    torch.set_flush_denormal(False)
    real = screen.matmul

    def flushing(a, b, fast_accum=False):
        threads = torch.get_num_threads()
        torch.set_num_threads(1)                      # keep the work on this thread, where the flag is set
        torch.set_flush_denormal(True)
        try:
            return real(a, b, fast_accum).clone()
        finally:
            torch.set_flush_denormal(False)
            torch.set_num_threads(threads)
    monkeypatch.setattr(screen, "matmul", flushing)
    if not patterns.flushes_subnormal_inputs(torch.device("cpu"), "fp32"):
        pytest.skip("this CPU's matmul ignores flush-to-zero")


def test_flushing_device_is_detected_and_judged_against_the_flushed_model(tmp_path, flushing_device):
    out = tmp_path / "p.jsonl"
    rc = patterns.main(["--device", "cpu", "--shapes", "32x256x32", "--iters", "3", "--dtypes", "fp32",
                        "--patterns", "subnormal", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    step = "gemm_fp32_subnormal_32x256x32"
    assert rc == 0 and diag[step]["verdict"] == "no-silent-errors"
    assert meas[step]["subnormal_inputs_flushed"] is True


def run_one_step(tmp_path, name, shape, case, allow_flush):
    import ocptv.output as tv
    writer = screen.FileWriter(str(tmp_path / "x.jsonl"))
    tv.config(writer=writer)
    run = tv.TestRun(name="t", version="0")
    dut = tv.Dut(id="t")
    hw = dut.add_hardware_info(name="cpu")
    run.start(dut=dut)
    step = run.add_step("s")
    step.start()
    ok, verdict, _, _ = screen.screen_case(step, torch.device("cpu"), name, shape, 3, 1234, 0,
                                           random.Random(1), hw, case=case, allow_flush=allow_flush)
    step.end(status=tv.TestStatus.COMPLETE)
    run.end(status=tv.TestStatus.COMPLETE, result=tv.TestResult.PASS)
    writer.close()
    return ok, verdict


def test_assuming_the_wrong_flushing_policy_fails(tmp_path, flushing_device):
    """The subnormal check is tight: judged against the kept-subnormals model, a flushing device fails."""
    case, _ = patterns.make_pattern_case("fp32", "subnormal", 32, 256, 32, seed=1234, flushed=False)
    assert float(case[2].abs().max()) > 0                          # the reference really is non-zero
    ok, verdict = run_one_step(tmp_path, "fp32", (32, 256, 32), case, allow_flush=False)
    assert not ok and verdict == "outside-error-bound"


def test_injected_flips_are_caught_on_every_pattern(tmp_path):
    out = tmp_path / "p.jsonl"
    rc = patterns.main(["--device", "cpu", "--shapes", "32x64x32", "--iters", "5", "--inject", "3", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 0
    for name, d in diag.items():
        assert d["verdict"] == "injection-self-test-pass", name
        assert meas[name]["injected_runs_detected"] == 3


def test_systematic_error_on_a_cancelling_product_is_caught(tmp_path):
    """The exact answer of the cancel pattern is zero; a stuck offset four times the proven bound is caught.
    (Worst-case bounds scale with the size of the terms, not of the answer, so smaller offsets can hide.)"""
    (a, b, ref, abs_ab), _ = patterns.make_pattern_case("fp32", "cancel", 32, 256, 32, seed=1234)
    bound = screen.error_bound(ref, abs_ab, 256, 2.0 ** -24, g=screen.accumulation_factor("fp32", 256, "cpu"))
    offset = 4.0 * float(bound.view(-1)[3])

    def hook(i, c):
        c.view(-1)[3] += offset
    screen._FAULT_HOOK = hook
    out = tmp_path / "p.jsonl"
    rc = patterns.main(["--device", "cpu", "--shapes", "32x256x32", "--iters", "3", "--dtypes", "fp32",
                        "--patterns", "cancel", "--out", str(out)])
    diag, _ = by_step(strict_load(out))
    assert rc == 1 and diag["gemm_fp32_cancel_32x256x32"]["verdict"] == "outside-error-bound"


def test_near_max_never_overflows_on_device(tmp_path):
    out = tmp_path / "p.jsonl"
    patterns.main(["--device", "cpu", "--shapes", "32x512x32", "--iters", "2", "--dtypes", "fp32,fp16,bf16",
                   "--patterns", "near_max,wide", "--out", str(out)])
    _, meas = by_step(strict_load(out))
    assert all(v["worst_nonfinite_values"] == 0 for v in meas.values())
    assert any(v["input_scale_power_of_two"] > 100 for v in meas.values())          # fp32 really reaches the top


def test_bad_args_rejected(tmp_path):
    with pytest.raises(SystemExit):
        patterns.main(["--device", "cpu", "--patterns", "zigzag", "--out", str(tmp_path / "p.jsonl")])


@pytest.mark.skipif(not os.environ.get("OCP_SCHEMA_DIR"), reason="set OCP_SCHEMA_DIR to ocp-diag-core/json_spec/output")
def test_output_matches_official_ocp_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schema_dir = pathlib.Path(os.environ["OCP_SCHEMA_DIR"])
    resources = []
    for p in schema_dir.glob("*.json"):
        doc = json.loads(p.read_text())
        resources.append((doc["$id"], referencing.Resource.from_contents(doc)))
    registry = referencing.Registry().with_resources(resources)
    validator = jsonschema.Draft202012Validator(json.loads((schema_dir / "root.json").read_text()), registry=registry)
    out = tmp_path / "p.jsonl"
    patterns.main(["--device", "cpu", "--shapes", "32x64x32", "--iters", "3", "--inject", "1", "--out", str(out)])
    for obj in strict_load(out):
        validator.validate(obj)


def test_wide_pattern_spans_many_binades():
    (a, _, _, _), _ = patterns.make_pattern_case("fp32", "wide", 32, 256, 32, seed=4)
    v = a.to(torch.float64).abs()
    span = math.log2(float(v.max())) - math.log2(float(v[v > 0].min()))
    assert span > 60
