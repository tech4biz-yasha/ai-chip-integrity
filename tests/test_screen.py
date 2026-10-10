"""
Tests for screen.py. Run from the repo root:  python -m pytest -q
All tests run on the CPU so they work on any machine.
"""

import json
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import screen  # noqa: E402

BASE = ["--device", "cpu", "--size", "96", "--iters", "8"]
SQ = "96x96x96"


def strict_load(path):
    """Parse every line as strict JSON (NaN or Infinity tokens fail the test)."""
    def no_constants(tok):
        raise ValueError(f"non-standard JSON constant {tok}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=no_constants) for line in f if line.strip()]


def diagnoses(objs):
    out = {}
    step_names = {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        if "testStepStart" in sa:
            step_names[sa["testStepId"]] = sa["testStepStart"]["name"]
        if "diagnosis" in sa:
            out[step_names[sa["testStepId"]]] = sa["diagnosis"]
    return out


def measurement(objs, step, name):
    step_names = {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        if "testStepStart" in sa:
            step_names[sa["testStepId"]] = sa["testStepStart"]["name"]
        m = sa.get("measurement")
        if m and m["name"] == name and step_names.get(sa["testStepId"]) == step:
            return m["value"]
    raise KeyError(name)


def run_end(objs):
    return [o for o in objs if "testRunArtifact" in o and "testRunEnd" in o["testRunArtifact"]][0][
        "testRunArtifact"]["testRunEnd"]


@pytest.fixture(autouse=True)
def clear_hook():
    screen._FAULT_HOOK = None
    yield
    screen._FAULT_HOOK = None


def test_clean_run_passes_all_dtypes(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32,fp16,bf16,int8,fp8,fp8fast", "--out", str(out)])
    objs = strict_load(out)
    d = diagnoses(objs)
    assert rc == 0
    assert run_end(objs)["result"] == "PASS"
    for dt in ("fp32", "fp16", "bf16", "int8", "fp8", "fp8fast"):
        name = f"gemm_{dt}_{SQ}"
        assert d[name]["verdict"] == "no-silent-errors"
        assert measurement(objs, name, "reference_check_failed_runs") == 0
        assert measurement(objs, name, "repeat_check_failed_runs") == 0
        if dt not in ("int8", "fp8", "fp8fast"):
            assert measurement(objs, name, "worst_error_over_bound_ratio") <= 1.0
        else:
            assert measurement(objs, name, "exact_check") is True
            assert measurement(objs, name, "worst_abs_error") == 0


def test_injected_bit_flips_are_all_detected(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32,fp16,bf16,int8", "--inject", "5", "--out", str(out)])
    objs = strict_load(out)
    d = diagnoses(objs)
    assert rc == 0
    for dt in ("fp32", "fp16", "bf16", "int8"):
        name = f"gemm_{dt}_{SQ}"
        assert d[name]["verdict"] == "injection-self-test-pass"
        assert measurement(objs, name, "injected_runs") == 5
        assert measurement(objs, name, "injected_runs_detected") == 5


@pytest.mark.parametrize("seed", range(20))
def test_every_bit_position_is_detected(tmp_path, seed):
    """Many seeds so the random flips cover sign, exponent and low mantissa bits."""
    out = tmp_path / "r.jsonl"
    rc = screen.main(["--device", "cpu", "--size", "32", "--iters", "12", "--dtypes", "fp32,fp16,int8",
                      "--inject", "11", "--seed", str(seed), "--out", str(out)])
    assert rc == 0


def test_intermittent_corruption_is_flagged(tmp_path):
    def hook(i, c):
        if i == 3:
            c.view(-1)[7] += 5.0
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert diagnoses(objs)[f"gemm_fp32_{SQ}"]["verdict"] == "silent-data-corruption"
    assert run_end(objs)["result"] == "FAIL"


def test_tiny_intermittent_change_inside_bound_is_still_flagged(tmp_path):
    """A one-ulp change on a single run stays inside the bound, so only the repeat check sees it."""
    def hook(i, c):
        if i == 5:
            v = c.view(torch.int32).view(-1)
            v[11] = v[11] ^ 1
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert measurement(objs, f"gemm_fp32_{SQ}", "reference_check_failed_runs") == 0
    assert measurement(objs, f"gemm_fp32_{SQ}", "repeat_check_failed_runs") == 1
    assert diagnoses(objs)[f"gemm_fp32_{SQ}"]["verdict"] == "silent-data-corruption"


def test_systematic_error_is_flagged_outside_bound(tmp_path):
    def hook(i, c):
        c.view(-1)[0] += 1.0
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert diagnoses(objs)[f"gemm_fp32_{SQ}"]["verdict"] == "outside-error-bound"


def test_widespread_tiny_differences_read_as_nondeterministic(tmp_path):
    def hook(i, c):
        if i % 2 == 1:
            v = c.view(torch.int32).view(-1)
            v[3] = v[3] ^ 1
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert diagnoses(objs)[f"gemm_fp32_{SQ}"]["verdict"] == "nondeterministic-kernel"


def test_nan_output_is_flagged_and_json_stays_valid(tmp_path):
    def hook(i, c):
        if i == 2:
            c.view(-1)[0] = float("nan")
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "fp32", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert measurement(objs, f"gemm_fp32_{SQ}", "worst_nonfinite_values") == 1
    assert diagnoses(objs)[f"gemm_fp32_{SQ}"]["verdict"] == "silent-data-corruption"


def test_int8_off_by_one_is_flagged(tmp_path):
    """Integer results are exact, so a single off-by-one on every run is a wrong answer."""
    def hook(i, c):
        c.view(-1)[5] += 1
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", "int8", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    assert diagnoses(objs)[f"gemm_int8_{SQ}"]["verdict"] == "outside-error-bound"


@pytest.mark.parametrize("dt", ["fp8", "fp8fast"])
def test_fp8_off_by_one_is_flagged(tmp_path, dt):
    """FP8 inputs from {-1, 0, 1} give an exact answer, so a single off-by-one is a wrong answer."""
    def hook(i, c):
        c.view(-1)[7] += 1.0
    screen._FAULT_HOOK = hook
    out = tmp_path / "r.jsonl"
    rc = screen.main(BASE + ["--dtypes", dt, "--out", str(out)])
    objs = strict_load(out)
    assert rc == 1
    d = diagnoses(objs)[f"gemm_{dt}_{SQ}"]
    assert d["verdict"] == "outside-error-bound"
    assert "13 significant bits" in d["message"]


def test_fp8_inputs_are_ternary_and_exact():
    a, b, ref, abs_ab = screen.make_case(32, 4096, 48, screen.FP8, seed=3)
    assert a.dtype == screen.FP8 and b.dtype == screen.FP8
    assert set(a.float().unique().tolist()) <= {-1.0, 0.0, 1.0}
    assert float(abs_ab.abs().max()) == 0.0                     # exact: the bound is zero
    assert float(ref.abs().max()) <= 4096


def test_fp8_with_too_large_k_is_skipped_not_misjudged(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = screen.main(["--device", "cpu", "--shapes", "16x8192x16", "--iters", "3", "--dtypes", "fp8", "--out", str(out)])
    objs = strict_load(out)
    errors = [o["testStepArtifact"]["error"] for o in objs if "error" in o.get("testStepArtifact", {})]
    assert rc == 0 and errors and "K <= 4096" in errors[0]["message"]


def test_rectangular_shapes_and_injection(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = screen.main(["--device", "cpu", "--shapes", "32x128x72,40x64x64", "--iters", "6",
                      "--dtypes", "fp32,int8", "--inject", "2", "--out", str(out)])
    objs = strict_load(out)
    d = diagnoses(objs)
    assert rc == 0
    for name in ("gemm_fp32_32x128x72", "gemm_int8_32x128x72", "gemm_fp32_40x64x64", "gemm_int8_40x64x64"):
        assert d[name]["verdict"] == "injection-self-test-pass"
    assert measurement(objs, "gemm_fp32_32x128x72", "shape_mkn") == "32x128x72"


def test_bad_shape_is_rejected(tmp_path):
    with pytest.raises(SystemExit):
        screen.main(["--device", "cpu", "--shapes", "12xab", "--out", str(tmp_path / "r.jsonl")])


def test_parse_shapes():
    assert screen.parse_shapes("1024, 32x4096x11008") == [(1024, 1024, 1024), (32, 4096, 11008)]


def test_bound_is_tight_enough_to_matter():
    """The fp32 bound must be far smaller than the values, or the reference check is useless."""
    a, b, ref, abs_ab = screen.make_case(256, 256, 256, torch.float32, 1)
    bound = screen.error_bound(ref, abs_ab, 256, screen.DTYPES["fp32"][4])
    assert float((bound / abs_ab).max()) < 1e-3


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
    root = json.loads((schema_dir / "root.json").read_text())
    validator = jsonschema.Draft202012Validator(root, registry=registry)

    out = tmp_path / "r.jsonl"
    screen.main(BASE + ["--dtypes", "fp32,fp16,int8", "--inject", "2", "--out", str(out)])
    for obj in strict_load(out):
        validator.validate(obj)
