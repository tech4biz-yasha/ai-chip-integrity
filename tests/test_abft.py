"""
Tests for abft.py (probe 6). Run from the repo root:  python -m pytest -q
All tests run on the CPU.
"""

import json
import os
import pathlib
import random
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import abft  # noqa: E402
from chip_integrity import patterns  # noqa: E402
from chip_integrity import screen  # noqa: E402


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def by_step(objs):
    names, diag, meas = {}, {}, {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        sid = sa["testStepId"]
        if "testStepStart" in sa:
            names[sid] = sa["testStepStart"]["name"]
        if names.get(sid) == "device_error_counters":
            continue
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


def setup(name="fp32", shape=(48, 96, 40), seed=3, case=None):
    m, k, n = shape
    dtype = screen.DTYPES[name][0]
    a, b, _, _ = case if case is not None else screen.make_case(m, k, n, dtype, seed)
    ac, br, d_col, d_row, eps_col, eps_row, sc, sr = abft.augment(a, b)
    thr_row, thr_col, off_row, off_col, beta, ref = abft.thresholds(a, b, ac, br, d_col, d_row, eps_col, eps_row,
                                                                    name, k, "cpu", sc, sr)
    cf = screen.matmul(ac, br)
    scale = (sc, sr)
    return cf, thr_row * 2.0 ** sr, thr_col * 2.0 ** sc, off_row * 2.0 ** sr, off_col * 2.0 ** sc, m, n, scale


def check_scaled(cf, m, n, thr_row, thr_col, off_row, off_col, scale):
    """abft.check, with thresholds and offsets given in the units of C (setup returns them that way)."""
    sc, sr = scale
    return abft.check(cf, m, n, thr_row * 2.0 ** -sr, thr_col * 2.0 ** -sc, off_row * 2.0 ** -sr, off_col * 2.0 ** -sc, sc, sr)


def test_checksums_are_scaled_to_stay_in_range():
    """Near the top of the range a plain checksum would overflow; the probe scales it by a power of two."""
    case, _ = patterns.make_pattern_case("fp16", "near_max", 32, 128, 24, seed=5)
    *_, sc, sr = abft.augment(case[0], case[1])
    assert sc > 0 and sr > 0


def test_clean_runs_pass_every_precision(tmp_path):
    out = tmp_path / "a.jsonl"
    rc = abft.main(["--device", "cpu", "--shapes", "64x96x48,16x128x272", "--iters", "4", "--out", str(out)])
    diag, meas = by_step(load(out))
    assert rc == 0 and len(diag) == 6
    for name, d in diag.items():
        assert d["verdict"] == "no-silent-errors", name
        assert meas[name]["worst_row_residual_over_threshold"] <= 1.0
        assert 0 < meas[name]["flop_overhead"] < 0.1


@pytest.mark.parametrize("seed", range(4))
def test_injected_errors_are_located_to_row_and_column(tmp_path, seed):
    out = tmp_path / "a.jsonl"
    rc = abft.main(["--device", "cpu", "--shapes", "64x96x48", "--iters", "6", "--inject", "4", "--seed", str(seed), "--out", str(out)])
    diag, meas = by_step(load(out))
    assert rc == 0
    for name, d in diag.items():
        assert d["verdict"] == "injection-self-test-pass", name
        assert meas[name]["injected_runs_located"] == 4


def test_a_single_error_points_at_its_element():
    cf, thr_row, thr_col, off_row, off_col, m, n, scale = setup()
    cf[7, 11] += 10.0 * float(max(thr_row[7], thr_col[11]))
    rows, cols, *_ = check_scaled(cf, m, n, thr_row, thr_col, off_row, off_col, scale)
    assert rows == [7] and cols == [11]


def test_a_fault_in_a_checksum_flags_only_its_line():
    cf, thr_row, thr_col, off_row, off_col, m, n, scale = setup()
    cf[5, n] += 10.0 * float(thr_row[5])                       # the row checksum itself is wrong
    rows, cols, *_ = check_scaled(cf, m, n, thr_row, thr_col, off_row, off_col, scale)
    assert rows == [5] and cols == []


@pytest.mark.parametrize("name", ["fp32", "fp16", "bf16"])
@pytest.mark.parametrize("pattern", ["alternate", "wide", "near_max", "cancel"])
def test_thresholds_hold_on_adversarial_patterns(name, pattern):
    """No false alarm when every sum has the same sign, values span many binades, or sums sit near the top."""
    case, _ = patterns.make_pattern_case(name, pattern, 32, 128, 24, seed=5)
    cf, thr_row, thr_col, off_row, off_col, m, n, scale = setup(name, (32, 128, 24), case=case)
    rows, cols, wr, wc, finite = check_scaled(cf, m, n, thr_row, thr_col, off_row, off_col, scale)
    assert finite and rows == [] and cols == [], (wr, wc)


def test_the_blind_spot_is_measured(tmp_path):
    """A 2x2 rectangle of errors keeps every row and column sum, so ABFT cannot see it. The reference check
    can, and the probe counts the run as an ABFT blind spot instead of passing it."""
    case = screen.make_case(32, 64, 24, torch.float32, 1234)
    _, thr_row, thr_col, _, _, _, _, _ = setup("fp32", (32, 64, 24), case=case)
    d = 50.0 * float(max(thr_row.max(), thr_col.max()))

    def rectangle(i, c):
        c[1, 2] += d; c[1, 9] -= d; c[6, 2] -= d; c[6, 9] += d
    screen._FAULT_HOOK = rectangle
    out = tmp_path / "a.jsonl"
    rc = abft.main(["--device", "cpu", "--shapes", "32x64x24", "--iters", "3", "--dtypes", "fp32", "--out", str(out)])
    diag, meas = by_step(load(out))
    step = "abft_fp32_32x64x24"
    assert rc == 1
    assert meas[step]["checksum_check_failed_runs"] == 0 and meas[step]["reference_check_failed_runs"] == 3
    assert meas[step]["checksum_missed_runs"] == 3 and "blind spot" in diag[step]["message"]


def test_nan_is_flagged():
    cf, thr_row, thr_col, off_row, off_col, m, n, scale = setup()
    cf[2, 3] = float("nan")
    rows, cols, _, _, finite = check_scaled(cf, m, n, thr_row, thr_col, off_row, off_col, scale)
    assert not finite and 2 in rows and 3 in cols


def test_bad_args_rejected(tmp_path):
    with pytest.raises(SystemExit):
        abft.main(["--device", "cpu", "--dtypes", "int8", "--out", str(tmp_path / "a.jsonl")])


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
    out = tmp_path / "a.jsonl"
    abft.main(["--device", "cpu", "--shapes", "32x64x24", "--iters", "4", "--inject", "2", "--out", str(out)])
    for obj in load(out):
        validator.validate(obj)


def test_random_seed_does_not_matter_for_location():
    rng = random.Random(9)
    cf, thr_row, thr_col, off_row, off_col, m, n, scale = setup(seed=8)
    for _ in range(10):
        i, j = rng.randrange(m), rng.randrange(n)
        c = cf.clone()
        c[i, j] += 8.0 * float(max(thr_row[i], thr_col[j]))
        rows, cols, *_ = check_scaled(c, m, n, thr_row, thr_col, off_row, off_col, scale)
        assert (rows, cols) == ([i], [j])
