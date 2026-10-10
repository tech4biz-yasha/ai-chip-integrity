"""
Tests for memcheck.py. Run from the repo root:  python -m pytest -q
All tests run on the CPU with a small buffer so they work on any machine.
"""

import json
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import memcheck  # noqa: E402

BASE = ["--device", "cpu", "--gb", "0.004", "--dwell", "0"]   # about 4 MB, 1 M words


def strict_load(path):
    def no_constants(tok):
        raise ValueError(f"non-standard JSON constant {tok}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=no_constants) for line in f if line.strip()]


def diagnosis(objs):
    for o in objs:
        sa = o.get("testStepArtifact")
        if sa and "diagnosis" in sa:
            return sa["diagnosis"]
    raise KeyError("no diagnosis")


def measurements(objs):
    out = {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if sa and "measurement" in sa:
            out[sa["measurement"]["name"]] = sa["measurement"]["value"]
    return out


def logs(objs):
    return [o["testStepArtifact"]["log"]["message"] for o in objs
            if o.get("testStepArtifact") and "log" in o["testStepArtifact"]]


@pytest.fixture(autouse=True)
def clear_hook():
    memcheck._FAULT_HOOK = None
    yield
    memcheck._FAULT_HOOK = None


def test_clean_sweep_passes(tmp_path):
    out = tmp_path / "m.jsonl"
    rc = memcheck.main(BASE + ["--out", str(out)])
    objs = strict_load(out)
    m = measurements(objs)
    assert rc == 0
    assert diagnosis(objs)["verdict"] == "no-memory-errors"
    for p in memcheck.PATTERNS:
        assert m[f"bad_words_{p}"] == 0
    assert m["bytes_tested"] == (int(0.004 * 2 ** 30) // 4) * 4


def test_injected_flips_are_located(tmp_path):
    out = tmp_path / "m.jsonl"
    rc = memcheck.main(BASE + ["--inject", "7", "--out", str(out)])
    objs = strict_load(out)
    m = measurements(objs)
    assert rc == 0
    assert diagnosis(objs)["verdict"] == "injection-self-test-pass"
    for p in memcheck.PATTERNS:
        assert m[f"injected_{p}"] == 7
        assert m[f"injected_detected_{p}"] == 7


def test_stuck_bit_is_reported_with_offset(tmp_path):
    """A cell whose bit 5 is stuck at 1 must fail on the patterns where that bit should be 0."""
    def hook(pattern, buf, start_word):
        if start_word == 0:
            buf[777] = buf[777] | 32
    memcheck._FAULT_HOOK = hook
    out = tmp_path / "m.jsonl"
    rc = memcheck.main(BASE + ["--out", str(out)])
    objs = strict_load(out)
    m = measurements(objs)
    assert rc == 1
    assert diagnosis(objs)["verdict"] == "memory-errors"
    assert m["bad_words_zeros"] == 1
    assert m["bad_words_ones"] == 0          # bit already 1 in the ones pattern
    assert m["bad_words_5555"] == 1          # 0x55555555 has bit 5 clear
    assert m["bad_words_aaaa"] == 0          # 0xAAAAAAAA has bit 5 set
    msgs = [x for x in logs(objs) if x.startswith("zeros:")]
    assert msgs and f"[{777 * 4}]" in msgs[0] and "0x00000020" in msgs[0]


def test_whole_row_failure_counts_every_word(tmp_path):
    def hook(pattern, buf, start_word):
        if start_word == 0:
            buf[1000:1256] = 0
    memcheck._FAULT_HOOK = hook
    out = tmp_path / "m.jsonl"
    rc = memcheck.main(BASE + ["--out", str(out)])
    m = measurements(strict_load(out))
    assert rc == 1
    assert m["bad_words_ones"] == 256
    assert m["bad_words_zeros"] == 0


def test_hash_patterns_are_deterministic_and_inverse():
    a = memcheck.expected("hash", 0, 4096, torch.device("cpu"), 1)
    b = memcheck.expected("hash", 0, 4096, torch.device("cpu"), 1)
    c = memcheck.expected("hash", 0, 4096, torch.device("cpu"), 2)
    inv = memcheck.expected("hash_inv", 0, 4096, torch.device("cpu"), 1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert torch.equal(a ^ inv, torch.full_like(a, -1))          # bitwise inverse
    assert len(set(a.tolist())) > 4000                           # words are nearly all distinct
    assert torch.equal(memcheck.expected("hash", 1000, 10, torch.device("cpu"), 1),
                       memcheck.expected("hash", 0, 1010, torch.device("cpu"), 1)[1000:])


def test_solid_patterns():
    dev = torch.device("cpu")
    assert int(memcheck.expected("ones", 0, 2, dev, 0)[0]) == -1
    assert int(memcheck.expected("zeros", 0, 2, dev, 0)[0]) == 0
    assert int(memcheck.expected("aaaa", 0, 2, dev, 0)[0]) & 0xFFFFFFFF == 0xAAAAAAAA
    assert int(memcheck.expected("5555", 0, 2, dev, 0)[0]) == 0x55555555


def test_multiple_passes_and_bad_args(tmp_path):
    out = tmp_path / "m.jsonl"
    rc = memcheck.main(BASE + ["--passes", "2", "--out", str(out)])
    objs = strict_load(out)
    assert rc == 0
    sweeps = [o for o in objs if o.get("testStepArtifact") and "diagnosis" in o["testStepArtifact"]
              and o["testStepArtifact"]["diagnosis"]["verdict"] in ("no-memory-errors", "memory-errors")]
    assert len(sweeps) == 2
    with pytest.raises(SystemExit):
        memcheck.main(BASE + ["--inject", "500", "--out", str(out)])


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

    def hook(pattern, buf, start_word):
        if start_word == 0:
            buf[5] = buf[5] | 1
    memcheck._FAULT_HOOK = hook
    out = tmp_path / "m.jsonl"
    memcheck.main(BASE + ["--out", str(out)])
    for obj in strict_load(out):
        validator.validate(obj)
