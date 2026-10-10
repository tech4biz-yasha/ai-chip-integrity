"""
Tests for counters.py (probe 11). No NVIDIA GPU is needed: a simulated NVML stands in for the real library.
Run from the repo root:  python -m pytest -q
"""

import json
import os
import pathlib
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_site  # noqa: E402
import counters  # noqa: E402
import screen  # noqa: E402


class FakeNVML(types.ModuleType):
    """Just enough of pynvml for counters.py, with counters a test can move."""
    NVML_MEMORY_ERROR_TYPE_CORRECTED, NVML_MEMORY_ERROR_TYPE_UNCORRECTED, NVML_VOLATILE_ECC = 0, 1, 0
    NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS, NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR = 0, 1
    NVML_TEMPERATURE_GPU, NVML_CLOCK_SM = 0, 1

    def __init__(self):
        super().__init__("pynvml")
        self.corrected, self.uncorrected, self.rows = 3, 0, [0, 0, 0, 0]

    def nvmlInit(self):
        pass

    def nvmlDeviceGetHandleByUUID(self, uuid):
        return "gpu0"

    def nvmlDeviceGetHandleByIndex(self, index):
        return "gpu0"

    def nvmlDeviceGetEccMode(self, h):
        return [1, 1]

    def nvmlDeviceGetTotalEccErrors(self, h, kind, counter):
        return self.corrected if kind == self.NVML_MEMORY_ERROR_TYPE_CORRECTED else self.uncorrected

    def nvmlDeviceGetRemappedRows(self, h):
        return tuple(self.rows)

    def nvmlDeviceGetRetiredPages(self, h, cause):
        raise RuntimeError("NVML_ERROR_NOT_SUPPORTED")          # A100 and newer remap rows instead

    def nvmlDeviceGetTemperature(self, h, sensor):
        return 47

    def nvmlDeviceGetClockInfo(self, h, clock):
        return 1980

    def nvmlDeviceGetPowerUsage(self, h):
        return 351200


@pytest.fixture
def nvml(monkeypatch):
    fake = FakeNVML()
    monkeypatch.setitem(sys.modules, "pynvml", fake)
    return fake


def by_name(objs):
    names, out = {}, {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        sid = sa["testStepId"]
        if "testStepStart" in sa:
            names[sid] = sa["testStepStart"]["name"]
        entry = out.setdefault(names.get(sid), {})
        if "measurement" in sa:
            entry[sa["measurement"]["name"]] = sa["measurement"]["value"]
        if "diagnosis" in sa:
            entry["_diag"] = sa["diagnosis"]
    return out


def load(path):
    return [json.loads(line) for line in open(path) if line.strip()]


def test_cpu_and_apple_devices_have_no_counters(nvml):
    assert counters.snapshot(torch.device("cpu")) is None
    assert counters.snapshot(torch.device("mps")) is None


def test_snapshot_reads_what_the_device_supports(nvml):
    s = counters.snapshot(torch.device("cuda", 0))
    assert s["ecc_enabled"] is True and s["ecc_corrected_volatile"] == 3 and s["ecc_uncorrected_volatile"] == 0
    assert s["remapped_rows_corrected"] == 0 and s["remap_pending"] is False
    assert "retired_pages_single_bit" not in s                 # unsupported counters are left out, never zero
    assert s["temperature_c"] == 47 and s["sm_clock_mhz"] == 1980 and s["power_w"] == 351.2


def test_missing_library_reads_as_unavailable(monkeypatch):
    monkeypatch.setitem(sys.modules, "pynvml", None)              # import fails, as on a Mac
    assert counters.snapshot(torch.device("cuda", 0)) is None


@pytest.mark.parametrize("move, probe_ok, injected, verdict, ok", [
    ({}, True, False, "counters-quiet", True),
    ({}, False, False, "silent-error-confirmed", False),
    ({"ecc_corrected_volatile": 2}, False, False, "hardware-noticed", False),
    ({"ecc_corrected_volatile": 2}, True, False, "corrected-during-run", True),
    ({"ecc_uncorrected_volatile": 1}, True, False, "uncorrected-during-run", False),
    ({"remapped_rows_uncorrected": 1}, True, False, "uncorrected-during-run", False),
    ({"ecc_corrected_volatile": 2}, False, True, "self-test-run", True),
])
def test_every_cross_check_verdict(move, probe_ok, injected, verdict, ok):
    before = {"ecc_corrected_volatile": 3, "ecc_uncorrected_volatile": 0, "remapped_rows_corrected": 0,
              "remapped_rows_uncorrected": 0, "remap_pending": False, "remap_failure": False}
    after = dict(before)
    for k, d in move.items():
        after[k] += d
    got, got_ok, msg, _ = counters.judge(before, after, probe_ok, injected)
    assert (got, got_ok) == (verdict, ok), msg


def test_unavailable_counters_never_fail_a_run():
    assert counters.judge(None, None, True, False)[:2] == ("counters-unavailable", True)
    assert counters.judge(None, None, False, False)[:2] == ("counters-unavailable", True)


def scripted(monkeypatch, *snaps):
    it = iter(snaps)
    monkeypatch.setattr(counters, "snapshot", lambda dev: next(it))


def run_screen(tmp_path):
    out = tmp_path / "r.jsonl"
    rc = screen.main(["--device", "cpu", "--size", "64", "--iters", "3", "--dtypes", "fp32", "--out", str(out)])
    return rc, by_name(load(out)), out


def test_every_probe_run_carries_the_counter_step(tmp_path):
    rc, steps, _ = run_screen(tmp_path)
    assert rc == 0
    assert steps[counters.STEP_NAME]["_diag"]["verdict"] == "counters-unavailable"
    assert steps[counters.STEP_NAME]["counters_available"] is False


def test_corrected_errors_are_reported_without_failing_the_run(tmp_path, monkeypatch, nvml):
    before = counters.snapshot(torch.device("cuda", 0))
    nvml.corrected += 5
    scripted(monkeypatch, before, counters.snapshot(torch.device("cuda", 0)))
    rc, steps, _ = run_screen(tmp_path)
    s = steps[counters.STEP_NAME]
    assert rc == 0 and s["_diag"]["verdict"] == "corrected-during-run"
    assert s["ecc_corrected_volatile_delta"] == 5 and s["temperature_c_after"] == 47


def test_uncorrected_errors_fail_the_run(tmp_path, monkeypatch, nvml):
    before = counters.snapshot(torch.device("cuda", 0))
    nvml.rows = [0, 1, 0, 0]
    scripted(monkeypatch, before, counters.snapshot(torch.device("cuda", 0)))
    rc, steps, _ = run_screen(tmp_path)
    assert rc == 1 and steps[counters.STEP_NAME]["_diag"]["verdict"] == "uncorrected-during-run"


def test_a_wrong_answer_with_quiet_counters_is_confirmed_silent(tmp_path, monkeypatch, nvml):
    snap = counters.snapshot(torch.device("cuda", 0))
    scripted(monkeypatch, snap, dict(snap))
    screen._FAULT_HOOK = lambda i, c: c.view(-1).__setitem__(0, c.view(-1)[0] + 1.0)
    try:
        rc, steps, _ = run_screen(tmp_path)
    finally:
        screen._FAULT_HOOK = None
    assert rc == 1 and steps[counters.STEP_NAME]["_diag"]["verdict"] == "silent-error-confirmed"


def test_site_keeps_the_counter_step_out_of_precisions_and_shows_it(tmp_path, monkeypatch, nvml):
    before = counters.snapshot(torch.device("cuda", 0))
    nvml.corrected += 1
    scripted(monkeypatch, before, counters.snapshot(torch.device("cuda", 0)))
    _, _, out = run_screen(tmp_path)
    compute, _, _, _ = build_site.summarise([build_site.parse(str(out))])
    assert compute[0]["precisions"] == ["fp32"]
    html = build_site.compute_rows(compute)
    assert "hardware corrected errors during the run" in html and 'class="ok"' in html


@pytest.mark.skipif(not os.environ.get("OCP_SCHEMA_DIR"), reason="set OCP_SCHEMA_DIR to ocp-diag-core/json_spec/output")
def test_counter_step_matches_official_ocp_schema(tmp_path, monkeypatch, nvml):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schema_dir = pathlib.Path(os.environ["OCP_SCHEMA_DIR"])
    resources = []
    for p in schema_dir.glob("*.json"):
        doc = json.loads(p.read_text())
        resources.append((doc["$id"], referencing.Resource.from_contents(doc)))
    registry = referencing.Registry().with_resources(resources)
    validator = jsonschema.Draft202012Validator(json.loads((schema_dir / "root.json").read_text()), registry=registry)
    before = counters.snapshot(torch.device("cuda", 0))
    nvml.corrected += 2
    scripted(monkeypatch, before, counters.snapshot(torch.device("cuda", 0)))
    _, _, out = run_screen(tmp_path)
    for obj in load(out):
        validator.validate(obj)
