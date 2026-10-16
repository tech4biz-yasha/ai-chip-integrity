"""
Probe 7, hours under load: every window must pass on a healthy device, every planted fault must be caught,
a real fault must fail its window with the chip's state recorded, and a precision whose first result or
repeats cannot be trusted must be left out of the soak. Short soaks on the CPU, a few tenths of a second each.
Run from the repo root:  python -m pytest -q
"""

import json
import os
import pathlib
import subprocess
import sys
import types

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import cli, counters, screen, soak  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
SHAPE = "48x40x32"


def load(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def steps(path):
    """Every step in file order: {name: {"m": measurements, "verdict", "type", "status", "errors"}}."""
    out, names = {}, {}
    for obj in load(path):
        a = obj.get("testStepArtifact")
        if not a:
            continue
        sid = a["testStepId"]
        if "testStepStart" in a:
            names[sid] = a["testStepStart"]["name"]
            out[names[sid]] = {"m": {}, "verdict": None, "type": None, "status": None, "errors": []}
        elif "measurement" in a:
            out[names[sid]]["m"][a["measurement"]["name"]] = a["measurement"]["value"]
        elif "diagnosis" in a:
            out[names[sid]]["verdict"] = a["diagnosis"]["verdict"]
            out[names[sid]]["type"] = a["diagnosis"]["type"]
        elif "error" in a:
            out[names[sid]]["errors"].append(a["error"])
        elif "testStepEnd" in a:
            out[names[sid]]["status"] = a["testStepEnd"]["status"]
    return out


def windows(st):
    return {k: v for k, v in st.items() if k.startswith("soak_window_")}


def run_soak(tmp_path, *extra, shape=SHAPE, minutes="0.01", window="0.2", dtypes="fp32"):
    out = tmp_path / "soak.jsonl"
    code = soak.main(["--device", "cpu", "--shape", shape, "--minutes", minutes, "--window-s", window,
                      "--dtypes", dtypes, "--out", str(out), *extra])
    return code, steps(out), out


class FakeTelemetry:
    """A chip that warms up by one degree per reading and reports a power cap and thermal slowdown."""

    def __init__(self, dev):
        self.t = 60

    available = True

    def read(self):
        self.t += 1
        return {"temperature_c": self.t, "sm_clock_mhz": 1980 - self.t, "memory_clock_mhz": 2619,
                "power_w": 300.5, "clock_event_reasons": 0x4 | 0x20}


def test_clean_soak_passes_every_window(tmp_path):
    code, st, _ = run_soak(tmp_path)
    assert code == 0
    assert st["soak_setup_fp32"]["verdict"] == "first-result-verified"
    assert st["soak_setup_fp32"]["m"]["first_result_outside_bound"] == 0
    assert st["soak_setup_fp32"]["m"]["repeat_runs_differing"] == 0
    win = windows(st)
    assert len(win) >= 2
    for w in win.values():
        assert w["verdict"] == "no-silent-errors" and w["type"] == "PASS"
        assert w["m"]["runs"] > 0 and w["m"]["differing_runs"] == 0
        assert w["m"]["telemetry_available"] is False          # no NVML on a CPU
    summary = st["soak_summary"]
    assert summary["verdict"] == "no-silent-errors"
    assert summary["m"]["runs"] == sum(w["m"]["runs"] for w in win.values())
    assert summary["m"]["windows"] == len(win) and summary["m"]["interrupted"] is False
    assert st[counters.STEP_NAME]["verdict"] == "counters-unavailable"


@pytest.mark.parametrize("model,planted", [("bit", 0), ("gpu-sdc", 3), ("mixed", 1)])
def test_self_test_catches_every_planted_fault(tmp_path, model, planted):
    code, st, _ = run_soak(tmp_path, "--inject", "3", "--inject-model", model)
    assert code == 0
    for w in windows(st).values():
        assert w["verdict"] == "injection-self-test-pass"
        assert w["m"]["injected_runs"] == 3 == w["m"]["injected_runs_detected"]
        assert w["m"]["gpu_sdc_faults_planted"] == planted == w["m"]["gpu_sdc_faults_detected"]
        assert w["m"]["differing_runs"] == 0
    assert st["soak_summary"]["verdict"] == "injection-self-test-pass"


def test_self_test_catches_faults_in_exact_int8(tmp_path):
    code, st, _ = run_soak(tmp_path, "--inject", "2", dtypes="int8")
    assert code == 0
    assert st["soak_setup_int8"]["m"]["exact_check"] is True
    for w in windows(st).values():
        assert w["m"]["injected_runs"] == 2 == w["m"]["injected_runs_detected"]


def test_intermittent_fault_fails_only_its_window(tmp_path, monkeypatch):
    def hook(window, run, c):
        if window == 1 and run == 5:
            c.view(torch.int32).view(-1)[7] ^= 1 << 20          # one mantissa bit, far outside the bound
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, _ = run_soak(tmp_path)
    assert code == 1
    win = list(windows(st).values())
    assert win[0]["verdict"] == "no-silent-errors"
    assert win[1]["verdict"] == "silent-data-corruption" and win[1]["type"] == "FAIL"
    assert win[1]["m"]["differing_runs"] == 1
    assert win[1]["m"]["differing_runs_checked_against_bound"] == 1 == win[1]["m"]["differing_runs_outside_bound"]
    assert all(w["verdict"] == "no-silent-errors" for w in win[2:])
    assert st["soak_summary"]["verdict"] == "silent-data-corruption"
    assert st["soak_summary"]["m"]["differing_runs"] == 1


def test_any_difference_in_exact_int8_is_wrong(tmp_path, monkeypatch):
    def hook(window, run, c):
        if window == 0 and run == 3:
            c.view(-1)[0] += 1
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, _ = run_soak(tmp_path, dtypes="int8")
    assert code == 1
    first = list(windows(st).values())[0]
    assert first["verdict"] == "silent-data-corruption"
    assert first["m"]["differing_runs_outside_bound"] == 1


def test_a_kernel_that_wobbles_inside_the_bound_is_not_called_corruption(tmp_path, monkeypatch):
    def hook(window, run, c):
        if window == 0:
            flat = c.view(-1)
            i = int(flat.abs().argmax())                       # a large element, so one ulp stays in bound
            flat.view(torch.int32)[i] ^= 1                      # lowest mantissa bit, every run
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, _ = run_soak(tmp_path)
    assert code == 1
    first = list(windows(st).values())[0]
    assert first["verdict"] == "nondeterministic-kernel"
    assert first["m"]["differing_runs"] == first["m"]["runs"]
    assert first["m"]["differing_runs_checked_against_bound"] == soak.MAX_BOUND_CHECKS
    assert first["m"]["differing_runs_outside_bound"] == 0


def test_first_result_outside_the_bound_is_left_out(tmp_path, monkeypatch):
    real = screen.matmul
    monkeypatch.setattr(screen, "matmul", lambda a, b, fast=False: real(a, b, fast) + 0.5)   # a faulty unit
    code, st, _ = run_soak(tmp_path)
    assert code == 1
    assert st["soak_setup_fp32"]["verdict"] == "outside-error-bound"
    assert st["soak_setup_fp32"]["m"]["first_result_outside_bound"] == 1
    assert not windows(st)
    assert st["soak_summary"]["verdict"] == "not-run"


def test_a_kernel_that_does_not_repeat_is_left_out_at_setup(tmp_path, monkeypatch):
    real, calls = screen.matmul, {"n": 0}

    def wobbly(a, b, fast=False):
        c = real(a, b, fast)
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            flat = c.view(-1)
            flat.view(torch.int32)[int(flat.abs().argmax())] ^= 1
        return c
    monkeypatch.setattr(screen, "matmul", wobbly)
    code, st, _ = run_soak(tmp_path)
    assert code == 1
    assert st["soak_setup_fp32"]["verdict"] == "nondeterministic-kernel"     # one fp32 ulp stays in the bound
    assert st["soak_setup_fp32"]["m"]["repeat_runs_differing"] >= 1
    assert not windows(st)


def test_precisions_rotate_one_per_window(tmp_path):
    code, st, _ = run_soak(tmp_path, dtypes="fp32,bf16", minutes="0.012", window="0.15")
    assert code == 0
    names = [w["m"]["precision"] for w in windows(st).values()]
    assert len(names) >= 3
    assert names == [("fp32", "bf16")[i % 2] for i in range(len(names))]


def test_unsupported_precision_is_skipped_and_the_rest_soaked(tmp_path):
    code, st, _ = run_soak(tmp_path, shape="32x8192x32", dtypes="fp8,fp32")   # FP8 needs K <= 4096
    assert code == 0
    assert st["soak_setup_fp8"]["status"] == "SKIP" and st["soak_setup_fp8"]["errors"]
    assert {w["m"]["precision"] for w in windows(st).values()} == {"fp32"}


def test_windows_carry_the_chips_state(tmp_path, monkeypatch):
    monkeypatch.setattr(soak, "Telemetry", FakeTelemetry)

    def hook(window, run, c):
        if window == 1 and run == 2:
            c.view(-1)[0] = float("nan")
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, _ = run_soak(tmp_path, "--sample-s", "0.05")
    assert code == 1
    win = list(windows(st).values())
    for w in win:
        m = w["m"]
        assert m["telemetry_available"] is True
        assert m["temperature_c_min"] <= m["temperature_c_max"]
        assert m["sm_clock_mhz_min"] <= m["sm_clock_mhz_max"]
        assert m["power_w_max"] == 300.5 and m["memory_clock_mhz_max"] == 2619
        assert m["clock_event_reasons"] == "sw-power-cap,sw-thermal-slowdown"
    assert "first_differing_run_temperature_c" in win[1]["m"]
    assert "first_differing_run_temperature_c" not in win[0]["m"]
    summary = st["soak_summary"]["m"]
    assert summary["temperature_c_max"] == max(w["m"]["temperature_c_max"] for w in win)
    assert summary["clock_event_reasons"] == "sw-power-cap,sw-thermal-slowdown"


class FakeNVML(types.ModuleType):
    NVML_TEMPERATURE_GPU, NVML_CLOCK_SM, NVML_CLOCK_MEM = 0, 1, 2

    def __init__(self):
        super().__init__("pynvml")

    def nvmlDeviceGetTemperature(self, h, sensor):
        return 71

    def nvmlDeviceGetClockInfo(self, h, clock):
        return {1: 1755, 2: 2619}[clock]

    def nvmlDeviceGetPowerUsage(self, h):
        return 412345

    def nvmlDeviceGetCurrentClocksEventReasons(self, h):
        return 0x4 | 0x40


def telemetry_with(monkeypatch, nv):
    monkeypatch.setattr(counters, "_nvml", lambda: nv)
    monkeypatch.setattr(counters, "_handle", lambda nv_, dev: "gpu0")
    return soak.Telemetry(types.SimpleNamespace(type="cuda"))


def test_telemetry_reads_nvml(monkeypatch):
    t = telemetry_with(monkeypatch, FakeNVML())
    assert t.available
    assert t.read() == {"temperature_c": 71, "sm_clock_mhz": 1755, "memory_clock_mhz": 2619, "power_w": 412.3,
                        "clock_event_reasons": 0x44}


def test_telemetry_falls_back_and_leaves_out_what_is_missing(monkeypatch):
    nv = FakeNVML()
    del FakeNVML.NVML_CLOCK_MEM                     # an older binding without the memory clock constant
    try:
        nv.nvmlDeviceGetCurrentClocksEventReasons = None
        nv.nvmlDeviceGetCurrentClocksThrottleReasons = lambda h: 0x8   # the pre-rename call
        nv.nvmlDeviceGetPowerUsage = lambda h: 1 / 0                   # a call that fails
        got = telemetry_with(monkeypatch, nv).read()
    finally:
        FakeNVML.NVML_CLOCK_MEM = 2
    assert got == {"temperature_c": 71, "sm_clock_mhz": 1755, "clock_event_reasons": 0x8}


def test_no_telemetry_off_nvidia():
    assert not soak.Telemetry(torch.device("cpu")).available
    assert soak.Telemetry(torch.device("cpu")).read() == {}


def test_clock_event_reasons_are_named():
    assert soak.reason_names(0) == []
    assert soak.reason_names(0x4 | 0x40) == ["sw-power-cap", "hw-thermal-slowdown"]
    assert soak.reason_names(0x1 | 0x1000) == ["gpu-idle", "other-0x1000"]


def test_ctrl_c_keeps_what_was_measured(tmp_path, monkeypatch):
    def hook(window, run, c):
        if window == 1 and run == 10:
            raise KeyboardInterrupt
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, out = run_soak(tmp_path, minutes="0.05")
    assert code == 0
    win = list(windows(st).values())
    assert len(win) == 2 and win[1]["m"]["runs"] == 10
    assert st["soak_summary"]["m"]["interrupted"] is True
    assert counters.STEP_NAME in st
    assert "testRunEnd" in load(out)[-1]["testRunArtifact"]


def test_self_test_still_fails_on_a_real_fault(tmp_path, monkeypatch):
    def hook(window, run, c):
        if window == 0 and run == 10:                          # never one of the planted runs (0 to 3)
            c.view(torch.int32).view(-1)[3] ^= 1 << 25
    monkeypatch.setattr(soak, "_FAULT_HOOK", hook)
    code, st, _ = run_soak(tmp_path, "--inject", "2")
    assert code == 1
    first = list(windows(st).values())[0]
    assert first["verdict"] == "injection-self-test-fail"
    assert first["m"]["injected_runs"] == 2 == first["m"]["injected_runs_detected"]
    assert first["m"]["differing_runs"] == 1


@pytest.mark.parametrize("bad", [["--minutes", "0"], ["--window-s", "0.01"], ["--shape", "8x8x8,16x16x16"],
                                 ["--dtypes", "fp33"], ["--inject", "-1"], ["--shape", "8x8"]])
def test_bad_args_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        soak.main(["--device", "cpu", "--out", str(tmp_path / "x.jsonl"), *bad])


def test_runs_as_a_chip_integrity_subcommand_and_a_script(tmp_path):
    out = tmp_path / "cli.jsonl"
    assert cli.main(["soak", "--device", "cpu", "--shape", SHAPE, "--minutes", "0.005", "--window-s", "0.2",
                     "--out", str(out)]) == 0
    assert any(name.startswith("soak_window_") for name in steps(out))
    assert "soak" not in cli.PROBE_NAMES                       # run stays the ten README commands
    done = subprocess.run([sys.executable, str(ROOT / "soak.py"), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--minutes" in done.stdout


@pytest.mark.skipif(not os.environ.get("OCP_SCHEMA_DIR"), reason="set OCP_SCHEMA_DIR to ocp-diag-core/json_spec/output")
def test_output_matches_official_ocp_schema(tmp_path, monkeypatch):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schema_dir = pathlib.Path(os.environ["OCP_SCHEMA_DIR"])
    resources = []
    for p in schema_dir.glob("*.json"):
        doc = json.loads(p.read_text())
        resources.append((doc["$id"], referencing.Resource.from_contents(doc)))
    registry = referencing.Registry().with_resources(resources)
    validator = jsonschema.Draft202012Validator(json.loads((schema_dir / "root.json").read_text()), registry=registry)
    monkeypatch.setattr(soak, "Telemetry", FakeTelemetry)
    _, _, out = run_soak(tmp_path, "--inject", "2", "--sample-s", "0.05", shape="32x8192x32", dtypes="fp8,fp32")
    for obj in load(out):
        validator.validate(obj)
