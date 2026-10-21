"""
Probe 5, reductions: every step must pass on a healthy device in every precision, every planted fault must be
caught and located, a wrong or differing result must fail with the right verdict, and the exact inputs must
really be exact. Small sizes on the CPU, a few tenths of a second each.
Run from the repo root:  python -m pytest -q
"""

import json
import os
import pathlib
import subprocess
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import arith, cli, counters, reduce as probe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(path):
    return [json.loads(line) for line in pathlib.Path(path).read_text().splitlines() if line.strip()]


def steps(path):
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


def run_reduce(tmp_path, *extra, dtypes="fp32", n="4096", side="64"):
    out = tmp_path / "reduce.jsonl"
    code = probe.main(["--device", "cpu", "--n", n, "--side", side, "--dtypes", dtypes, "--out", str(out), *extra])
    return code, steps(out), out


def probe_steps(st):
    return {k: v for k, v in st.items() if k.startswith("reduce_") and k != "reduce_summary"}


@pytest.mark.parametrize("dtypes", ["fp32", "fp16,bf16"])
def test_clean_run_passes_every_step(tmp_path, dtypes):
    code, st, out = run_reduce(tmp_path, dtypes=dtypes)
    assert code == 0
    ps = probe_steps(st)
    assert len(ps) == 9 * len(dtypes.split(","))
    for name, s in ps.items():
        assert s["verdict"] == "no-silent-errors" and s["type"] == "PASS", name
        assert s["m"]["repeat_runs_differing"] == 0 and s["m"]["results_wrong"] == 0
        if s["m"]["input_kind"] == "exact":
            assert s["m"]["largest_error"] == 0
        else:
            assert 0 <= s["m"]["worst_error_over_bound_ratio"] < 1
    assert st["reduce_summary"]["verdict"] == "no-silent-errors"
    assert counters.STEP_NAME in st
    assert "testRunEnd" in load(out)[-1]["testRunArtifact"]


def test_exact_inputs_are_exact_in_every_precision():
    gen = torch.Generator().manual_seed(5)
    x = probe.make_exact(4096, gen)
    assert float(x.sum()) == probe.EXACT_TAIL
    prefixes = x.cumsum(0)
    assert float(prefixes.min()) >= -1 and float(prefixes.max()) == probe.EXACT_TAIL
    for dt in (torch.float32, torch.float16, torch.bfloat16):
        assert torch.equal(x.to(dt).to(torch.float64), x)                     # inputs representable
        assert torch.equal(prefixes.to(dt).to(torch.float64), prefixes)       # every prefix representable


def test_bound_holds_for_any_summation_order():
    gen = torch.Generator().manual_seed(9)
    x = probe.make_random(1 << 12, gen).to(torch.float32)
    x64 = x.to(torch.float64)
    ref, abs_sum = x64.sum().reshape(1), x64.abs().sum().reshape(1)
    bound = probe.sum_bound(ref, abs_sum, x.numel(), 2.0 ** -24, torch.float32)
    sequential = torch.zeros((), dtype=torch.float32)
    for v in x:                                              # the worst-case order, one at a time
        sequential = sequential + v
    for s in (x.sum(), sequential, x.flip(0).sum(), x.reshape(64, 64).sum(dim=1).sum(), x.reshape(64, 64).sum(dim=0).sum()):
        assert abs(float(s) - float(ref)) <= float(bound)


@pytest.mark.parametrize("seed", ["1234", "7", "99", "2026"])
def test_self_test_catches_and_locates_every_planted_fault(tmp_path, seed):
    code, st, _ = run_reduce(tmp_path, "--inject", "2", "--seed", seed, dtypes="fp32,bf16")
    assert code == 0
    for name, s in probe_steps(st).items():
        assert s["verdict"] == "injection-self-test-pass", name
        assert s["m"]["injected_runs"] == 2 == s["m"]["injected_runs_detected"] == s["m"]["injected_runs_located"]
        assert s["m"]["repeat_runs_differing"] == 0 and s["m"]["results_wrong"] == 0     # planted runs are not evidence
    assert st["reduce_summary"]["verdict"] == "injection-self-test-pass"


def test_a_run_that_differs_and_is_wrong_is_corruption(tmp_path, monkeypatch):
    def hook(step, run, y):
        if step == "reduce_rowsum_bounded_fp32" and run == 1:
            y[3] = y[3] * 2 + 1                       # one wrong row sum on one run only
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_reduce(tmp_path)
    assert code == 1
    s = st["reduce_rowsum_bounded_fp32"]
    assert s["verdict"] == "silent-data-corruption"
    assert s["m"]["repeat_runs_differing"] == 1 and s["m"]["results_wrong"] == 1 and s["m"]["worst_result_index"] == 3
    assert st["reduce_summary"]["verdict"] == "silent-data-corruption"
    assert st["reduce_colsum_bounded_fp32"]["verdict"] == "no-silent-errors"


def test_the_same_wrong_answer_every_run_is_a_faulty_unit(tmp_path, monkeypatch):
    def hook(step, run, y):
        if step == "reduce_sum_exact_fp32":
            y[0] = y[0] + 1                            # consistently off by one, every run
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_reduce(tmp_path)
    assert code == 1
    s = st["reduce_sum_exact_fp32"]
    assert s["verdict"] == "outside-error-bound"
    assert s["m"]["repeat_runs_differing"] == 0 and s["m"]["results_wrong"] == 3 and s["m"]["largest_error"] == 1   # one per run


def test_a_run_that_differs_inside_the_bound_is_reported_not_called_a_fault(tmp_path, monkeypatch):
    def hook(step, run, y):
        if step == "reduce_colsum_bounded_fp32" and run == 2:
            y[5] = torch.nextafter(y[5], y[5] * 2)        # a last-bit wobble, inside the bound
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_reduce(tmp_path)
    assert code == 1
    s = st["reduce_colsum_bounded_fp32"]
    assert s["verdict"] == "nondeterministic-kernel"
    assert s["m"]["repeat_runs_differing"] == 1 and s["m"]["results_wrong"] == 0


def test_self_test_still_fails_on_a_real_fault(tmp_path, monkeypatch):
    def hook(step, run, y):
        if step == "reduce_cumsum_exact_fp32" and run == 1:
            y[100] = y[100] + 1
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_reduce(tmp_path, "--inject", "2")
    assert code == 1
    s = st["reduce_cumsum_exact_fp32"]
    assert s["verdict"] == "injection-self-test-fail"
    assert s["m"]["injected_runs"] == 2 == s["m"]["injected_runs_detected"] and s["m"]["results_wrong"] == 1


def test_ops_can_be_chosen(tmp_path):
    code, st, _ = run_reduce(tmp_path, "--ops", "mean,colsum")
    assert code == 0
    assert set(probe_steps(st)) == {"reduce_mean_bounded_fp32", "reduce_colsum_exact_fp32", "reduce_colsum_bounded_fp32"}


@pytest.mark.parametrize("bad", [["--runs", "1"], ["--inject", "-1"], ["--dtypes", "int8"], ["--ops", "max"],
                                 ["--n", "4095"], ["--n", "128"], ["--side", "48"], ["--n", str(probe.MAX_EXACT_N * 2)]])
def test_bad_args_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        probe.main(["--device", "cpu", "--out", str(tmp_path / "x.jsonl"), *bad])


def test_gamma_is_defined_for_the_largest_allowed_n():
    assert arith.ieee_dot(probe.MAX_EXACT_N) <= 1.0                  # 2^23 terms at u = 2^-24: nu = 1/2, gamma = 1
    with pytest.raises(ValueError):
        arith.ieee_dot(probe.MAX_EXACT_N * 2)                        # 2^24 terms would make the bound undefined


def test_runs_as_a_chip_integrity_subcommand_and_a_script(tmp_path):
    out = tmp_path / "cli.jsonl"
    assert cli.main(["reduce", "--device", "cpu", "--n", "1024", "--side", "32", "--ops", "sum", "--out", str(out)]) == 0
    assert "reduce_sum_exact_fp32" in steps(out)
    assert "reduce" not in cli.PROBE_NAMES                     # run stays the ten README commands
    done = subprocess.run([sys.executable, str(ROOT / "reduce.py"), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--side" in done.stdout


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
    _, _, out = run_reduce(tmp_path, "--inject", "2", dtypes="fp32,fp16")
    for obj in load(out):
        validator.validate(obj)
