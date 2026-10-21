"""
Probe 4, full model forward pass: a clean run must pass on a healthy device with every multiply inside its
bound, every planted fault must be caught by at least one of the three checks, a changed weight or a differing
run must fail, and the file must carry the model's pins. All on the two-layer tiny model on the CPU.
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
pytest.importorskip("transformers")
from chip_integrity import cli, counters, model as probe  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]


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


def run_model(tmp_path, *extra, dtypes="fp32", prompts="3", runs="3"):
    out = tmp_path / "model.jsonl"
    code = probe.main(["--device", "cpu", "--model", "tiny", "--dtypes", dtypes, "--max-prompts", prompts,
                       "--max-tokens", "12", "--new-tokens", "3", "--runs", runs, "--out", str(out), *extra])
    return code, steps(out), out


def test_clean_run_passes_and_pins_the_model(tmp_path):
    code, st, out = run_model(tmp_path)
    assert code == 0
    load_ = st["model_load"]
    assert load_["verdict"] == "model-pinned" and len(load_["m"]["weights_sha256"]) == 64
    assert load_["m"]["layers"] == 2 and load_["m"]["prompts"] == 3 and load_["m"]["tokenizer_vocab_sha256"] == "none"
    assert st["model_reference_float64"]["verdict"] == "reference-computed"
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "no-silent-errors" and fwd["type"] == "PASS"
    m = fwd["m"]
    assert m["repeat_runs_differing"] == 0 and m["linear_calls_outside_bound"] == 0 and m["weight_elements_changed"] == 0
    assert m["linear_calls"] == 3 * 3 * 15 * 4            # prompts x runs x Linear modules x (prefill + 3 tokens)
    assert 0 < m["worst_error_over_bound_ratio"] < 1
    assert m["positions_compared_with_float64"] == st["model_load"]["m"]["prompt_tokens"]
    assert m["positions_disagreeing_with_float64"] == 0 and m["first_divergence_token_min"] == -1
    assert m["max_logit_abs_diff_at_decision"] > 0        # fp32 differs from float64, but not by a token
    assert {k for k in m if k.endswith("_worst_error_over_bound_ratio")} >= {
        "layer_00_worst_error_over_bound_ratio", "layer_01_worst_error_over_bound_ratio", "head_worst_error_over_bound_ratio"}
    assert st["model_summary"]["verdict"] == "no-silent-errors"
    assert counters.STEP_NAME in st
    assert "testRunEnd" in load(out)[-1]["testRunArtifact"]


def test_weights_are_hashed_the_same_way_every_time(tmp_path):
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    _, a, _ = run_model(tmp_path / "a")
    _, b, _ = run_model(tmp_path / "b")
    assert a["model_load"]["m"]["weights_sha256"] == b["model_load"]["m"]["weights_sha256"]


@pytest.mark.parametrize("planted,seed", [("2", "1234"), ("4", "1234"), ("2", "7"), ("2", "99"), ("4", "2026")])
def test_self_test_catches_every_planted_fault(tmp_path, planted, seed):
    code, st, _ = run_model(tmp_path, "--inject", planted, "--seed", seed, prompts="16")
    assert code == 0
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "injection-self-test-pass"
    assert fwd["m"]["injected_runs"] == int(planted) == fwd["m"]["injected_runs_detected"] == fwd["m"]["injected_runs_localised"]
    assert fwd["m"]["injected_kinds"].split(",") == ["activation", "weight"] * (int(planted) // 2)
    assert fwd["m"]["repeat_runs_differing"] == 0             # planted runs never count as clean evidence
    assert fwd["m"]["weight_elements_changed"] == 0           # the flipped weight was restored
    assert st["model_summary"]["verdict"] == "injection-self-test-pass"


def test_a_weight_changed_on_the_chip_fails_the_run(tmp_path, monkeypatch):
    def hook(name, run, model):
        if run == 1:
            w = dict(model.named_parameters())["model.layers.1.mlp.down_proj.weight"]
            w.detach().view(torch.int32).view(-1)[7] ^= 1 << 30
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_model(tmp_path)
    assert code == 1
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "silent-data-corruption"
    assert fwd["m"]["weight_elements_changed"] == 2             # one element, seen after each of runs 1 and 2
    assert st["model_summary"]["verdict"] == "silent-data-corruption"


def test_a_run_that_differs_with_a_multiply_outside_its_bound_is_corruption(tmp_path, monkeypatch):
    handles = []

    def hook(name, run, model):
        if run == 2:                                           # corrupt one multiply during one clean run only
            lin = dict(model.named_modules())["model.layers.0.self_attn.q_proj"]
            handles.append(lin.register_forward_hook(lambda m, i, o: o * 3.0, prepend=True))
        elif handles:
            handles.pop().remove()
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_model(tmp_path, runs="4")
    assert code == 1
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "silent-data-corruption"
    assert fwd["m"]["repeat_runs_differing"] == 1 and fwd["m"]["linear_calls_outside_bound"] > 0
    assert fwd["m"]["layer_00_calls_outside_bound"] > 0 and fwd["m"]["layer_01_calls_outside_bound"] == 0


def test_runs_that_agree_but_sit_outside_the_bound_are_a_faulty_unit(tmp_path, monkeypatch):
    def hook(name, run, model):
        if run == 0:                                           # the same wrong multiply on every run
            lin = dict(model.named_modules())["model.layers.1.mlp.up_proj"]
            lin.register_forward_hook(lambda m, i, o: o * 1.5, prepend=True)
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_model(tmp_path)
    assert code == 1
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "outside-error-bound"
    assert fwd["m"]["repeat_runs_differing"] == 0 and fwd["m"]["linear_calls_outside_bound"] > 0
    assert fwd["m"]["layer_01_calls_outside_bound"] > 0 and fwd["m"]["layer_00_calls_outside_bound"] == 0
    assert fwd["m"]["worst_error_over_bound_ratio"] > 1


def test_a_run_that_differs_inside_the_bound_is_reported_not_called_a_fault(tmp_path, monkeypatch):
    handles = []

    def hook(name, run, model):
        if run == 1:                                           # a last-bit wobble in one multiply, one run
            lin = dict(model.named_modules())["model.layers.0.mlp.gate_proj"]
            handles.append(lin.register_forward_hook(lambda m, i, o: torch.nextafter(o, o * 2), prepend=True))
        elif handles:
            handles.pop().remove()
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_model(tmp_path)
    assert code == 1
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "nondeterministic-kernel"
    assert fwd["m"]["repeat_runs_differing"] == 1 and fwd["m"]["linear_calls_outside_bound"] == 0


def test_self_test_still_fails_on_a_real_fault(tmp_path, monkeypatch):
    def hook(name, run, model):
        if run == 1:
            w = dict(model.named_parameters())["model.layers.0.self_attn.v_proj.weight"]
            w.detach().view(torch.int32).view(-1)[0] ^= 1 << 29
    monkeypatch.setattr(probe, "_FAULT_HOOK", hook)
    code, st, _ = run_model(tmp_path, "--inject", "2")
    assert code == 1
    fwd = st["model_forward_fp32"]
    assert fwd["verdict"] == "injection-self-test-fail"
    assert fwd["m"]["injected_runs"] == 2 == fwd["m"]["injected_runs_detected"]
    assert fwd["m"]["weight_elements_changed"] >= 1


def test_prompt_file_and_second_precision(tmp_path):
    pf = tmp_path / "p.txt"
    pf.write_text("one\n\ntwo\nthree\nfour\n")
    code, st, _ = run_model(tmp_path, "--prompts", str(pf), dtypes="fp32,bf16", prompts="4")
    assert code == 0
    assert st["model_load"]["m"]["prompts"] == 4                # blank lines are not prompts
    assert st["model_forward_bf16"]["verdict"] == "no-silent-errors"
    assert st["model_forward_bf16"]["m"]["worst_error_over_bound_ratio"] < 1
    assert st["model_summary"]["m"]["precisions"] == "fp32,bf16"


@pytest.mark.parametrize("bad", [["--runs", "1"], ["--inject", "-1"], ["--dtypes", "int8"], ["--max-prompts", "0"],
                                 ["--new-tokens", "-1"], ["--prompts", "/nonexistent/prompts.txt"]])
def test_bad_args_rejected(tmp_path, bad):
    with pytest.raises(SystemExit):
        probe.main(["--device", "cpu", "--model", "tiny", "--out", str(tmp_path / "x.jsonl"), *bad])


def test_runs_as_a_chip_integrity_subcommand_and_a_script(tmp_path):
    out = tmp_path / "cli.jsonl"
    assert cli.main(["model", "--device", "cpu", "--model", "tiny", "--max-prompts", "2", "--max-tokens", "8",
                     "--new-tokens", "2", "--out", str(out)]) == 0
    assert "model_forward_fp32" in steps(out)
    assert "model" not in cli.PROBE_NAMES                      # run stays the ten README commands
    done = subprocess.run([sys.executable, str(ROOT / "model.py"), "--help"], capture_output=True, text=True)
    assert done.returncode == 0 and "--new-tokens" in done.stdout


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
    _, _, out = run_model(tmp_path, "--inject", "2", dtypes="fp32,bf16")
    for obj in load(out):
        validator.validate(obj)
