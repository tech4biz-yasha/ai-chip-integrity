"""
The GitHub Action must do what the README says it does, and must reference only
inputs and steps that exist. These tests fail the moment either drifts.
Run from the repo root:  python -m pytest -q
"""

import pathlib
import re

import pytest

yaml = pytest.importorskip("yaml")

ROOT = pathlib.Path(__file__).resolve().parents[1]
ACTION = yaml.safe_load((ROOT / "action.yml").read_text())
ACTION_TEXT = (ROOT / "action.yml").read_text()
README = (ROOT / "README.md").read_text()
INPUTS = set(ACTION["inputs"])
# Probes `chip-integrity run` executes, so `skip` can only name one of these.
RUN_PROBES = {"screen", "kernels", "abft", "patterns", "memcheck"}


def readme_table():
    """The input table in the Run it in CI section, as {input: default}."""
    body = README[README.index("## Run it in CI"):]
    body = body[body.index("| Input | Default |"):]
    table = body[:body.index("\n\n")]
    rows = {}
    for line in table.splitlines()[2:]:
        name, default, _ = (cell.strip() for cell in line.strip("|").split("|"))
        rows[name.strip("`")] = None if default == "empty" else default.strip("`")
    return rows


def test_the_action_is_composite_and_named():
    assert ACTION["runs"]["using"] == "composite"
    assert ACTION["name"] and ACTION["description"]
    assert ACTION["author"] == "Yasha Khandelwal"


def test_every_step_declares_a_shell_or_uses_an_action():
    for step in ACTION["runs"]["steps"]:
        assert ("shell" in step) != ("uses" in step), f"{step.get('name')}: needs a shell or a uses, not both"


def test_every_referenced_input_exists():
    referenced = set(re.findall(r"inputs\.([a-z-]+)", ACTION_TEXT))
    referenced |= set(re.findall(r"inputs\['([a-z-]+)'\]", ACTION_TEXT))
    assert referenced <= INPUTS, f"action.yml references inputs that do not exist: {sorted(referenced - INPUTS)}"


def test_hyphenated_inputs_use_the_bracket_form():
    """inputs.out-dir would be read as a subtraction, so hyphenated names need brackets."""
    for name in INPUTS:
        if "-" in name:
            assert f"inputs.{name}" not in ACTION_TEXT, f"use inputs['{name}'], not inputs.{name}"


def test_every_output_points_at_a_real_step():
    ids = {step["id"] for step in ACTION["runs"]["steps"] if "id" in step}
    for output in ACTION["outputs"].values():
        step = re.search(r"steps\.([a-z-]+)\.outputs", output["value"]).group(1)
        assert step in ids, f"output points at missing step id '{step}'"


def test_the_run_step_sets_every_output_it_promises():
    run = next(step for step in ACTION["runs"]["steps"] if step.get("id") == "run")
    for name in ACTION["outputs"]:
        assert f"{name}=" in run["run"], f"the run step never writes {name} to GITHUB_OUTPUT"


def test_the_readme_table_matches_the_action():
    documented = readme_table()
    assert set(documented) == INPUTS, "the README input table and action.yml disagree"
    for name, default in documented.items():
        actual = ACTION["inputs"][name].get("default", "")
        assert (actual or None) == default, f"{name}: README says {default!r}, action.yml says {actual!r}"


def test_the_skip_input_names_only_probes_run_executes():
    documented = ACTION["inputs"]["skip"]["description"]
    named = set(re.search(r"from ([a-z,]+)", documented).group(1).split(","))
    assert named == RUN_PROBES


@pytest.mark.parametrize("snippet", re.findall(r"uses: tech4biz-yasha/ai-chip-integrity@([^\s]+)", README))
def test_the_readme_snippets_pin_a_released_tag(snippet):
    assert re.fullmatch(r"v\d+\.\d+\.\d+", snippet), "pin the Action to a v tag, not a branch"


def test_the_readme_snippets_only_pass_real_inputs():
    for block in re.findall(r"uses: tech4biz-yasha/ai-chip-integrity@[^\n]+\n(\s+)with:\n((?:\1  .+\n)+)", README):
        for line in block[1].splitlines():
            assert line.split(":")[0].strip() in INPUTS, f"snippet passes an input that does not exist: {line.strip()}"


def test_a_run_error_always_fails_the_build():
    """fail-on-fault may let a fault through; it must never let a broken run through."""
    verdict = next(step for step in ACTION["runs"]["steps"] if step["name"] == "Read the verdict")
    error_branch = verdict["run"][verdict["run"].index("*)"):]
    assert "exit 1" in error_branch
    assert "FAIL_ON_FAULT" not in error_branch
