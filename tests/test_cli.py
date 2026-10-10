"""
The chip-integrity command: `run` must execute exactly the ten README commands, report every failure,
and refuse bad input; single probes and the old `python screen.py` wrappers must keep working.
Run from the repo root:  python -m pytest -q
"""

import pathlib
import re
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from chip_integrity import __version__, cli  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()


def run_args(name="chip", out_dir=None, quick=False, skip="", device="auto"):
    argv = ["run", "--name", name, "--device", device]
    if out_dir is not None:
        argv += ["--out-dir", str(out_dir)]
    if quick:
        argv.append("--quick")
    if skip:
        argv += ["--skip", skip]
    return cli.build_parser().parse_args(argv)


def test_version_names_the_suite_and_every_probe(capsys):
    with pytest.raises(SystemExit) as e:
        cli.main(["--version"])
    assert e.value.code == 0
    out = capsys.readouterr().out.strip()
    assert out.startswith(f"ai-chip-integrity {__version__} (")
    for module in cli.VERSIONED:
        version = re.search(r'^VERSION = "([\d.]+)"', (ROOT / "chip_integrity" / f"{module}.py").read_text(), re.M).group(1)
        assert f"{module} {version}" in out


def test_run_plan_is_exactly_the_ten_readme_commands(tmp_path):
    steps = cli.plan("NAME", tmp_path, "auto", quick=False, skip=set())
    assert len(steps) == 10
    files = [s[3].name for s in steps]
    assert files == ["NAME_clean.jsonl", "NAME_inject.jsonl", "NAME_kernels.jsonl", "NAME_kernels_inject.jsonl",
                     "NAME_abft.jsonl", "NAME_abft_inject.jsonl", "NAME_patterns.jsonl", "NAME_patterns_inject.jsonl",
                     "NAME_memcheck.jsonl", "NAME_memcheck_inject.jsonl"]
    for name in files:   # the same names the README asks contributors to send
        assert name in README
    flips = {s[0]: s[2][s[2].index("--inject") + 1] for s in steps if "--inject" in s[2]}
    assert flips == {"screen": "5", "kernels": "3", "abft": "3", "patterns": "3", "memcheck": "5"}
    for module in flips:   # and the same self-test strength as the README's commands
        assert re.search(rf"python {module}\.py --inject {flips[module]}\b", README)
    for module, _, argv, _ in steps:   # full runs use each probe's defaults: no size or iteration overrides
        assert not {"--size", "--shapes", "--iters", "--gb"} & set(argv)
        assert argv[:5] == [sys.executable, "-m", f"chip_integrity.{module}", "--device", "auto"]


def test_quick_mode_only_shrinks_sizes(tmp_path):
    full = cli.plan("q", tmp_path, "cpu", quick=False, skip=set())
    quick = cli.plan("q", tmp_path, "cpu", quick=True, skip=set())
    for (m, _, a, f), (m2, _, b, f2) in zip(full, quick):
        assert (m, f) == (m2, f2)
        assert b == a[:5] + cli.QUICK[m] + a[5:]   # the same command with the small sizes added, nothing else
    assert run_args(quick=True).quick


def test_skip_leaves_whole_probes_out(tmp_path):
    steps = cli.plan("s", tmp_path, "auto", quick=False, skip={"memcheck", "kernels"})
    assert {s[0] for s in steps} == {"screen", "abft", "patterns"} and len(steps) == 6


def test_every_failure_is_reported_and_fails_the_run(tmp_path, capsys):
    calls = []

    def runner(argv):
        calls.append(argv)
        return 1 if "chip_integrity.abft" in argv and "--inject" in argv else 0

    code = cli.run(run_args(out_dir=tmp_path), runner=runner)
    out = capsys.readouterr().out
    assert code == 1 and len(calls) == 10   # one failure does not stop the remaining steps
    assert "checksum matrix multiply self-test: FAIL (exit code 1)" in out
    assert "1 of 10 steps failed" in out


def test_a_clean_run_passes_and_invites_a_row(tmp_path, capsys):
    assert cli.run(run_args(out_dir=tmp_path), runner=lambda argv: 0) == 0
    out = capsys.readouterr().out
    assert "All 10 steps passed." in out and "pull request" in out
    assert cli.run(run_args(out_dir=tmp_path, quick=True), runner=lambda argv: 0) == 0
    assert "pull request" not in capsys.readouterr().out   # a quick run is never offered as a row


@pytest.mark.parametrize("kw", [dict(name="bad name"), dict(name="../x"), dict(skip="gpu"),
                                dict(skip="screen,kernels,abft,patterns,memcheck")])
def test_bad_input_is_refused_before_anything_runs(tmp_path, kw):
    called = []
    assert cli.run(run_args(out_dir=tmp_path, **kw), runner=lambda argv: called.append(argv) or 0) == 2
    assert not called


def test_one_probe_runs_with_its_own_options(tmp_path):
    out = tmp_path / "one.jsonl"
    assert cli.main(["screen", "--size", "16", "--iters", "2", "--dtypes", "fp32", "--out", str(out)]) == 0
    assert out.read_text().count("\n") > 5


def test_old_script_paths_still_run():
    for script in ["screen.py", "memcheck.py", "kernels.py", "abft.py", "patterns.py"]:
        r = subprocess.run([sys.executable, script, "--help"], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0 and "--device" in r.stdout, script


def test_a_real_quick_run_writes_its_files(tmp_path):
    r = subprocess.run([sys.executable, "-m", "chip_integrity", "run", "--quick", "--device", "cpu", "--name", "t",
                        "--skip", "kernels,patterns,memcheck", "--out-dir", str(tmp_path)],
                       cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["t_abft.jsonl", "t_abft_inject.jsonl", "t_clean.jsonl", "t_inject.jsonl"]
    assert "All 4 steps passed." in r.stdout
