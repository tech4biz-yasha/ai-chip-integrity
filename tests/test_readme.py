"""
The README must say what the code does. These tests fail the moment it drifts.
Run from the repo root:  python -m pytest -q
"""

import ast
import pathlib
import re

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text()
PROBES = ["screen.py", "memcheck.py", "kernels.py", "patterns.py"]
VERSIONED = PROBES + ["counters.py"]


def options_table(script):
    start = README.index(f"Options for `{script}`:")
    body = README[start:]
    return body[:body.index("\n\n", body.index("|---"))]


def argparse_options(script):
    tree = ast.parse((ROOT / script).read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "attr", "") == "add_argument":
            default = next((k.value for k in node.keywords if k.arg == "default"), None)
            yield node.args[0].value, default.value if isinstance(default, ast.Constant) else None


@pytest.mark.parametrize("script", VERSIONED)
def test_status_line_carries_the_code_version(script):
    version = re.search(r'VERSION = "([\d.]+)"', (ROOT / script).read_text()).group(1)
    assert f"`{script}` v{version}" in README


@pytest.mark.parametrize("script", PROBES)
def test_every_option_and_default_is_documented(script):
    table = options_table(script)
    rows = {re.match(r"\| `(--[\w-]+)`", line).group(1): line for line in table.splitlines() if line.startswith("| `--")}
    for flag, default in argparse_options(script):
        assert flag in rows, f"{script}: {flag} missing from its options table"
        if default is not None and not (isinstance(default, str) and "," in default):
            assert str(default) in rows[flag], f"{script}: {flag} default {default!r} not in: {rows[flag]}"


def result_files():
    """Result files the repository tracks; in a plain download, every file in results/."""
    if (ROOT / ".git").exists():
        import subprocess
        try:
            out = subprocess.run(["git", "ls-files", "results"], cwd=ROOT, capture_output=True, text=True, check=True).stdout
            return [pathlib.Path(line).name for line in out.splitlines() if line.endswith(".jsonl")]
        except (OSError, subprocess.CalledProcessError):
            pass
    return [p.name for p in (ROOT / "results").glob("*.jsonl")]


def test_every_result_file_is_in_the_readme():
    missing = [name for name in result_files() if name not in README]
    assert not missing, f"result files with no README row or note: {missing}"


def test_every_script_and_test_file_is_documented():
    for script in VERSIONED + ["arith.py", "build_site.py"]:
        assert f"| `{script}` |" in README, f"{script} missing from the Files table"
    for test_file in (ROOT / "tests").glob("test_*.py"):
        assert f"`tests/{test_file.name}`" in README, f"{test_file.name} missing from the Tests section"


def test_citation_version_matches_citation_file():
    cff = (ROOT / "CITATION.cff").read_text()
    version = re.search(r"^version: ([\d.]+)", cff, re.M).group(1)
    assert f"Version {version}, 2026." in README
