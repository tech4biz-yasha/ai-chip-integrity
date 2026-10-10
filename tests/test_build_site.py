"""
The site may never say more than the result files do. Run from the repo root:  python -m pytest -q
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_site  # noqa: E402
import kernels  # noqa: E402


def make_kernel_file(path, version, kernel_list, inject=0):
    kernels.main(["--device", "cpu", "--size", "32", "--iters", "3", "--kernels", kernel_list,
                  "--inject", str(inject), "--out", str(path)])
    if version != kernels.VERSION:
        lines = [json.loads(line) for line in open(path)]
        for o in lines:
            start = o.get("testRunArtifact", {}).get("testRunStart")
            if start:
                start["version"] = version
        path.write_text("".join(json.dumps(o) + "\n" for o in lines))


def test_newer_version_replaces_older_row(tmp_path):
    old = tmp_path / "a_kernels_old.jsonl"
    new = tmp_path / "b_kernels_new.jsonl"
    make_kernel_file(old, "0.1.0", "softmax,gelu")
    make_kernel_file(new, kernels.VERSION, ",".join(kernels.KERNELS))
    for order in ([old, new], [new, old]):
        _, _, rows = build_site.summarise([build_site.parse(str(p)) for p in order])
        assert len(rows) == 1
        row = rows[0]
        assert row["version"] == kernels.VERSION
        assert row["kernels"] == kernels.KERNELS
        assert row["runs"] == 3 * len(kernels.KERNELS) * 3


def test_kernel_row_renders_rope_and_attention_sizes(tmp_path):
    clean = tmp_path / "k.jsonl"
    inj = tmp_path / "k_inject.jsonl"
    make_kernel_file(clean, kernels.VERSION, ",".join(kernels.KERNELS))
    make_kernel_file(inj, kernels.VERSION, ",".join(kernels.KERNELS), inject=2)
    _, _, rows = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
    html = build_site.kernel_rows(rows)
    assert "rope 32×128" in html and "attention 8×8×128" in html
    assert f'{2 * len(kernels.KERNELS) * 3} of {2 * len(kernels.KERNELS) * 3}' in html
