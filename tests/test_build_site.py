"""
The site may never say more than the result files do. Run from the repo root:  python -m pytest -q
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import build_site  # noqa: E402
from chip_integrity import kernels  # noqa: E402


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
        _, _, rows, _, _ = build_site.summarise([build_site.parse(str(p)) for p in order])
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
    _, _, rows, _, _ = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
    html = build_site.kernel_rows(rows)
    assert "rope 32×128" in html and "attention 8×8×128" in html
    assert f'{2 * len(kernels.KERNELS) * 3} of {2 * len(kernels.KERNELS) * 3}' in html


def test_pattern_rows_render_with_flush_policy(tmp_path, monkeypatch):
    """The subnormal cell shows what was measured. bf16 is made to flush on every host, because hosts differ:
    CPUs with AMX or AVX-512 BF16 flush bf16 subnormals in their larger kernels, other CPUs keep them."""
    from chip_integrity import patterns
    from chip_integrity import screen
    import torch
    real = screen.matmul

    def bf16_flushes(a, b, fast_accum=False):
        if a.dtype == torch.bfloat16:
            a = torch.where(a.abs() < torch.finfo(a.dtype).tiny, torch.zeros_like(a), a)
        return real(a, b, fast_accum)
    monkeypatch.setattr(screen, "matmul", bf16_flushes)
    clean, inj = tmp_path / "p.jsonl", tmp_path / "p_inject.jsonl"
    patterns.main(["--device", "cpu", "--shapes", "32x64x32", "--iters", "3", "--out", str(clean)])
    patterns.main(["--device", "cpu", "--shapes", "32x64x32", "--iters", "3", "--inject", "1", "--out", str(inj)])
    _, _, _, rows, _ = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
    assert len(rows) == 1 and rows[0]["patterns"] == build_site.PATTERN_ORDER and "bf16" in rows[0]["flushed"]
    html = build_site.pattern_rows(rows)
    steps = sum(len(v) for v in patterns.APPLIES.values())
    assert f"{steps} of {steps}" in html and "no-silent-errors" in html
    assert "<td>flushed: " + ", ".join(rows[0]["flushed"]) + "</td>" in html
    assert "<td>kept</td>" in build_site.pattern_rows([dict(rows[0], flushed=[], mixed=[])])


def test_checksum_rows_render(tmp_path):
    from chip_integrity import abft
    clean, inj = tmp_path / "a.jsonl", tmp_path / "a_inject.jsonl"
    abft.main(["--device", "cpu", "--shapes", "32x64x24", "--iters", "3", "--out", str(clean)])
    abft.main(["--device", "cpu", "--shapes", "32x64x24", "--iters", "4", "--inject", "2", "--out", str(inj)])
    *_, rows = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
    assert len(rows) == 1 and rows[0]["precisions"] == ["fp32", "fp16", "bf16"] and rows[0]["missed"] == 0
    html = build_site.checksum_rows(rows)
    assert "6 of 6" in html and "no-silent-errors" in html


def test_mixed_subnormal_policy_is_shown(tmp_path, monkeypatch):
    from chip_integrity import patterns
    from chip_integrity import screen
    import torch
    real = screen.matmul

    def top_half_flushes(a, b, fast_accum=False):
        if a.dtype == torch.float32:
            rows = (torch.arange(a.shape[0])[:, None] < a.shape[0] // 2).expand_as(a)
            a = torch.where((a.abs() < torch.finfo(a.dtype).tiny) & rows, torch.zeros_like(a), a)
        return real(a, b, fast_accum)
    monkeypatch.setattr(screen, "matmul", top_half_flushes)
    out = tmp_path / "p.jsonl"
    patterns.main(["--device", "cpu", "--shapes", "32x64x24", "--iters", "3", "--dtypes", "fp32",
                   "--patterns", "subnormal,alternate", "--out", str(out)])
    *_, rows, _ = build_site.summarise([build_site.parse(str(out))])
    assert rows[0]["mixed"] == ["fp32"] and "mixed: fp32" in build_site.pattern_rows(rows)


def test_attention_kernel_is_shown_when_chosen():
    assert build_site.attention_note({"attention": {"default": ["fp32", "fp16", "bf16"]}}) == ""
    note = build_site.attention_note({"attention": {"flash": ["bf16", "fp16"], "efficient": ["fp32"]}})
    assert note == " · attention: efficient (fp32); flash (fp16, bf16)"
