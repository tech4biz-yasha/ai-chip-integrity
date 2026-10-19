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
        _, _, rows, _, _, _ = build_site.summarise([build_site.parse(str(p)) for p in order])
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
    _, _, rows, _, _, _ = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
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
    _, _, _, rows, _, _ = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
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
    *_, rows, _ = build_site.summarise([build_site.parse(str(p)) for p in (clean, inj)])
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
    *_, rows, _, _ = build_site.summarise([build_site.parse(str(out))])
    assert rows[0]["mixed"] == ["fp32"] and "mixed: fp32" in build_site.pattern_rows(rows)


def test_attention_kernel_is_shown_when_chosen():
    assert build_site.attention_note({"attention": {"default": ["fp32", "fp16", "bf16"]}}) == ""
    note = build_site.attention_note({"attention": {"flash": ["bf16", "fp16"], "efficient": ["fp32"]}})
    assert note == " · attention: efficient (fp32); flash (fp16, bf16)"


def site_from_repo():
    """The parsed result files in results/ and the chip map the pages are built from."""
    import glob, os
    runs = [build_site.parse(p) for p in sorted(glob.glob(os.path.join(build_site.RESULTS, "*.jsonl")))]
    compute, memory, kernel, pattern, checksum, soak = build_site.summarise(runs)
    return runs, build_site.chip_rows({"compute": compute, "memory": memory, "kernel": kernel, "pattern": pattern,
                                       "checksum": checksum, "soak": soak})


def test_every_chip_gets_its_own_page_with_its_files():
    runs, chip_map = site_from_repo()
    by_file = {r["file"]: r for r in runs}
    assert len({build_site.chip_slug(d) for d in chip_map}) == len(chip_map) > 0
    for device, rows in chip_map.items():
        html = build_site.chip_page(device, chip_map, by_file, "2026-01-01")
        slug = build_site.chip_slug(device)
        assert f'<link rel="canonical" href="https://chipintegrity.org/chips/{slug}.html">' in html
        assert f"<h1>{build_site.esc(device)}</h1>" in html
        for row in rows.values():
            for key in ("clean_file", "inject_file"):
                if row.get(key):
                    assert f'/blob/main/{row[key]}"' in html, (device, row[key])
        st = build_site.chip_stats(rows)
        assert f'{st["caught"]} of {st["injected"]}' in html and f'{st["runs"]:,}' in html
        assert "{{" not in html
    index = build_site.chips_index(chip_map, "2026-01-01")
    for device in chip_map:
        assert f'href="{build_site.chip_slug(device)}.html"' in index


def test_chip_totals_add_up_to_the_home_page():
    _, chip_map = site_from_repo()
    stats = [build_site.chip_stats(rows) for rows in chip_map.values()]
    html = (pathlib.Path(build_site.ROOT) / "docs" / "index.html").read_text()
    assert f'{sum(s["runs"] for s in stats):,}' in html
    assert f'{sum(s["caught"] for s in stats)} of {sum(s["injected"] for s in stats)}' in html


def test_sitemap_and_llms_list_every_chip_page():
    _, chip_map = site_from_repo()
    xml = build_site.sitemap(sorted(chip_map), "2026-01-01")
    llms = build_site.llms_measured(chip_map)
    for device in chip_map:
        url = f"https://chipintegrity.org/chips/{build_site.chip_slug(device)}.html"
        assert f"<loc>{url}</loc>" in xml and url in llms
    assert "<loc>https://chipintegrity.org/chips/index.html</loc>" in xml


def test_every_link_on_every_page_opens_from_a_plain_folder():
    """The site must work opened straight from its folder or the deploy zip, and on any static host: every link is
    relative, points at a real file (never a bare folder), and every #section it names exists in that file."""
    import os
    import re
    docs = pathlib.Path(build_site.ROOT) / "docs"
    pages = sorted(docs.glob("*.html")) + sorted((docs / "chips").glob("*.html")) + sorted((docs / "findings").glob("*.html"))
    assert len(pages) >= 17
    for page in pages:
        html = page.read_text()
        for link in re.findall(r'(?:href|src)="([^"]+)"', html):
            if link.startswith(("https://", "http://", "mailto:")):
                continue
            assert not link.startswith("/"), f"{page.name}: {link} only works on a web server"
            path, _, frag = link.partition("#")
            target = (page.parent / path).resolve() if path else page
            assert target.is_file(), f"{page.relative_to(docs)}: {link} does not open a file"
            if frag:
                assert f'id="{frag}"' in target.read_text(), f"{page.relative_to(docs)}: {link} has no #{frag}"


def test_every_page_has_a_share_image_and_the_feed_is_valid():
    """Links shared on LinkedIn, X or Slack show the page's own image, and the updates feed parses as Atom."""
    import re
    import xml.etree.ElementTree as ET
    docs = pathlib.Path(build_site.ROOT) / "docs"
    pages = sorted(docs.glob("*.html")) + sorted((docs / "chips").glob("*.html")) + sorted((docs / "findings").glob("*.html"))
    for page in pages:
        m = re.search(r'<meta property="og:image" content="https://chipintegrity\.org/(og/[^"]+\.png)">', page.read_text())
        assert m and (docs / m.group(1)).is_file(), f"{page.name}: no share image"
    root = ET.parse(docs / "feed.xml").getroot()
    entries = root.findall("{http://www.w3.org/2005/Atom}entry")
    assert len(entries) == len(build_site.FINDINGS) + len(build_site.RELEASES)


def test_glossary_terms_link_to_their_definitions():
    learn = (pathlib.Path(build_site.ROOT) / "docs" / "learn.html").read_text()
    method = (pathlib.Path(build_site.ROOT) / "docs" / "method.html").read_text()
    terms = build_site.glossary(build_site.read_site("pages/learn.html"))
    assert len(terms) >= 10
    for _, slug, _ in terms:
        assert f'id="g-{slug}"' in learn, slug
    assert 'class="term" href="learn.html#g-' in method


def test_soak_rows_render_with_conditions(tmp_path):
    """A soak file becomes one row and one probe-7 section, with its runs kept apart from the element-checked
    compute runs, and a planted-fault file fills the self-test cell."""
    from chip_integrity import soak
    clean, inj = tmp_path / "s.jsonl", tmp_path / "s_inject.jsonl"
    for path, inject in ((clean, "0"), (inj, "2")):
        assert soak.main(["--device", "cpu", "--shape", "48x40x32", "--minutes", "0.01", "--window-s", "0.2",
                          "--dtypes", "fp32", "--inject", inject, "--out", str(path)]) == 0
    runs = [build_site.parse(str(p)) for p in (clean, inj)]
    *_, rows = build_site.summarise(runs)
    assert len(rows) == 1
    row = rows[0]
    assert row["runs"] > 0 and row["windows"] >= 1 and row["faults"] == 0 and row["precisions"] == ["fp32"]
    assert row["injected"] > 0 and row["caught"] == row["injected"]
    html = build_site.soak_rows(rows)
    assert f'{row["runs"]:,}' in html and f'{row["caught"]} of {row["injected"]}' in html
    assert "no telemetry on this device" in html and "no-silent-errors" in html
    chip_map = build_site.chip_rows({"soak": rows})
    st = build_site.chip_stats(chip_map[row["device"]])
    assert st["runs"] == 0 and st["soak_runs"] == row["runs"]
    page = build_site.chip_page(row["device"], chip_map, {r["file"]: r for r in runs}, "2026-01-01")
    assert "PROBE 7" in page and f'{row["runs"]:,} runs in {row["windows"]} windows' in page


def test_driver_and_cuda_versions_are_read_from_the_file(tmp_path):
    """Files written since v0.4.3 carry the CUDA runtime and driver; the chip page lists them."""
    from chip_integrity import screen
    out = tmp_path / "c.jsonl"
    screen.main(["--device", "cpu", "--size", "16", "--iters", "2", "--dtypes", "fp32", "--out", str(out)])
    lines = [json.loads(line) for line in open(out)]
    start = next(o["testRunArtifact"]["testRunStart"] for o in lines
                 if "testRunArtifact" in o and "testRunStart" in o["testRunArtifact"])
    start["dutInfo"]["softwareInfos"] += [{"softwareInfoId": "x_2", "name": "cuda", "version": "12.8"},
                                          {"softwareInfoId": "x_3", "name": "nvidia-driver", "version": "580.126.09"}]
    out.write_text("".join(json.dumps(o) + "\n" for o in lines))
    r = build_site.parse(str(out))
    assert r["driver"] == "580.126.09" and r["cuda"] == "12.8"
    compute, *_ = build_site.summarise([r])
    chip_map = build_site.chip_rows({"compute": compute})
    page = build_site.chip_page(r["device"], chip_map, {r["file"]: r}, "2026-01-01")
    assert "NVIDIA driver" in page and "580.126.09" in page and "CUDA runtime" in page and "12.8" in page
