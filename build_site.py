#!/usr/bin/env python3
"""
build_site.py  -  generates docs/index.html for chipintegrity.org from results/*.jsonl

Every number on the page is read from the OCP result files in results/. Run it
after adding result files, then commit docs/.

    python build_site.py
"""

import glob
import json
import math
import os
import re
import shutil
from collections import defaultdict
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "docs", "index.html")
REPO = "https://github.com/tech4biz-yasha/ai-chip-integrity"


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def parse(path):
    objs = load(path)
    start = next(o["testRunArtifact"]["testRunStart"] for o in objs
                 if "testRunArtifact" in o and "testRunStart" in o["testRunArtifact"])
    end = next(o["testRunArtifact"]["testRunEnd"] for o in objs
               if "testRunArtifact" in o and "testRunEnd" in o["testRunArtifact"])
    ts = next(o["timestamp"] for o in objs if "timestamp" in o)
    steps = defaultdict(dict)
    names = {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        sid = sa["testStepId"]
        if "testStepStart" in sa:
            names[sid] = sa["testStepStart"]["name"]
        if "measurement" in sa:
            steps[sid][sa["measurement"]["name"]] = sa["measurement"]["value"]
        if "diagnosis" in sa:
            steps[sid]["_verdict"] = sa["diagnosis"]["verdict"]
        if "error" in sa:
            steps[sid]["_skipped"] = True
    named = {names.get(k, k): v for k, v in steps.items()}
    counter_step = named.pop(COUNTER_STEP, None)          # probe 11's cross-check, not a precision or kernel
    hw = start["dutInfo"]["hardwareInfos"][0]["name"]
    sw = {s["name"]: s["version"] for s in start["dutInfo"].get("softwareInfos", [])}
    return {
        "file": os.path.relpath(path, ROOT), "run": start["name"], "version": start["version"],
        "params": start.get("parameters", {}), "result": end["result"], "date": ts[:10],
        "device": hw, "torch": sw.get("torch", ""), "steps": named, "counters": counter_step,
    }


COUNTER_STEP = "device_error_counters"
COUNTER_NOTES = {
    "counters-quiet": "error counters quiet",
    "corrected-during-run": "hardware corrected errors during the run",
    "uncorrected-during-run": "uncorrected hardware errors during the run",
    "silent-error-confirmed": "error counters stayed quiet while answers were wrong",
    "hardware-noticed": "error counters moved with the wrong answers",
    "counters-unavailable": "no error counters",
    "ecc-off": "ECC off, so the error counters cannot see memory errors",
}


def counter_verdict(step):
    """Probe 11's verdict for one run. Files from counters 0.1.0 said counters-quiet even with ECC off, when the
    counters cannot move at all; they are shown as ecc-off, the verdict counters 0.1.1 records for that case."""
    if not step:
        return None
    if step.get("_verdict") == "counters-quiet" and step.get("ecc_enabled") is False:
        return "ecc-off"
    return step.get("_verdict")
KERNEL_ORDER = ["softmax", "layernorm", "gelu", "rope", "attention"]
PRECISION_ORDER = ["fp32", "fp16", "bf16", "int8", "fp8", "fp8fast"]
PATTERN_ORDER = ["wide", "mantissa", "cancel", "alternate", "sparse", "subnormal", "near_max", "extremes"]


def vkey(version):
    return tuple(int(p) for p in re.findall(r"\d+", str(version))[:3])


def row_for(table, r, **base):
    """One row per device and probe. A newer tool version replaces the row; an older one is left out of the
    table (its files stay in results/ and in the git history)."""
    key = r["device"]
    row = table.get(key)
    if row is None or vkey(r["version"]) > vkey(row["version"]):
        row = table[key] = dict(device=key, dates=set(), version=r["version"], **base)
    elif vkey(r["version"]) < vkey(row["version"]):
        return None
    row["dates"].add(r["date"])
    return row


def summarise(runs):
    """Group runs by device and probe into table rows."""
    compute, memory, kernel, pattern, checksum = {}, {}, {}, {}, {}
    for r in runs:
        inject = int(r["params"].get("inject", 0)) > 0
        if r["run"] == "ai-chip-integrity-screen":
            row = row_for(compute, r, torch=r["torch"])
            if row is None:
                continue
            if inject:
                row["injected"] = sum(s.get("injected_runs", 0) for s in r["steps"].values())
                row["caught"] = sum(s.get("injected_runs_detected", 0) for s in r["steps"].values())
                row["inject_file"] = r["file"]
            else:
                live = [s for s in r["steps"].values() if "_skipped" not in s]
                row["runs"] = sum(s.get("runs", 0) for s in live)
                row["steps"] = len(live)
                row["skipped"] = sum(1 for s in r["steps"].values() if "_skipped" in s)
                row["precisions"] = sorted({n.split("_")[1] for n, s in r["steps"].items() if "_skipped" not in s},
                                           key=lambda p: (PRECISION_ORDER.index(p) if p in PRECISION_ORDER else 99, p))
                row["shapes"] = r["params"].get("shapes", r["params"].get("size", ""))
                row["ref_fail"] = sum(s.get("reference_check_failed_runs", 0) for s in live)
                row["rep_fail"] = sum(s.get("repeat_check_failed_runs", 0) for s in live)
                row["verdicts"] = sorted({s["_verdict"] for s in live})
                row["clean_file"] = r["file"]
                row["counters"] = counter_verdict(r.get("counters"))
        elif r["run"] == "ai-chip-integrity-kernels":
            row = row_for(kernel, r, torch=r["torch"])
            if row is None:
                continue
            if inject:
                row["injected"] = sum(s.get("injected_runs", 0) for s in r["steps"].values())
                row["caught"] = sum(s.get("injected_runs_detected", 0) for s in r["steps"].values())
                row["inject_file"] = r["file"]
            else:
                live = [s for s in r["steps"].values() if "_skipped" not in s]
                row["runs"] = sum(s.get("runs", 0) for s in live)
                row["skipped"] = sum(1 for s in r["steps"].values() if "_skipped" in s)
                row["kernels"] = sorted({n.split("_")[0] for n, s in r["steps"].items() if "_skipped" not in s},
                                        key=lambda n: (KERNEL_ORDER.index(n) if n in KERNEL_ORDER else 99, n))
                row["precisions"] = sorted({n.split("_")[1] for n, s in r["steps"].items() if "_skipped" not in s},
                                           key=["fp32", "fp16", "bf16"].index)
                row["size"] = r["params"].get("size", "")
                row["attention"] = {}
                for n, s in r["steps"].items():
                    if n.startswith("attention_") and "attention_backend" in s:
                        row["attention"].setdefault(s["attention_backend"], []).append(n.split("_")[1])
                row["ratio"] = max(s.get("worst_error_over_bound_ratio", 0) for s in live)
                row["ref_fail"] = sum(s.get("reference_check_failed_runs", 0) for s in live)
                row["rep_fail"] = sum(s.get("repeat_check_failed_runs", 0) for s in live)
                row["verdicts"] = sorted({s["_verdict"] for s in live})
                row["clean_file"] = r["file"]
                row["counters"] = counter_verdict(r.get("counters"))
        elif r["run"] == "ai-chip-integrity-patterns":
            row = row_for(pattern, r, torch=r["torch"])
            if row is None:
                continue
            if inject:
                row["injected"] = sum(s.get("injected_runs", 0) for s in r["steps"].values())
                row["caught"] = sum(s.get("injected_runs_detected", 0) for s in r["steps"].values())
                row["inject_file"] = r["file"]
            else:
                live = {n: s for n, s in r["steps"].items() if "_skipped" not in s}
                order = lambda lst: (lambda x: (lst.index(x) if x in lst else 99, x))
                row["runs"] = sum(s.get("runs", 0) for s in live.values())
                row["skipped"] = len(r["steps"]) - len(live)
                row["patterns"] = sorted({s.get("pattern") or "_".join(n.split("_")[2:-1]) for n, s in live.items()},
                                         key=order(PATTERN_ORDER))
                row["precisions"] = sorted({n.split("_")[1] for n in live}, key=order(PRECISION_ORDER))
                row["flushed"] = sorted({n.split("_")[1] for n, s in live.items() if s.get("subnormal_inputs_flushed") is True},
                                        key=order(PRECISION_ORDER))
                row["mixed"] = sorted({n.split("_")[1] for n, s in live.items() if s.get("subnormal_input_policy") == "mixed"},
                                      key=order(PRECISION_ORDER))
                row["subnormal_checked"] = any("subnormal_inputs_flushed" in s for s in live.values())
                row["ref_fail"] = sum(s.get("reference_check_failed_runs", 0) for s in live.values())
                row["rep_fail"] = sum(s.get("repeat_check_failed_runs", 0) for s in live.values())
                row["verdicts"] = sorted({s["_verdict"] for s in live.values()})
                row["clean_file"] = r["file"]
                row["counters"] = counter_verdict(r.get("counters"))
        elif r["run"] == "ai-chip-integrity-abft":
            row = row_for(checksum, r, torch=r["torch"])
            if row is None:
                continue
            steps = r["steps"].values()
            if inject:
                row["injected"] = sum(s.get("injected_runs", 0) for s in steps)
                row["caught"] = sum(s.get("injected_runs_located", 0) for s in steps)
                row["inject_file"] = r["file"]
            else:
                live = {n: s for n, s in r["steps"].items() if "_skipped" not in s}
                row["runs"] = sum(s.get("runs", 0) for s in live.values())
                row["skipped"] = len(r["steps"]) - len(live)
                row["precisions"] = sorted({n.split("_")[1] for n in live},
                                           key=lambda p: (PRECISION_ORDER.index(p) if p in PRECISION_ORDER else 99, p))
                row["shapes"] = r["params"].get("shapes", "")
                row["faults"] = sum(s.get("checksum_check_failed_runs", 0) + s.get("reference_check_failed_runs", 0)
                                    + s.get("repeat_check_failed_runs", 0) for s in live.values())
                row["missed"] = sum(s.get("checksum_missed_runs", 0) for s in live.values())
                row["thr_ratio"] = max((s.get("threshold_over_typical_value", 0) for s in live.values()), default=0)
                row["verdicts"] = sorted({s["_verdict"] for s in live.values()})
                row["clean_file"] = r["file"]
                row["counters"] = counter_verdict(r.get("counters"))
        elif r["run"] == "ai-chip-integrity-memcheck":
            row = row_for(memory, r)
            if row is None:
                continue
            s = next(iter(r["steps"].values()))
            if inject:
                row["injected"] = sum(v for k, v in s.items() if k.startswith("injected_") and "detected" not in k)
                row["caught"] = sum(v for k, v in s.items() if k.startswith("injected_detected_"))
                row["inject_file"] = r["file"]
            else:
                row["bytes"] = s["bytes_tested"]
                row["total"] = s.get("device_memory_bytes")
                row["patterns"] = sum(1 for k in s if k.startswith("bad_words_"))
                row["bad"] = sum(v for k, v in s.items() if k.startswith("bad_words_"))
                row["dwell"] = s.get("dwell_seconds")
                row["seconds"] = s.get("seconds")
                row["verdict"] = s["_verdict"]
                row["clean_file"] = r["file"]
                row["counters"] = counter_verdict(r.get("counters"))
    order = lambda d: (0 if "H100" in d else 1 if "A100" in d else 2 if "NVIDIA" in d else 3, d)
    return ([compute[k] for k in sorted(compute, key=order)], [memory[k] for k in sorted(memory, key=order)],
            [kernel[k] for k in sorted(kernel, key=order)], [pattern[k] for k in sorted(pattern, key=order)],
            [checksum[k] for k in sorted(checksum, key=order)])


def esc(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def gib(b):
    return f"{b / 2**30:.2f} GiB"


def compute_rows(rows):
    out = []
    for r in rows:
        if "runs" not in r:
            continue
        verdict = ", ".join(r["verdicts"])
        ok = r["verdicts"] == ["no-silent-errors"] and not counters_note(r)[1]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        prec = ", ".join(r["precisions"]) + (f' ({r["skipped"]} skipped)' if r["skipped"] else "")
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">PyTorch {esc(r['torch'])} · tool v{esc(r['version'])}{counters_note(r)[0]}</span></td>
<td class="date">{esc(min(r['dates']))}</td>
<td>{esc(r['shapes']).replace(',', '<br>')}</td>
<td>{esc(prec)}</td>
<td>{r['runs']:,}</td>
<td>{r['ref_fail'] + r['rep_fail']}</td>
<td>{caught}</td>
<td class="{'ok' if ok else 'bad'}">{esc(verdict)}</td>
<td>{files}</td>
</tr>""")
    return "\n".join(out)


def counters_note(r):
    """Probe 11's result for the row, as a short note under the device name, and whether it is a failure."""
    v = r.get("counters")
    if not v:
        return "", False
    return f" · {esc(COUNTER_NOTES.get(v, v))}", v in ("uncorrected-during-run", "silent-error-confirmed", "hardware-noticed")


def pattern_rows(rows):
    out = []
    for r in rows:
        if "runs" not in r:
            continue
        verdict = ", ".join(r["verdicts"])
        ok = r["verdicts"] == ["no-silent-errors"] and not counters_note(r)[1]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        prec = ", ".join(r["precisions"]) + (f' ({r["skipped"]} skipped)' if r["skipped"] else "")
        if not r["subnormal_checked"]:
            sub = "not run"
        elif r["flushed"] or r.get("mixed"):
            sub = "; ".join(f"{label}: " + ", ".join(r[key]) for key, label in (("flushed", "flushed"), ("mixed", "mixed"))
                            if r.get(key))
        else:
            sub = "kept"
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">PyTorch {esc(r['torch'])} · tool v{esc(r['version'])}{counters_note(r)[0]}</span></td>
<td class="date">{esc(min(r['dates']))}</td>
<td>{esc(", ".join(r['patterns']))}</td>
<td>{esc(prec)}</td>
<td>{r['runs']:,}</td>
<td>{r['ref_fail'] + r['rep_fail']}</td>
<td>{caught}</td>
<td>{esc(sub)}</td>
<td class="{'ok' if ok else 'bad'}">{esc(verdict)}</td>
<td>{files}</td>
</tr>""")
    return "\n".join(out)


def checksum_rows(rows):
    out = []
    for r in rows:
        if "runs" not in r:
            continue
        verdict = ", ".join(r["verdicts"])
        ok = r["verdicts"] == ["no-silent-errors"] and not counters_note(r)[1]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        prec = ", ".join(r["precisions"]) + (f' ({r["skipped"]} skipped)' if r["skipped"] else "")
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">PyTorch {esc(r['torch'])} · tool v{esc(r['version'])}{counters_note(r)[0]}</span></td>
<td class="date">{esc(min(r['dates']))}</td>
<td>{esc(r['shapes']).replace(',', '<br>')}</td>
<td>{esc(prec)}</td>
<td>{r['runs']:,}</td>
<td>{r['faults']}</td>
<td>{r['missed']}</td>
<td>{r['thr_ratio']:.2f}</td>
<td>{caught}</td>
<td class="{'ok' if ok else 'bad'}">{esc(verdict)}</td>
<td>{files}</td>
</tr>""")
    return "\n".join(out)


def kernel_sizes(r):
    n = int(r["size"]) if str(r["size"]).isdigit() else 0
    parts = [f"{n}×{n}"]
    if "rope" in r["kernels"]:
        parts.append(f"rope {n}×128")
    if "attention" in r["kernels"]:
        parts.append(f"attention 8×{n // 4}×128")
    return "; ".join(parts)


def attention_note(r):
    """Which fused attention kernel ran, per precision, when the probe chose it (CUDA); empty otherwise."""
    att = {k: v for k, v in (r.get("attention") or {}).items() if k != "default"}
    if not att:
        return ""
    order = lambda p: PRECISION_ORDER.index(p) if p in PRECISION_ORDER else 99
    return " · attention: " + "; ".join(f"{k} ({', '.join(sorted(v, key=order))})" for k, v in sorted(att.items()))


def kernel_rows(rows):
    out = []
    for r in rows:
        if "runs" not in r:
            continue
        verdict = ", ".join(r["verdicts"])
        ok = r["verdicts"] == ["no-silent-errors"] and not counters_note(r)[1]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        prec = ", ".join(r["precisions"]) + (f' ({r["skipped"]} skipped)' if r["skipped"] else "")
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">PyTorch {esc(r['torch'])} · tool v{esc(r['version'])}{counters_note(r)[0]}</span></td>
<td class="date">{esc(min(r['dates']))}</td>
<td>{esc(", ".join(r['kernels']))}<span class="sub">{kernel_sizes(r)}{esc(attention_note(r))}</span></td>
<td>{esc(prec)}</td>
<td>{r['runs']:,}</td>
<td>{math.floor(r['ratio'] * 1000) / 1000:.3f}</td>
<td>{r['ref_fail'] + r['rep_fail']}</td>
<td>{caught}</td>
<td class="{'ok' if ok else 'bad'}">{esc(verdict)}</td>
<td>{files}</td>
</tr>""")
    return "\n".join(out)


def memory_rows(rows):
    out = []
    for r in rows:
        if "bytes" not in r:
            continue
        ok = r["verdict"] == "no-memory-errors" and not counters_note(r)[1]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        total = f' of {gib(r["total"])}' if r.get("total") else ""
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">tool v{esc(r['version'])}{counters_note(r)[0]}</span></td>
<td class="date">{esc(min(r['dates']))}</td>
<td>{gib(r['bytes'])}{total}<span class="sub">{r['bytes'] // 4:,} words</span></td>
<td>{r['patterns']}</td>
<td>{r['dwell']} s</td>
<td>{r['bad']}</td>
<td>{caught}</td>
<td class="{'ok' if ok else 'bad'}">{esc(r['verdict'])}</td>
<td>{files}</td>
</tr>""")
    return "\n".join(out)


# ---- One page per chip --------------------------------------------------------------------------------------
# Every chip in the tables gets its own page at /chips/<slug>/, generated here from the same result files, so the
# pages can always be rebuilt from the repository and never say more than the files do.

SITE = "https://chipintegrity.org"
CHIPS_DIR = os.path.join(ROOT, "docs", "chips")
PASS_VERDICTS = {"no-silent-errors", "injection-self-test-pass", "no-memory-errors"}
PROBE_INFO = [  # table, anchor, roadmap number, title, what the probe does
    ("compute", "compute", 1, "Compute probe", "Matrix multiplies at several shapes and precisions. Every output element "
     "is checked against an exact or float64 reference under a proven rounding-error bound, and every run is compared "
     "bit for bit with the first run."),
    ("kernel", "kernels", 3, "Transformer kernel probe", "Softmax, layer norm, GELU, rotary embeddings and fused "
     "attention, each checked against a float64 reference under its own proven bound and against the first run bit "
     "for bit."),
    ("checksum", "checksum", 6, "Checksum probe", "A matrix multiply protected by row and column checksums (Huang and "
     "Abraham) with proven thresholds, so a wrong element is located to its exact row and column."),
    ("pattern", "patterns", 9, "Data pattern probe", "Inputs built to reach particular bits (wide exponents, long "
     "mantissas, cancellation, alternating signs, sparse inputs, subnormals and values near the top of the range), "
     "fed through the same matrix multiply checks."),
    ("memory", "memory", 2, "Memory probe", "Most of the device memory written with six patterns, left to dwell, and "
     "read back word by word."),
]


def chip_slug(device):
    return re.sub(r"[^a-z0-9]+", "-", device.lower()).strip("-")


def citation():
    """Version and DOI from CITATION.cff, so every page cites the current release."""
    with open(os.path.join(ROOT, "CITATION.cff"), encoding="utf-8") as f:
        text = f.read()
    return re.search(r"^version: (\S+)", text, re.M).group(1), re.search(r"^doi: (\S+)", text, re.M).group(1)


def chip_rows(tables):
    """{device: {table: row}} across the five tables."""
    out = defaultdict(dict)
    for name, rows in tables.items():
        for r in rows:
            out[r["device"]][name] = r
    return out


def chip_stats(rows):
    """The same totals as the home page, for one chip."""
    runs = sum(rows[k].get("runs", 0) for k in ("compute", "kernel", "pattern", "checksum") if k in rows)
    injected = sum(r.get("injected", 0) for r in rows.values())
    caught = sum(r.get("caught", 0) for r in rows.values())
    words = rows["memory"].get("bytes", 0) // 4 if "memory" in rows else 0
    faults = (sum(rows[k].get("ref_fail", 0) + rows[k].get("rep_fail", 0) for k in ("compute", "kernel", "pattern")
                  if k in rows) + rows.get("checksum", {}).get("faults", 0) + rows.get("memory", {}).get("bad", 0))
    return {"runs": runs, "words": words, "faults": faults, "caught": caught, "injected": injected}


def nice_date(d):
    day = datetime.strptime(d, "%Y-%m-%d")
    return f"{day.day} {day.strftime('%B %Y')}"


def chip_dates(rows):
    return sorted(set().union(*(r["dates"] for r in rows.values())))


def file_link(path, label):
    return f'<a href="{REPO}/blob/main/{esc(path)}">{esc(label)}</a>'


def step_table(run):
    """Every step of one result file, as the file records it."""
    if run is None:
        return ""
    body = []
    for name, s in run["steps"].items():
        if "_skipped" in s:
            body.append(f'<tr><td><code>{esc(name)}</code></td><td></td><td></td>'
                        f'<td><span class="sub">not supported on this device</span></td></tr>')
            continue
        size = f'{s["runs"]:,} runs' if "runs" in s else (gib(s["bytes_tested"]) if "bytes_tested" in s else "")
        ratio = s.get("worst_error_over_bound_ratio")
        if ratio is None:
            bound = "exact" if s.get("exact_check") else ""
        else:
            bound = "under 0.001" if ratio < 0.001 else f"{ratio:.3f}"
        verdict = s.get("_verdict", "")
        cls = "ok" if verdict in PASS_VERDICTS else "bad"
        body.append(f'<tr><td><code>{esc(name)}</code></td><td>{size}</td>'
                    f'<td>{bound}</td><td class="{cls}">{esc(verdict)}</td></tr>')
    return ('<div class="tablewrap"><table><thead><tr><th>Step in the result file</th><th>Size</th>'
            '<th>Worst error / bound</th><th>Verdict</th></tr></thead><tbody>' + "\n".join(body) + "</tbody></table></div>")


def chip_summary(st):
    if st["faults"] == 0:
        first = (f'Across {st["runs"]:,} compute runs, every output element stayed inside its proven error bound and '
                 f'every repeat matched the first run bit for bit.') if st["runs"] else "No compute rows yet."
        if st["words"]:
            first += f' All {st["words"]:,} memory words read back correctly.'
    else:
        first = f'{st["faults"]} checks failed on this chip. The probe tables below show which steps and why.'
    return first + (f' The self-tests caught {st["caught"]} of the {st["injected"]} faults planted to prove the checks '
                    f'work.' if st["injected"] else "")


def chip_card(device, rows, href):
    st, dates = chip_stats(rows), chip_dates(rows)
    return (f'<a class="panel chipcard" href="{href}"><h3>{esc(device)}</h3><p>Tested {nice_date(dates[-1])}. '
            f'{st["runs"]:,} compute runs, {st["faults"]} silent faults, {st["caught"]} of {st["injected"]} planted '
            f'faults caught.</p></a>')




PRECISION_LABEL = {"fp32": "FP32", "fp16": "FP16", "bf16": "BF16", "int8": "INT8", "fp8": "FP8",
                   "fp8fast": "FP8 fast-accumulation"}
KERNEL_LABEL = {"softmax": "Softmax", "layernorm": "Layer norm", "gelu": "GELU", "rope": "RoPE", "attention": "Attention"}


def bar(label, frac, value, cls="", limit=False):
    """One labelled bar. frac is the filled share of the track, 0 to 1."""
    width = max(0.0, min(frac, 1.0)) * 100
    return (f'<div class="brow{(" " + cls) if cls else ""}"><div class="lab">{label}</div><div class="track">'
            f'<div class="fill" style="width:{width:.1f}%"></div>{"<div class=limitline></div>" if limit else ""}</div>'
            f'<div class="val">{value}</div></div>')


def budget_rows(run, group):
    """Worst error over bound for each precision (or kernel) in one result file: how much of the proven error budget
    the chip used. Exact checks are listed apart, because any difference at all fails them."""
    if run is None:
        return [], []
    worst, exact = {}, set()
    for name, s in run["steps"].items():
        if "_skipped" in s:
            continue
        key = group(name)
        if s.get("exact_check"):
            exact.add(key)
        elif s.get("worst_error_over_bound_ratio") is not None:
            worst[key] = max(worst.get(key, 0.0), s["worst_error_over_bound_ratio"])
    return sorted(worst.items(), key=lambda kv: -kv[1]), sorted(exact)


def die_svg(st):
    cells = []
    for i in range(14):
        for j in range(6):
            cells.append(f'<rect x="{34 + i * 27}" y="{34 + j * 27}" width="21" height="21" class="cell" '
                         f'style="animation-delay:{(i * 6 + j) * 0.02:.2f}s"/>')
    caption = f'{st["runs"]:,} runs, every output checked' if st["runs"] else "memory checked word by word"
    verdict = "0 wrong answers" if st["faults"] == 0 else f'{st["faults"]} failed checks'
    return (f'<svg class="chipfig" viewBox="0 0 440 260" role="img" aria-label="{caption}, {verdict}">'
            f'<rect x="20" y="20" width="400" height="190" class="frame"/>{"".join(cells)}'
            f'<rect x="20" y="20" width="3" height="190" class="scan"/>'
            f'<text x="20" y="236" class="cap">{caption.upper()}</text>'
            f'<text x="420" y="236" text-anchor="end" class="cap {"good" if st["faults"] == 0 else "badtxt"}">'
            f'{verdict.upper()}</text></svg>')


def spec_sheet(groups):
    out = []
    for title, items in groups:
        cells = "".join(f'<div><div class="k">{k}</div><div class="v">{v}</div></div>' for k, v in items if v)
        out.append(f'<h3>{title}</h3><div class="spec">{cells}</div>')
    return "".join(out)


def chip_budget(rows, runs_by_file):
    """How much of its proven error bound each precision and kernel used, worst first, plus the exact-checked ones."""
    budget, exact_list = [], []
    for name, label, group in (("compute", "matrix multiply", lambda n: n.split("_")[1]),
                               ("pattern", "data patterns", lambda n: n.split("_")[1]),
                               ("kernel", "", lambda n: n)):
        if name not in rows:
            continue
        worst, exact = budget_rows(runs_by_file.get(rows[name].get("clean_file")), group)
        for key, ratio in worst:
            if label:
                lab = f"{PRECISION_LABEL.get(key, key)} {label}"
            else:
                kern, _, prec = key.partition("_")
                lab = f"{KERNEL_LABEL.get(kern, kern)} {PRECISION_LABEL.get(prec, prec)}"
            budget.append((ratio, lab))
        exact_list += [f"{PRECISION_LABEL.get(k, k)} {label}".strip() for k in exact]
    return sorted(budget, key=lambda x: -x[0]), list(dict.fromkeys(exact_list))


def chip_page(device, chip_map, runs_by_file, built):
    rows, devices = chip_map[device], sorted(chip_map)
    slug, st, dates = chip_slug(device), chip_stats(rows), chip_dates(rows)
    here = ' aria-current="page"'
    version, doi = citation()
    url = f"{SITE}/chips/{slug}.html"
    files = [(r.get(k), lbl) for name, r in sorted(rows.items()) for k, lbl in (("clean_file", "result"), ("inject_file", "self-test"))
             if r.get(k)]
    title = f"{device}: silent computation error test results"
    desc = (f'{device} tested by Chip Integrity: {st["runs"]:,} compute runs checked element by element, '
            f'{st["faults"]} silent faults found, {st["caught"]} of {st["injected"]} planted faults caught. '
            f'Every result file is public.')
    ld = {"@context": "https://schema.org", "@type": "Dataset", "name": f"Silent computation error tests: {device}",
          "description": desc, "url": url, "license": "https://opensource.org/licenses/MIT", "isAccessibleForFree": True,
          "creator": {"@type": "Person", "name": "Yasha Khandelwal"}, "dateModified": dates[-1],
          "measurementTechnique": "OCP Test and Validation output from ai-chip-integrity",
          "distribution": [{"@type": "DataDownload", "encodingFormat": "application/x-ndjson",
                            "contentUrl": f"{REPO}/blob/main/{f}"} for f, _ in files]}
    present = [p for p in PROBE_INFO if p[0] in rows]
    toc = "".join(f'<a href="#{a}">{t}</a>' for a, t in
                  [("top", "Summary"), ("budget", "Error budget used"), ("proof", "Proof the checks work")]
                  + [(a, t) for _, a, _, t, _ in present]
                  + [("setup", "Setup and files"), ("limits", "What this result shows"), ("run", "Run it on your chip"),
                     ("chips", "Other chips")])

    budget, exact_list = chip_budget(rows, runs_by_file)
    budget = [(r, bar(esc(l), r, ('under 0.1% of bound' if r < 0.001 else f'{r * 100:.1f}% of bound'), limit=True)) for r, l in budget]
    budget_html = "".join(b for _, b in sorted(budget, key=lambda x: -x[0]))
    exact_html = (f'<p class="note">Checked exactly, where any difference at all fails: '
                  f'{esc(", ".join(exact_list))}. Every one matched.</p>' if exact_list else "")

    # Proof: every planted fault caught, probe by probe, and how much memory the sweep covered.
    proof = "".join(bar(esc(t), r["caught"] / r["injected"], f'{r["caught"]} of {r["injected"]} caught', cls="proof")
                    for n, _, _, t, _ in present for r in [rows[n]] if r.get("injected"))
    mem_run = runs_by_file.get(rows.get("memory", {}).get("clean_file"))
    if mem_run:
        tested = max(s.get("bytes_tested", 0) for s in mem_run["steps"].values())
        total = max(s.get("device_memory_bytes", 0) for s in mem_run["steps"].values())
        if total:
            proof += bar("Device memory swept", tested / total, f"{gib(tested)} of {gib(total)}", cls="proof")

    sections = []
    for name, anchor, number, ptitle, what in present:
        r = rows[name]
        run = runs_by_file.get(r.get("clean_file"))
        nsteps = sum(1 for s in run["steps"].values()) if run else 0
        meta = f'Tool v{esc(r["version"])}' + (f', PyTorch {esc(r["torch"])}' if r.get("torch") else "")
        meta += ", " + ", ".join(nice_date(d) for d in sorted(r["dates"]))
        bad = r.get("ref_fail", 0) + r.get("rep_fail", 0) + r.get("faults", 0) + r.get("bad", 0)
        size = (f'{r.get("bytes", 0) // 4:,} memory words read back' if name == "memory"
                else f'{r.get("runs", 0):,} runs across {r.get("steps", nsteps)} steps')
        verdict = ("No silent faults." if bad == 0 else f"{bad} failed checks.")
        selftest = (f' The self-test caught {r["caught"]} of {r["injected"]} planted faults.' if r.get("injected") else "")
        links = " · ".join(x for x in [file_link(r["clean_file"], "Result file") if r.get("clean_file") else "",
                                       file_link(r["inject_file"], "Self-test file") if r.get("inject_file") else ""] if x)
        counters = (f'<p class="note">Error counters: {esc(COUNTER_NOTES.get(r["counters"], r["counters"]))}.</p>'
                    if r.get("counters") else "")
        sections.append(
            f'<section id="{anchor}"><div class="kicker">PROBE {number}</div><h2>{ptitle}</h2>'
            f'<p class="lede">{what}</p>'
            f'<div class="probehead"><div class="big {"good" if bad == 0 else "badtxt"}">{verdict}</div>'
            f'<div>{size}.{selftest}</div></div>'
            f'<p class="note">{meta}. {links}</p>{counters}'
            f'<details><summary>Show all {nsteps} steps from the result file</summary>{step_table(run)}</details>'
            "</section>")
    stats = "".join(f'<div class="stat"><div class="n">{n}</div><div class="l">{l}</div></div>' for n, l in [
        (len(present), "probes run on this chip"), (f'{st["runs"]:,}', "compute runs checked element by element"),
        (f'{st["words"] / 1e9:.1f} billion' if st["words"] else "0", "memory words written and read back"),
        (st["faults"], "silent faults found"), (f'{st["caught"]} of {st["injected"]}', "planted faults caught in self-tests")])
    tiles = []
    for name, anchor, number, ptitle, _ in present:
        r = rows[name]
        size = (f'{r.get("bytes", 0) // 4 / 1e9:.1f} billion words' if name == "memory" else f'{r.get("runs", 0):,} runs')
        bad = (r.get("ref_fail", 0) + r.get("rep_fail", 0) + r.get("faults", 0) + r.get("bad", 0))
        state = "no silent faults" if bad == 0 else f"{bad} failed checks"
        tiles.append(f'<a class="stat" href="#{anchor}"><div class="n">{ptitle}</div><div class="l">Probe {number} · '
                     f'{size} · {state}</div></a>')
    others = "".join(chip_card(d, r, f"{chip_slug(d)}.html") for d, r in sorted(chip_map.items()) if d != device)
    torch_versions = sorted({r["torch"] for r in rows.values() if r.get("torch")})
    counters_seen = sorted({COUNTER_NOTES.get(r["counters"], r["counters"]) for r in rows.values() if r.get("counters")})
    spec = spec_sheet([
        ("The chip", [("Name, as the driver reports it", esc(device)),
                      ("Device memory", gib(total) if mem_run and total else ""),
                      ("Error counters", esc("; ".join(counters_seen)))]),
        ("The software", [("PyTorch", esc(", ".join(torch_versions))),
                          ("Probe versions", ", ".join(f'{t} v{esc(rows[n]["version"])}' for n, _, _, t, _ in present))]),
        ("The test", [("Dates tested", ", ".join(nice_date(d) for d in dates)),
                      ("Result files", f"{len(files)} files in the OCP Test and Validation format")])])
    file_list = "".join(f'<li>{file_link(f, f.split("/")[-1])} <span class="sub">{lbl}</span></li>' for f, lbl in files)
    body = f"""<div class="wrap chiplayout"><div id="top">
<div class="hero">
<div class="crumb"><a href="../index.html">chipintegrity.org</a> / <a href="index.html">chips</a> / {esc(slug)}</div>
<div class="hero-grid">
<div>
<div class="kicker">CHIP RESULT, LAST TESTED {nice_date(dates[-1]).upper()}</div>
<h1>{esc(device)}</h1>
<p class="lede">{chip_summary(st)}</p>
<p>Every number on this page is read from the result files listed at the bottom, written in the OCP Test and
Validation format. New to this? Start with <a href="../learn.html#silent">what a silent error is</a> and
<a href="../method.html#method">how a chip is tested here</a>.</p>
</div>
<div class="fig" aria-hidden="true">{die_svg(st)}</div>
</div>
<div class="stats">{stats}</div>
<div class="probemap">{"".join(tiles)}</div>
</div>
<section id="budget"><div class="kicker">HOW CLOSE IT CAME</div><h2>Error budget used</h2>
<p class="lede">Floating-point answers are never exact, so each check allows a rounding error and proves how large it
can honestly be. The red line is that proven limit. A bar shows the largest error this chip produced, as a share of
the limit, across every run. Anything past the line would be a wrong answer.</p>
<div class="budget">{budget_html}</div>{exact_html}
<p class="note">Bars close to the line at FP16 and BF16 are expected: there the limit is dominated by the single
rounding of the output, so a correct chip has almost nothing to spare. A bar past the line is a failed run.</p></section>
<section id="proof"><div class="kicker">PROOF THE CHECKS WORK</div><h2>Every planted fault was caught</h2>
<p class="lede">A clean result is only worth something if the tester would have seen a fault. So next to every clean
run, the self-test plants faults on purpose, including the patterns NVIDIA measured at gate level, and every one must
be caught. <a href="../method.html#selftest">How the self-test works</a>.</p>
<div class="budget">{proof}</div></section>
{"".join(sections)}
<section id="setup"><div class="kicker">EVIDENCE</div><h2>Setup and files</h2>
{spec}
<h3>Result files</h3><ul class="files">{file_list}</ul></section>
<section id="limits"><div class="callout"><div class="kicker">READ BEFORE RELYING ON IT</div><h2>What this result shows</h2>
<p>These numbers describe one unit of this chip, tested on the dates above. A clean result means no silent error
showed up in these runs under these checks. It cannot rule out a fault that appears under other workloads,
temperatures or ages, and another unit of the same model can behave differently. <a href="../method.html#limits">What a PASS
means</a> explains the limits in full.</p></div></section>
<section id="run"><div class="kicker">TAKE PART</div><h2>Run it on your chip</h2>
<p class="lede">The same tests on your own hardware, in one command. Every result file it writes can be published
here as a new row.</p>
<pre>pip install ai-chip-integrity
chip-integrity run --name {slug}</pre>
<p class="note">Cite the tool as: Yasha Khandelwal, ai-chip-integrity, version {esc(version)}, 2026,
<a href="https://doi.org/{esc(doi)}">https://doi.org/{esc(doi)}</a></p></section>
<section id="chips"><div class="kicker">ALL CHIPS</div><h2>Other chips measured</h2>
<div class="grid3">{others}</div></section>
</div><aside class="toc"><div class="fh">ON THIS PAGE</div>{toc}</aside></div>"""
    body = re.sub(r'<section id="([a-z]+)">', r'<section id="\1" class="reveal">', body)
    head = f'<script type="application/ld+json">{json.dumps(ld)}</script>'
    return render("chips/index", f"{title} | Chip Integrity", desc, url, body, "../", built, head, og=f"chips-{slug}")


def chips_index(chip_map, built):
    """chips/index.html: every chip measured, one card each, like the stores index on agentmemoryintegrity.org."""
    cards = "".join(chip_card(d, r, f"{chip_slug(d)}.html") for d, r in sorted(chip_map.items()))
    body = (f'<div class="wrap"><div class="pagehead"><div class="crumb"><a href="../index.html">chipintegrity.org</a> / chips</div>'
            f'<div class="kicker">EVIDENCE</div><h1>Every chip measured</h1><p class="lede">One page per chip, each generated '
            f'from its result files: what was run, how close every precision came to its proven error limit, and proof '
            f'that every planted fault was caught.</p></div><section class="reveal"><div class="grid3">{cards}</div>'
            f'</section></div>')
    return render("chips/index", "Every chip measured | Chip Integrity", "Every AI chip tested by Chip Integrity for "
                  "silent computation errors, one page each, generated from public result files.",
                  f"{SITE}/chips/index.html", body, "../", built)


def finding_page(slug, built):
    date, title, summary = next((d, t, m) for sl, d, t, m in FINDINGS if sl == slug)
    body = (f'<div class="wrap"><div class="pagehead"><div class="crumb"><a href="../index.html">chipintegrity.org</a> / '
            f'<a href="index.html">findings</a> / {slug}</div><div class="kicker">FINDING, {nice_date(date).upper()}</div>'
            f'<h1>{esc(title)}</h1><p class="lede">{esc(summary)}</p></div><section class="reveal"><article class="post">'
            + add_terms(read_site(f"findings/{slug}.html"), glossary(read_site("pages/learn.html")), "../")
            + '</article></section></div>')
    return render("findings/index", f"{title} | Chip Integrity", summary, f"{SITE}/findings/{slug}.html", body, "../",
                  built, og=f"findings-{slug}")


def findings_index(built):
    cards = "".join(f'<a class="panel finding" href="{sl}.html"><div class="date">{nice_date(d)}</div><h3>{esc(t)}</h3>'
                    f'<p>{esc(m)}</p></a>' for sl, d, t, m in FINDINGS)
    body = (f'<div class="wrap"><div class="pagehead"><div class="crumb"><a href="../index.html">chipintegrity.org</a> / '
            f'findings</div><div class="kicker">FINDINGS</div><h1>What the tests have turned up</h1><p class="lede">Short '
            f'write-ups of what measuring real chips has shown, each with its evidence. New ones also appear in the '
            f'<a href="../feed.xml">updates feed</a>.</p></div><section class="reveal"><div class="grid3">{cards}</div>'
            f'</section></div>')
    return render("findings/index", "Findings | Chip Integrity", "What testing real AI chips for silent computation "
                  "errors has turned up, each with its evidence.", f"{SITE}/findings/index.html", body, "../", built,
                  og="findings")


def feed(built):
    """An Atom feed of findings and releases, newest first."""
    items = [(d, t, f"{SITE}/findings/{sl}.html", m) for sl, d, t, m in FINDINGS]
    items += [(d, f"Release {v}: {t}", f"{REPO}/releases/tag/v{v}", t) for v, d, t in RELEASES]
    items.sort(key=lambda x: x[0], reverse=True)
    entries = "".join(f"<entry><title>{esc(t)}</title><link href=\"{u}\"/><id>{u}</id><updated>{d}T00:00:00Z</updated>"
                      f"<summary>{esc(m)}</summary></entry>" for d, t, u, m in items)
    return (f'<?xml version="1.0" encoding="utf-8"?>\n<feed xmlns="http://www.w3.org/2005/Atom"><title>Chip Integrity'
            f'</title><subtitle>Findings and releases from an open tester for silent computation errors in AI chips'
            f'</subtitle><link href="{SITE}/"/><link rel="self" href="{SITE}/feed.xml"/><id>{SITE}/</id><updated>'
            f'{items[0][0]}T00:00:00Z</updated><author><name>Yasha Khandelwal</name></author>{entries}</feed>\n')


# ---- The site: six pages and the chip pages, all through one layout ------------------------------------------

SITE_DIR = os.path.join(ROOT, "site")
NAV = [("learn", "Learn"), ("method", "Method"), ("results", "Results"), ("chips/index", "Chips"),
       ("findings/index", "Findings"), ("run", "Run it"), ("about", "About")]
FINDINGS = [  # newest first: slug, date, title, one-line summary
    ("nvidia-fault-patterns", "2026-10-15", "NVIDIA's own fault patterns, caught on a real GPU",
     "The self-test now plants faults shaped like NVIDIA's gate-level measurements. On an RTX 4090 the reference check "
     "alone flagged all 294."),
    ("ecc-off", "2026-10-11", "When ECC is off, the memory error counters cannot see anything",
     "Two rented cards ran with ECC switched off, so their counters could not move. The tester now says so instead of "
     "calling them quiet."),
    ("intel-amx-bf16", "2026-10-10", "BF16 matrix multiplies on Intel AMX treat tiny inputs as zero",
     "Above 16 × 16 × 16, BF16 matrix multiplies on AMX CPUs treat subnormal inputs as zero, whatever "
     "set_flush_denormal says."),
]
RELEASES = [("0.4.2", "2026-10-15", "Self-test faults shaped like NVIDIA's gate-level measurements"),
            ("0.4.1", "2026-10-11", "The ecc-off verdict, faster exact references, and the RTX 4090 and RTX A5000 rows"),
            ("0.4.0", "2026-10-10", "Six probes, one command, OCP Test and Validation output")]
PAGES = {
    "learn": dict(kicker="LEARN", h1="Silent errors in AI chips, from the beginning",
                  title="What a silent error is, and why it matters | Chip Integrity",
                  intro="Start here if the subject is new. This page explains how a chip can return a wrong number without "
                        "raising any error, why that matters for anyone training or serving AI models, and the words used "
                        "everywhere else on this site.",
                  learn=["What a silent computation error is, with a worked example",
                         "Why it matters at the scale AI now runs at", "The terms used in the method and the results"]),
    "method": dict(kicker="METHOD", h1="The method behind every verdict", title="How a chip is tested | Chip Integrity",
                   intro="Every probe follows the same idea: run real AI maths many times, check each answer two ways, and "
                         "prove the checks work by planting faults on purpose. This page walks through each probe, how a "
                         "verdict is read, the self-test, and what a PASS can and cannot tell you.",
                   learn=["The probes, and what each one exercises",
                          "The two checks behind every verdict: a proven error bound and a bit-for-bit repeat",
                          "How the self-test shows a clean result means something", "What a PASS means, and its limits"]),
    "results": dict(kicker="EVIDENCE", h1="Results from every chip measured", title="Measured results | Chip Integrity",
                    intro="Every row on this page is generated from a result file in the repository, and every chip also "
                          "has its own page with the full detail. Nothing here is modelled or estimated.",
                    learn=["Which chips have been measured, and when", "What each probe found on each chip",
                           "Where to download every raw result file"]),
    "run": dict(kicker="TAKE PART", h1="Test your own chip", title="Run it on your chip | Chip Integrity",
                intro="The same tests run on an NVIDIA GPU, an Apple GPU or a CPU with one command. This page shows how to "
                      "install and run them, and what is planned next.",
                learn=["How to install the tester and run every probe on your own chip",
                       "What is planned next on the roadmap"]),
    "about": dict(kicker="ABOUT", h1="Research, standards and how to cite",
                  title="Research, standards and how to cite | Chip Integrity",
                  intro="The checks build on published work in numerical analysis and hardware reliability, and write their "
                        "results in an open industry format. This page lists that work, who maintains the project, and how "
                        "to cite it.",
                  learn=["The papers and standards the checks rely on", "Who maintains the project", "How to cite the tool"]),
}
HOME_TITLE = "Chip Integrity: an open tester for silent computation errors in AI chips"
HOME_DESC = ("A GPU can pass its vendor diagnostics and still return wrong numbers with no error raised. Chip Integrity is "
             "an open, vendor-neutral tester that checks whether an AI chip computed correctly, with every result "
             "published as an OCP Test and Validation file.")


GLOSSARY_SKIP = {"fp32, fp16, bf16, FP8, int8"}       # precision names appear in every table; no hover links for them
NO_TERMS_INSIDE = {"a", "h1", "h2", "h3", "h4", "code", "pre", "script", "style", "svg", "button", "label", "select",
                   "option", "figcaption", "dt", "dd", "title", "summary", "textarea", "output"}


def glossary(learn_src):
    """[(aliases, slug, definition)] from the glossary on the Learn page."""
    out = []
    for dt, dd in re.findall(r"<dt>(.*?)</dt>\s*<dd>(.*?)</dd>", learn_src, re.S):
        name = re.sub(r"<[^>]+>", "", dt).strip()
        if name in GLOSSARY_SKIP:
            continue
        aliases = [x.strip() for part in name.split(",") for x in re.split(r"[()]", part) if x.strip()]
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", dd)).strip()
        out.append((aliases, chip_slug(aliases[0]), text))
    return out


def add_terms(html, terms, prefix):
    """Link the first use of each glossary term on a page to its definition, with the definition on hover."""
    used, stack, out = set(), [], []
    pairs = sorted(((a, slug, d) for aliases, slug, d in terms for a in aliases), key=lambda x: -len(x[0]))
    for tok in re.split(r"(<[^>]+>)", html):
        if tok.startswith("<"):
            m = re.match(r"<(/?)([a-zA-Z0-9]+)", tok)
            if m and not m.group(1) and 'class="kicker"' in tok:      # small capitals labels never carry links
                stack.append(m.group(2).lower())
            elif m and m.group(2).lower() in NO_TERMS_INSIDE and not tok.endswith("/>"):
                if m.group(1):
                    if stack and stack[-1] == m.group(2).lower():
                        stack.pop()
                else:
                    stack.append(m.group(2).lower())
            elif m and m.group(1) and stack and stack[-1] == m.group(2).lower() and m.group(2).lower() not in NO_TERMS_INSIDE:
                stack.pop()                                               # the end of a kicker label
            out.append(tok)
            continue
        if not stack:
            for alias, slug, text in pairs:
                if slug in used:
                    continue
                mm = re.search(rf"(?<![\w-]){re.escape(alias)}s?(?![\w-])", tok, re.I if len(alias) > 4 else 0)
                if mm:
                    used.add(slug)
                    tok = (tok[:mm.start()] + f'<a class="term" href="{prefix}learn.html#g-{slug}" data-def="{esc(text)}">'
                           + mm.group(0) + "</a>" + tok[mm.end():])
                    break
        out.append(tok)
    return "".join(out)


def make_og(name, kicker, title, sub):
    """A 1200 x 630 share image, drawn only where Pillow is installed; the images are committed with the site."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        return False
    fonts = os.path.join(SITE_DIR, "og-fonts")
    big, small, mono = (ImageFont.truetype(os.path.join(fonts, f), size) for f, size in
                        (("IBMPlexSans-SemiBold.ttf", 64), ("IBMPlexSans-Regular.ttf", 30), ("IBMPlexMono-Regular.ttf", 22)))
    im = Image.new("RGB", (1200, 630), "#080D18")
    d = ImageDraw.Draw(im)
    for x in range(13, 1200, 26):
        for y in range(13, 630, 26):
            d.point((x, y), fill="#152036")
    s = 1.0
    for x in (23, 32, 41):                                   # the chip mark, as in the logo
        for (x0, y0, x1, y1) in ((x, 7, x, 14), (x, 50, x, 57), (7, x, 14, x), (50, x, 57, x)):
            d.line([(72 + x0 * s, 64 + y0 * s), (72 + x1 * s, 64 + y1 * s)], fill="#39B9EE", width=3)
    d.rounded_rectangle([72 + 14, 64 + 14, 72 + 50, 64 + 50], radius=5, outline="#39B9EE", width=3)
    d.rounded_rectangle([72 + 37.5, 64 + 20, 72 + 44, 64 + 26.5], radius=1, fill="#2BD7B3")
    d.text((150, 80), "CHIPINTEGRITY.ORG", font=mono, fill="#E8ECF4")
    d.text((72, 190), kicker.upper(), font=mono, fill="#39B9EE")
    words, lines, line = title.split(), [], ""
    for w in words:
        if d.textlength(f"{line} {w}".strip(), font=big) > 1050 and line:
            lines.append(line)
            line = w
        else:
            line = f"{line} {w}".strip()
    lines.append(line)
    y = 232
    for ln in lines[:3]:
        d.text((72, y), ln, font=big, fill="#E8ECF4")
        y += 78
    d.text((72, y + 18), sub, font=small, fill="#93A0BA")
    d.line([(72, 560), (1128, 560)], fill="#1A2540", width=1)
    d.text((72, 578), "Open tests for silent computation errors in AI chips", font=mono, fill="#8291AD")
    os.makedirs(os.path.join(ROOT, "docs", "og"), exist_ok=True)
    im.save(os.path.join(ROOT, "docs", "og", f"{name}.png"), optimize=True)
    return True


def read_site(name):
    with open(os.path.join(SITE_DIR, name), encoding="utf-8") as f:
        return f.read()


def render(page, title, desc, canonical, body, prefix, built, head_extra="", og=None):
    """Wrap a page body in the shared layout. prefix is how far the page sits below the site root."""
    current = ' aria-current="page"'
    nav = "\n".join(f'<a href="{prefix}{key}.html"{current if key == page else ""}>{label}</a>' for key, label in NAV)
    foot = "".join(f'<a href="{prefix}{key}.html">{label}</a>' for key, label in [("index", "Home")] + NAV)
    version, doi = citation()
    cite = (f'Yasha Khandelwal, ai-chip-integrity, version {esc(version)}, 2026, '
            f'<a href="https://doi.org/{esc(doi)}">https://doi.org/{esc(doi)}</a>')
    og = og or {"index": "home", "chips/index": "chips", "findings/index": "findings"}.get(page, page)
    html = read_site("layout.html")
    for key, value in (("OG", f"{SITE}/og/{og}.png"), ("TITLE", esc(title)), ("DESC", esc(desc)), ("CANONICAL", canonical), ("P", prefix),
                       ("PAGE", page), ("NAV", nav), ("FOOTNAV", foot), ("CITE", cite), ("BUILT", built),
                       ("HEAD_EXTRA", head_extra), ("BODY", body)):
        html = html.replace("{{" + key + "}}", value)
    assert not re.search(r"{{[A-Z_]+}}", html), f"unfilled placeholder on {page}"
    return html


def site_page(name, fills, built):
    body = read_site(f"pages/{name}.html")
    for key, value in fills.items():
        body = body.replace("{{" + key + "}}", value)
    if name == "index":
        version, doi = citation()
        ld = re.sub(r'"version": "[^"]*"', f'"version": "{version}"', read_site("home-ld.json.html"))
        ld = re.sub(r'"identifier": "https://doi\.org/[^"]*"', f'"identifier": "https://doi.org/{doi}"', ld)
        body = add_terms(body, glossary(read_site("pages/learn.html")), "")
        return render("index", HOME_TITLE, HOME_DESC, f"{SITE}/", body, "", built, ld)
    meta = PAGES[name]
    if name == "learn":
        body = re.sub(r"<dt>(.*?)</dt>", lambda m: f'<dt id="g-{chip_slug(re.split(r"[,(]", re.sub("<[^>]+>", "", m.group(1)))[0].strip())}">{m.group(1)}</dt>', body)
    body = add_terms(body, glossary(read_site("pages/learn.html")), "")
    head = (f'<div class="pagehead"><div class="kicker">{meta["kicker"]}</div><h1>{meta["h1"]}</h1>'
            f'<p class="lede">{meta["intro"]}</p><div class="learnh">ON THIS PAGE YOU WILL LEARN</div><ol class="learn">'
            + "".join(f"<li>{x}</li>" for x in meta["learn"]) + "</ol></div>")
    body = re.sub(r'<section id="([a-z0-9]+)">', r'<section id="\1" class="reveal">', body.replace("{{INTRO}}", head))
    return render(name, meta["title"], meta["intro"], f"{SITE}/{name}.html", f'<div class="wrap">{body}</div>', "", built)


def sitemap(devices, built):
    urls = ([(f"{SITE}/", "1.0")] + [(f"{SITE}/{p}.html", "0.9") for p in PAGES]
            + [(f"{SITE}/chips/index.html", "0.9"), (f"{SITE}/findings/index.html", "0.9"), (f"{SITE}/llms.txt", "0.6")]
            + [(f"{SITE}/findings/{sl}.html", "0.8") for sl, *_ in FINDINGS]) + [(f"{SITE}/chips/{chip_slug(d)}.html", "0.8") for d in devices]
    body = "\n".join(f"  <url><loc>{u}</loc><lastmod>{built}</lastmod><changefreq>weekly</changefreq>"
                     f"<priority>{p}</priority></url>" for u, p in urls)
    return f'<?xml version="1.0" encoding="UTF-8"?>\n<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n{body}\n</urlset>\n'


def llms_measured(chip_map):
    lines = ["## Measured so far", "", "One page per chip, generated from the result files:"]
    for d, rows in sorted(chip_map.items()):
        st = chip_stats(rows)
        lines.append(f'- {d}: {SITE}/chips/{chip_slug(d)}.html (last tested {chip_dates(rows)[-1]}; {st["runs"]:,} compute '
                     f'runs, {st["faults"]} silent faults, {st["caught"]} of {st["injected"]} planted faults caught)')
    return "\n".join(lines) + "\n"


def main():
    runs = [parse(p) for p in sorted(glob.glob(os.path.join(RESULTS, "*.jsonl")))]
    compute, memory, kernel, pattern, checksum = summarise(runs)
    chips = sorted({r["device"] for r in runs})
    total_runs = sum(r.get("runs", 0) for r in compute + kernel + pattern + checksum)
    total_caught = sum(r.get("caught", 0) for r in compute + kernel + pattern + checksum + memory)
    total_injected = sum(r.get("injected", 0) for r in compute + kernel + pattern + checksum + memory)
    total_words = sum(r.get("bytes", 0) for r in memory) // 4
    faults = (sum(r.get("ref_fail", 0) + r.get("rep_fail", 0) for r in compute + kernel + pattern) + sum(r.get("faults", 0) for r in checksum)
              + sum(r.get("bad", 0) for r in memory))
    built = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    chip_map = chip_rows({"compute": compute, "memory": memory, "kernel": kernel, "pattern": pattern,
                          "checksum": checksum})
    devices = sorted(chip_map)
    assert len({chip_slug(d) for d in devices}) == len(devices), "two chips share a page address"
    cards = "".join(chip_card(d, chip_map[d], f"chips/{chip_slug(d)}.html") for d in devices)
    fills = {
        "CHIPS": str(len(chips)), "RUNS": f"{total_runs:,}", "WORDS": f"{total_words / 1e9:.1f} billion",
        "WORDS_NUM": f"{total_words / 1e9:.1f}", "FAULTS": str(faults), "CAUGHT": f"{total_caught} of {total_injected}",
        "CAUGHT_NUM": str(total_caught), "CHIP_CARDS": cards, "CHIP_JSON": json.dumps(
            [dict(name=d, slug=chip_slug(d), **chip_stats(chip_map[d]),
                  budget=[[l, round(r, 4)] for r, l in chip_budget(chip_map[d], {r["file"]: r for r in runs})[0]])
             for d in devices]).replace("</", "<\\/"),
        "COMPUTE_ROWS": compute_rows(compute), "MEMORY_ROWS": memory_rows(memory),
        "CHECKSUM_ROWS": checksum_rows(checksum) or '<tr><td colspan="11">No rows yet. The checksum probe is new; rows are added as the files come in.</td></tr>',
        "PATTERN_ROWS": pattern_rows(pattern) or '<tr><td colspan="10">No rows yet. The data pattern probe is new; rows are added as the files come in.</td></tr>',
        "KERNEL_ROWS": kernel_rows(kernel) or '<tr><td colspan="10">No rows yet. The kernel probe was added after the first four chips were measured; rows are added as the files come in.</td></tr>',
    }
    docs = os.path.join(ROOT, "docs")
    for name in ["index"] + list(PAGES):
        with open(os.path.join(docs, f"{name}.html"), "w", encoding="utf-8") as f:
            f.write(site_page(name, fills, built))
    if os.path.isdir(os.path.join(docs, "assets")):
        shutil.rmtree(os.path.join(docs, "assets"))
    shutil.copytree(os.path.join(SITE_DIR, "assets"), os.path.join(docs, "assets"))
    for src, dst in (("style.css", "site.css"), ("site.js", "site.js")):
        shutil.copyfile(os.path.join(SITE_DIR, src), os.path.join(docs, "assets", dst))
    runs_by_file = {r["file"]: r for r in runs}
    if os.path.isdir(CHIPS_DIR):
        shutil.rmtree(CHIPS_DIR)
    os.makedirs(CHIPS_DIR)
    for d in devices:
        with open(os.path.join(CHIPS_DIR, f"{chip_slug(d)}.html"), "w", encoding="utf-8") as f:
            f.write(chip_page(d, chip_map, runs_by_file, built))
    with open(os.path.join(CHIPS_DIR, "index.html"), "w", encoding="utf-8") as f:
        f.write(chips_index(chip_map, built))
    fdir = os.path.join(docs, "findings")
    if os.path.isdir(fdir):
        shutil.rmtree(fdir)
    os.makedirs(fdir)
    for slug, *_ in FINDINGS:
        with open(os.path.join(fdir, f"{slug}.html"), "w", encoding="utf-8") as f:
            f.write(finding_page(slug, built))
    with open(os.path.join(fdir, "index.html"), "w", encoding="utf-8") as f:
        f.write(findings_index(built))
    with open(os.path.join(docs, "feed.xml"), "w", encoding="utf-8") as f:
        f.write(feed(built))
    shots = [("home", "Open source, MIT", "Does this AI chip still give the right answer?", "Open, vendor-neutral tests for silent computation errors")]
    shots += [(n, m["kicker"], m["h1"], "chipintegrity.org") for n, m in PAGES.items()]
    shots += [("chips", "Evidence", "Every chip measured", "One page per chip, from public result files"),
              ("findings", "Findings", "What the tests have turned up", "Each with its evidence")]
    shots += [(f"findings-{sl}", "Finding", t, nice_date(d)) for sl, d, t, _ in FINDINGS]
    shots += [(f"chips-{chip_slug(d)}", "Chip result", d, "Silent computation error test results") for d in devices]
    drawn = all(make_og(*shot) for shot in shots)
    if not drawn:
        print("Pillow is not installed, so the share images in docs/og were left as they are")
    with open(os.path.join(ROOT, "docs", "sitemap.xml"), "w", encoding="utf-8") as f:
        f.write(sitemap(devices, built))
    llms_path = os.path.join(ROOT, "docs", "llms.txt")
    with open(llms_path, encoding="utf-8") as f:
        llms = f.read()
    llms = re.sub(r"## Measured so far.*?(?=\n## |\Z)", lambda m: llms_measured(chip_map), llms, flags=re.S)
    with open(llms_path, "w", encoding="utf-8") as f:
        f.write(llms)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(chips)} chips, {total_runs:,} compute runs, "
          f"{total_words:,} memory words, {total_caught}/{total_injected} injected faults caught; "
          f"{len(PAGES) + 1} pages and {len(devices)} chip pages")


if __name__ == "__main__":
    main()
