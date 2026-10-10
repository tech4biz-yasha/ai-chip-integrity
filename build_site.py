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
}
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
                row["counters"] = (r.get("counters") or {}).get("_verdict")
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
                row["ratio"] = max(s.get("worst_error_over_bound_ratio", 0) for s in live)
                row["ref_fail"] = sum(s.get("reference_check_failed_runs", 0) for s in live)
                row["rep_fail"] = sum(s.get("repeat_check_failed_runs", 0) for s in live)
                row["verdicts"] = sorted({s["_verdict"] for s in live})
                row["clean_file"] = r["file"]
                row["counters"] = (r.get("counters") or {}).get("_verdict")
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
                row["counters"] = (r.get("counters") or {}).get("_verdict")
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
                row["counters"] = (r.get("counters") or {}).get("_verdict")
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
                row["counters"] = (r.get("counters") or {}).get("_verdict")
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
<td>{esc(", ".join(r['kernels']))}<span class="sub">{kernel_sizes(r)}</span></td>
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

    with open(os.path.join(ROOT, "docs", "template.html"), encoding="utf-8") as f:
        html = f.read()
    html = (html.replace("{{CHIPS}}", str(len(chips)))
                .replace("{{RUNS}}", f"{total_runs:,}")
                .replace("{{WORDS}}", f"{total_words / 1e9:.1f} billion")
                .replace("{{FAULTS}}", str(faults))
                .replace("{{CAUGHT}}", f"{total_caught} of {total_injected}")
                .replace("{{COMPUTE_ROWS}}", compute_rows(compute))
                .replace("{{MEMORY_ROWS}}", memory_rows(memory))
                .replace("{{CHECKSUM_ROWS}}", checksum_rows(checksum) or '<tr><td colspan="11">No rows yet. The checksum probe is new; rows are added as the files come in.</td></tr>')
                .replace("{{PATTERN_ROWS}}", pattern_rows(pattern) or '<tr><td colspan="10">No rows yet. The data pattern probe is new; rows are added as the files come in.</td></tr>')
                .replace("{{KERNEL_ROWS}}", kernel_rows(kernel) or '<tr><td colspan="10">No rows yet. The kernel probe was added after the first four chips were measured; rows are added as the files come in.</td></tr>')
                .replace("{{BUILT}}", built))
    assert not re.search(r"{{[A-Z_]+}}", html), "unfilled placeholder"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(chips)} chips, {total_runs:,} compute runs, "
          f"{total_words:,} memory words, {total_caught}/{total_injected} injected faults caught")


if __name__ == "__main__":
    main()
