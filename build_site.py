#!/usr/bin/env python3
"""
build_site.py  -  generates docs/index.html for chipintegrity.org from results/*.jsonl

Every number on the page is read from the OCP result files in results/. Run it
after adding result files, then commit docs/.

    python build_site.py
"""

import glob
import json
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
    hw = start["dutInfo"]["hardwareInfos"][0]["name"]
    sw = {s["name"]: s["version"] for s in start["dutInfo"].get("softwareInfos", [])}
    return {
        "file": os.path.relpath(path, ROOT), "run": start["name"], "version": start["version"],
        "params": start.get("parameters", {}), "result": end["result"], "date": ts[:10],
        "device": hw, "torch": sw.get("torch", ""), "steps": {names.get(k, k): v for k, v in steps.items()},
    }


def summarise(runs):
    """Group runs by device and probe into table rows."""
    compute, memory = {}, {}
    for r in runs:
        inject = int(r["params"].get("inject", 0)) > 0
        key = r["device"]
        if r["run"] == "ai-chip-integrity-screen":
            row = compute.setdefault(key, {"device": key, "torch": r["torch"], "dates": set(), "version": r["version"]})
            row["dates"].add(r["date"])
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
                                           key=["fp32", "fp16", "bf16", "int8"].index)
                row["shapes"] = r["params"].get("shapes", r["params"].get("size", ""))
                row["ref_fail"] = sum(s.get("reference_check_failed_runs", 0) for s in live)
                row["rep_fail"] = sum(s.get("repeat_check_failed_runs", 0) for s in live)
                row["verdicts"] = sorted({s["_verdict"] for s in live})
                row["clean_file"] = r["file"]
        elif r["run"] == "ai-chip-integrity-memcheck":
            row = memory.setdefault(key, {"device": key, "dates": set(), "version": r["version"]})
            row["dates"].add(r["date"])
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
    order = lambda d: (0 if "H100" in d else 1 if "A100" in d else 2 if "NVIDIA" in d else 3, d)
    return ([compute[k] for k in sorted(compute, key=order)], [memory[k] for k in sorted(memory, key=order)])


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
        ok = r["verdicts"] == ["no-silent-errors"]
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        prec = ", ".join(r["precisions"]) + (f' ({r["skipped"]} skipped)' if r["skipped"] else "")
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">PyTorch {esc(r['torch'])} · tool v{esc(r['version'])}</span></td>
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


def memory_rows(rows):
    out = []
    for r in rows:
        if "bytes" not in r:
            continue
        ok = r["verdict"] == "no-memory-errors"
        caught = f'{r.get("caught", 0)} of {r.get("injected", 0)}' if "injected" in r else "not run"
        total = f' of {gib(r["total"])}' if r.get("total") else ""
        files = f'<a href="{REPO}/blob/main/{r["clean_file"]}">clean</a>'
        if "inject_file" in r:
            files += f' · <a href="{REPO}/blob/main/{r["inject_file"]}">self-test</a>'
        out.append(f"""<tr>
<td>{esc(r['device'])}<span class="sub">tool v{esc(r['version'])}</span></td>
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
    compute, memory = summarise(runs)
    chips = sorted({r["device"] for r in runs})
    total_runs = sum(r.get("runs", 0) for r in compute)
    total_caught = sum(r.get("caught", 0) for r in compute) + sum(r.get("caught", 0) for r in memory)
    total_injected = sum(r.get("injected", 0) for r in compute) + sum(r.get("injected", 0) for r in memory)
    total_words = sum(r.get("bytes", 0) for r in memory) // 4
    faults = sum(r.get("ref_fail", 0) + r.get("rep_fail", 0) for r in compute) + sum(r.get("bad", 0) for r in memory)
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
                .replace("{{BUILT}}", built))
    assert not re.search(r"{{[A-Z_]+}}", html), "unfilled placeholder"
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"wrote {os.path.relpath(OUT, ROOT)}: {len(chips)} chips, {total_runs:,} compute runs, "
          f"{total_words:,} memory words, {total_caught}/{total_injected} injected faults caught")


if __name__ == "__main__":
    main()
