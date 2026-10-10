"""The ``chip-integrity`` command.

    chip-integrity run                  every probe and its self-test, ten OCP result files
    chip-integrity run --quick          a smoke test with small sizes, about a minute on a laptop
    chip-integrity screen [options]     one probe with its own options (also memcheck, kernels, abft, patterns)
    chip-integrity --version            the suite version and every probe's version

``run`` starts each probe in its own Python process, exactly as running the ten commands in the README
one after another would, so the files are the same and one probe's memory never limits the next.
"""

import argparse
import importlib
import re
import subprocess
import sys
import time
from pathlib import Path

from chip_integrity import __version__

HERE = Path(__file__).resolve().parent
REPO = "https://github.com/tech4biz-yasha/ai-chip-integrity"

# Probe module, what it checks, flips per step for its self-test, and its two result files.
# Same order and file names as the ten commands in the README; the memory sweep runs last
# because it takes the longest and fills most of the device memory.
PROBES = [
    ("screen", "compute", 5, "{name}_clean.jsonl", "{name}_inject.jsonl"),
    ("kernels", "transformer kernels", 3, "{name}_kernels.jsonl", "{name}_kernels_inject.jsonl"),
    ("abft", "checksum matrix multiply", 3, "{name}_abft.jsonl", "{name}_abft_inject.jsonl"),
    ("patterns", "data patterns", 3, "{name}_patterns.jsonl", "{name}_patterns_inject.jsonl"),
    ("memcheck", "memory", 5, "{name}_memcheck.jsonl", "{name}_memcheck_inject.jsonl"),
]
PROBE_NAMES = [p[0] for p in PROBES]
VERSIONED = PROBE_NAMES + ["counters"]

# Small sizes for --quick. They exercise every code path, but the files are a smoke test, not a row.
QUICK = {
    "screen": ["--size", "64", "--iters", "5"],
    "kernels": ["--size", "256", "--iters", "5"],
    "abft": ["--shapes", "64x64x64", "--iters", "5"],
    "patterns": ["--shapes", "64x64x64", "--iters", "5"],
    "memcheck": ["--gb", "0.0625", "--chunk-mb", "16", "--dwell", "0.1"],
}


def probe_versions():
    """Each probe's VERSION, read from its source so that --version needs no torch import."""
    versions = {}
    for module in VERSIONED:
        match = re.search(r'^VERSION = "([\d.]+)"', (HERE / f"{module}.py").read_text(), re.M)
        versions[module] = match.group(1) if match else "unknown"
    return versions


def version_line():
    probes = ", ".join(f"{m} {v}" for m, v in probe_versions().items())
    return f"ai-chip-integrity {__version__} ({probes})"


def build_parser():
    p = argparse.ArgumentParser(
        prog="chip-integrity",
        formatter_class=argparse.RawDescriptionHelpFormatter,   # keeps the version line on one line
        description="Open tests for silent computation errors in AI chips.\n"
                    "Use `chip-integrity run` for every probe, or name one probe to run it with its own options.",
        epilog="Probes: " + ", ".join(PROBE_NAMES) + ".\nExample: chip-integrity screen --help",
    )
    p.add_argument("--version", action="version", version=version_line(), help="print the suite and probe versions")
    sub = p.add_subparsers(dest="command", metavar="command")
    run = sub.add_parser("run", help="every probe and its self-test, ten OCP result files",
                         description="Run every probe and its self-test, writing ten OCP result files.")
    run.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    run.add_argument("--name", default="chip", help="prefix for the result file names")
    run.add_argument("--out-dir", default=None, help="folder for the result files (default chip-integrity-results, or chip-integrity-quick with --quick)")
    run.add_argument("--quick", action="store_true", help="small sizes: a smoke test, not a publishable row")
    run.add_argument("--skip", default="", help="comma list of probes to leave out, from " + ",".join(PROBE_NAMES))
    for name in PROBE_NAMES:
        sub.add_parser(name, help=f"run the {name} probe with its own options", add_help=False)
    return p


def plan(name, out_dir, device, quick, skip):
    """The commands `run` executes, as (probe, label, argv, output file)."""
    steps = []
    for module, what, flips, clean_file, inject_file in PROBES:
        if module in skip:
            continue
        base = [sys.executable, "-m", f"chip_integrity.{module}", "--device", device] + (QUICK[module] if quick else [])
        clean = out_dir / clean_file.format(name=name)
        inject = out_dir / inject_file.format(name=name)
        steps.append((module, f"{what}", base + ["--out", str(clean)], clean))
        steps.append((module, f"{what} self-test", base + ["--inject", str(flips), "--out", str(inject)], inject))
    return steps


def run(args, runner=subprocess.call):
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.name):
        print("--name may only use letters, digits, '.', '_' and '-'", file=sys.stderr)
        return 2
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    unknown = skip - set(PROBE_NAMES)
    if unknown:
        print(f"--skip: unknown probe(s) {', '.join(sorted(unknown))}; choose from {', '.join(PROBE_NAMES)}", file=sys.stderr)
        return 2
    out_dir = Path(args.out_dir or ("chip-integrity-quick" if args.quick else "chip-integrity-results"))
    out_dir.mkdir(parents=True, exist_ok=True)
    steps = plan(args.name, out_dir, args.device, args.quick, skip)
    if not steps:
        print("nothing to run: every probe was skipped", file=sys.stderr)
        return 2

    print(f"{version_line()}\nRunning {len(steps)} steps on device '{args.device}', writing to {out_dir}/"
          + ("\nQuick mode: small sizes, a smoke test rather than a publishable row." if args.quick else ""), flush=True)
    results = []
    for i, (module, label, argv, out_file) in enumerate(steps, 1):
        print(f"\n[{i}/{len(steps)}] {label}: {' '.join(argv[1:])}", flush=True)
        start = time.time()
        code = runner(argv)
        results.append((label, code, time.time() - start, out_file))
        print(f"[{i}/{len(steps)}] {label}: {'PASS' if code == 0 else f'FAIL (exit code {code})'} in {time.time() - start:.1f} s", flush=True)

    width = max(len(r[0]) for r in results)
    print("\nSummary")
    for label, code, seconds, out_file in results:
        print(f"  {label:<{width}}  {'PASS' if code == 0 else 'FAIL'}  {seconds:7.1f} s  {out_file}")
    failed = [r for r in results if r[1] != 0]
    if failed:
        print(f"\n{len(failed)} of {len(results)} steps failed. The result files above say which checks fired and where.")
        return 1
    print(f"\nAll {len(results)} steps passed.")
    if not args.quick and not skip:
        print(f"To add this chip to the public table, open a pull request with the ten files, the device name, "
              f"driver version and PyTorch version: {REPO}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] in PROBE_NAMES:
        # One probe, with its own options, run in this process exactly as `python <probe>.py` would.
        module = importlib.import_module(f"chip_integrity.{argv[0]}")
        sys.argv = [f"chip-integrity {argv[0]}"] + argv[1:]   # so the probe's usage line names the subcommand
        return module.main(argv[1:])
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return run(args)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
