#!/usr/bin/env python3
"""
memcheck.py  -  AI Chip Integrity Suite, memory sweep v0.1.0

Fills as much of the device memory as it can, writes known patterns into every
32-bit word, waits a dwell time, reads everything back and counts every word
that does not match. A matrix multiply touches a few megabytes; this touches
most of the device memory, so a bad row, a stuck bit or a weak cell there shows up.

Patterns, in order: all zeros, all ones, 0xAAAAAAAA, 0x55555555, an
address-hash (every word different, pseudo-random from its own address) and
the bitwise inverse of that hash. The expected value is recomputed on the
device at read time, so the comparison runs at device speed; the first ten
mismatching addresses and the XOR of the bad bits are recorded.

--inject N flips one random bit in N random words of device memory after each
pattern is written, to prove the sweep catches it.

Output: OCP Test & Validation JSON (one object per line) to --out, plus a
short summary on the terminal.
"""

import argparse
import os
import platform
import random
import socket
import sys
import time

import torch
import ocptv.output as tv

from screen import FileWriter, pick_device, device_label

VERSION = "0.1.0"
WORD = 4
MAX_RECORD = 1000
PATTERNS = ["zeros", "ones", "aaaa", "5555", "hash", "hash_inv"]
SOLID = {"zeros": 0x00000000, "ones": 0xFFFFFFFF, "aaaa": 0xAAAAAAAA, "5555": 0x55555555}
_FAULT_HOOK = None   # tests only: callable(pattern, chunk_tensor, word_offset) that corrupts device memory


def to_i32(x):
    """Reinterpret unsigned 32-bit values held in an int64 tensor as int32."""
    return ((x + 2 ** 31) % 2 ** 32 - 2 ** 31).to(torch.int32)


def expected(pattern, start_word, n, dev, seed):
    """Pattern value for words [start_word, start_word + n) on the device."""
    if pattern in SOLID:
        return torch.full((n,), int(to_i32(torch.tensor([SOLID[pattern]], dtype=torch.int64))[0]),
                          dtype=torch.int32, device=dev)
    idx = torch.arange(start_word, start_word + n, dtype=torch.int64, device=dev)
    # LCG-style mixing with multipliers below 2^21, so every product stays inside int64
    h = ((idx % 2 ** 32) * 1664525 + (idx >> 32) * 22695477 + 1013904223 + seed * 40503) % 2 ** 32
    h = h ^ (h >> 13)
    h = (h * 1664525 + 1013904223) % 2 ** 32
    h = h ^ (h >> 16)
    if pattern == "hash_inv":
        h = (2 ** 32 - 1) - h
    return to_i32(h)


def memory_budget_bytes(dev, gb_arg, fraction):
    if gb_arg:
        return int(gb_arg * 2 ** 30)
    if dev.type == "cuda":
        free, total = torch.cuda.mem_get_info(dev)
        return int(free * fraction)
    if dev.type == "mps":
        return int(torch.mps.recommended_max_memory() * fraction * 0.6)   # unified memory, leave room for macOS
    return 2 * 2 ** 30


def total_device_bytes(dev):
    if dev.type == "cuda":
        return torch.cuda.mem_get_info(dev)[1]
    if dev.type == "mps":
        return torch.mps.recommended_max_memory()
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError, AttributeError):
        return 0


def allocate(dev, budget, chunk_bytes):
    """Grab the budget in chunks; stop quietly at the first allocation failure."""
    chunks = []
    got = 0
    while got + chunk_bytes <= budget:
        try:
            chunks.append(torch.empty(chunk_bytes // WORD, dtype=torch.int32, device=dev))
            got += chunk_bytes
        except (RuntimeError, MemoryError):
            break
    return chunks, got


def sweep(step, dev, chunks, dwell, seed, inject, rng, hw):
    n_chunk = chunks[0].numel()
    total_words = n_chunk * len(chunks)
    report = {}
    all_ok = True

    for pattern in PATTERNS:
        for ci, buf in enumerate(chunks):
            buf.copy_(expected(pattern, ci * n_chunk, n_chunk, dev, seed))
        injected = []
        for _ in range(inject):
            ci = rng.randrange(len(chunks))
            w = rng.randrange(n_chunk)
            bit = rng.randrange(32)
            mask = torch.tensor(1 << bit if bit < 31 else -(1 << 31), dtype=torch.int32, device=dev)
            chunks[ci][w] = chunks[ci][w] ^ mask
            injected.append((ci * n_chunk + w) * WORD)
        if _FAULT_HOOK is not None:
            for ci, buf in enumerate(chunks):
                _FAULT_HOOK(pattern, buf, ci * n_chunk)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        elif dev.type == "mps":
            torch.mps.synchronize()
        time.sleep(dwell)

        bad_words = 0
        bad_offsets = []          # byte offsets of bad words, first MAX_RECORD only
        xor_bits = 0              # OR of (read XOR expected) over the recorded bad words
        for ci, buf in enumerate(chunks):
            exp = expected(pattern, ci * n_chunk, n_chunk, dev, seed)
            bad = buf != exp
            count = int(bad.sum())
            if count:
                bad_words += count
                idx = torch.nonzero(bad).flatten()[:MAX_RECORD]
                bad_offsets += [int((ci * n_chunk + i) * WORD) for i in idx.tolist()]
                diff = (buf[idx].to(torch.int64) & 0xFFFFFFFF) ^ (exp[idx].to(torch.int64) & 0xFFFFFFFF)
                for v in diff[:64].tolist():
                    xor_bits |= v
        bad_set = set(bad_offsets)
        caught = sum(1 for off in injected if off in bad_set) if inject else 0
        report[pattern] = (bad_words, bad_offsets[:10], xor_bits, len(injected), caught)

        v = None if inject else [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
        step.add_measurement(name=f"bad_words_{pattern}", value=bad_words, validators=v, hardware_info=hw)
        if bad_words:
            step.add_log(tv.LogSeverity.WARNING,
                         f"{pattern}: {bad_words} bad words, first byte offsets {bad_offsets[:10]}, bad bits 0x{xor_bits:08X}")
        if inject:
            step.add_measurement(name=f"injected_{pattern}", value=len(injected), hardware_info=hw)
            step.add_measurement(name=f"injected_detected_{pattern}", value=caught,
                                 validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=len(injected))],
                                 hardware_info=hw)
            ok = caught == len(injected) and bad_words == len(injected)
        else:
            ok = bad_words == 0
        all_ok &= ok

    return all_ok, report, total_words


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite memory sweep")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--gb", type=float, default=None, help="memory to test in GiB (default: a fraction of what is free)")
    p.add_argument("--fraction", type=float, default=0.8, help="fraction of free device memory to test when --gb is not given")
    p.add_argument("--chunk-mb", type=int, default=256, help="allocation chunk size in MiB")
    p.add_argument("--dwell", type=float, default=2.0, help="seconds to wait between write and read-back")
    p.add_argument("--passes", type=int, default=1, help="repeat the whole pattern set this many times")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="bit flips to inject per pattern (self-test, max 100)")
    p.add_argument("--out", default="memcheck.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)
    if (args.passes < 1 or args.chunk_mb < 1 or args.dwell < 0 or not 0 <= args.inject <= 100
            or (args.gb is not None and args.gb <= 0)):
        p.error("check --passes >= 1, --chunk-mb >= 1, --dwell >= 0, 0 <= --inject <= 100, --gb > 0")

    dev = pick_device(args.device)
    rng = random.Random(args.seed)
    chunk_bytes = args.chunk_mb * 2 ** 20
    budget = memory_budget_bytes(dev, args.gb, args.fraction)
    if budget < chunk_bytes:
        chunk_bytes = max(WORD * 1024, (budget // WORD) * WORD)
    chunks, got = allocate(dev, budget, chunk_bytes)
    if not chunks:
        print("could not allocate any device memory")
        return 2
    total = total_device_bytes(dev)

    writer = FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(name="ai-chip-integrity-memcheck", version=VERSION,
                     parameters={"device": str(dev), "bytes_tested": got, "chunk_mb": args.chunk_mb,
                                 "dwell_s": args.dwell, "passes": args.passes, "seed": args.seed,
                                 "inject": args.inject})
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=device_label(dev))
    dut.add_software_info(name="torch", version=torch.__version__)
    dut.add_software_info(name="python", version=platform.python_version())

    print(f"Device: {device_label(dev)} ({dev})  testing {got / 2**30:.2f} GiB"
          + (f" of {total / 2**30:.1f} GiB" if total else "") + f"  dwell={args.dwell}s  passes={args.passes}  inject={args.inject}")
    all_ok = True
    run.start(dut=dut)
    try:
        for pn in range(1, args.passes + 1):
            step = run.add_step(f"memory_sweep_pass{pn}")
            step.start()
            step.add_measurement(name="bytes_tested", value=got, unit="B", hardware_info=hw)
            if total:
                step.add_measurement(name="device_memory_bytes", value=total, unit="B", hardware_info=hw)
                step.add_measurement(name="coverage_fraction", value=round(got / total, 4), hardware_info=hw)
            step.add_measurement(name="dwell_seconds", value=args.dwell, unit="s", hardware_info=hw)
            t0 = time.time()
            ok, report, words = sweep(step, dev, chunks, args.dwell, args.seed + pn, args.inject, rng, hw)
            secs = time.time() - t0
            step.add_measurement(name="seconds", value=round(secs, 2), unit="s", hardware_info=hw)
            if args.inject:
                inj = sum(r[3] for r in report.values())
                det = sum(r[4] for r in report.values())
                verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
                msg = f"{det}/{inj} injected bit flips located; {sum(r[0] for r in report.values()) - det} unexpected bad words"
            else:
                bad = sum(r[0] for r in report.values())
                verdict = "no-memory-errors" if ok else "memory-errors"
                msg = (f"{words} words x {len(PATTERNS)} patterns, all read back correctly" if ok
                       else f"{bad} bad words across patterns " + ", ".join(k for k, r in report.items() if r[0]))
            step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL,
                               verdict=verdict, message=msg, hardware_info=hw)
            step.end(status=tv.TestStatus.COMPLETE)
            all_ok &= ok
            print(f"  pass {pn}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
            for k, (bw, offs, xb, inj, det) in report.items():
                if bw and not args.inject:
                    print(f"         {k:9s} {bw} bad words, first byte offsets {offs}, bad bits 0x{xb:08X}")
    finally:
        run.end(status=tv.TestStatus.COMPLETE,
                result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()
    del chunks
    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
