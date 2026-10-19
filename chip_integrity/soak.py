#!/usr/bin/env python3
"""
soak.py  -  AI Chip Integrity Suite, hours under load (probe 7) v0.1.0

Keeps the chip busy for as long as you ask and checks every result. One fixed matrix multiply runs back to
back, one precision at a time in rotation, and every result is compared bit for bit with the first one, which
was itself checked against the proven rounding bound (or, for int8 and FP8, against the exact answer) before
the soak began. With TF32 off and deterministic kernels the same multiply must return the same bits every
time, so a result that differs is a fault: the intermittent kind that may only show up once a chip has run hot
for a while, lowered its clocks or reached its power limit.

The soak is cut into windows (one minute by default), one precision per window, and each window becomes one
OCP step with the chip's state beside the verdict: temperature, SM and memory clock, power, and the reasons the
driver gives for running below full clock (power cap, thermal slowdown and so on), read through NVML on NVIDIA
GPUs. A result that differs is also checked against the bound, and the temperature at that moment is recorded,
so every failure carries the conditions it happened in.

  no-silent-errors          every run in the window was bit-identical to the verified first result
  silent-data-corruption    runs differed from the verified first result; for int8 and FP8, whose answer is
                            exact, any difference is a wrong answer
  nondeterministic-kernel   many runs differed but every one checked sat inside the bound, so this kernel does
                            not repeat itself exactly on this device and the repeat check cannot judge it

Before the soak each precision gets a setup step: the first result is checked against the bound and repeated
a few times to make sure the kernel gives the same bits. A precision that fails its setup is left out of the
soak and fails the run.

--inject N plants a fault in N runs of every window, with the screener's fault models (one random bit, a fault
shaped like NVIDIA's gate-level measurements, or both alternating), to prove the check catches corruption while
the chip is under load. Faults are planted on a copy taken back from the device, so they test the check, not
the chip.

Output: OCP Test & Validation JSON, one object per line, written to --out.
"""

import argparse
import os
import random
import socket
import sys
import time

# must be set before CUDA initialises so cuBLAS runs deterministically (as in screen.py)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import ocptv.output as tv  # noqa: E402

from chip_integrity import arith  # noqa: E402
from chip_integrity import counters  # noqa: E402
from chip_integrity import screen  # noqa: E402

VERSION = "0.1.0"

# K stays at 4096 on CUDA so FP8 keeps its exact answer (screen.FP8_EXACT_K); M and N are large enough that each
# multiply takes several times longer than the bit-for-bit comparison that follows it.
DEFAULT_SHAPE = {"cuda": "8192x4096x8192", "mps": "1024x1024x1024", "cpu": "256x256x256"}
DEFAULT_DTYPES = {"cuda": "bf16,fp16,fp32,int8,fp8", "mps": "fp16,fp32", "cpu": "fp32"}
WARMUP_RUNS = 3          # repeats of the first result in each precision's setup step
MAX_BOUND_CHECKS = 3     # differing runs per window that are also checked against the bound
# Why the clock is below its maximum, as NVML reports it (nvmlClocksEventReasons; older drivers call them
# throttle reasons). The bit values are fixed by the NVML API.
CLOCK_EVENT_REASONS = ((0x1, "gpu-idle"), (0x2, "applications-clocks-setting"), (0x4, "sw-power-cap"),
                       (0x8, "hw-slowdown"), (0x10, "sync-boost"), (0x20, "sw-thermal-slowdown"),
                       (0x40, "hw-thermal-slowdown"), (0x80, "hw-power-brake-slowdown"),
                       (0x100, "display-clock-setting"))
TELEMETRY_KEYS = (("temperature_c", "C"), ("sm_clock_mhz", "MHz"), ("memory_clock_mhz", "MHz"), ("power_w", "W"))
_FAULT_HOOK = None       # tests only: callable(window, run, result) that corrupts a device result in place


def reason_names(mask):
    """The clock-event reason bits as names; bits this version does not know are kept as other-0x...."""
    names = [name for bit, name in CLOCK_EVENT_REASONS if mask & bit]
    unknown = mask & ~sum(bit for bit, _ in CLOCK_EVENT_REASONS)
    if unknown:
        names.append(f"other-0x{unknown:x}")
    return names


class Telemetry:
    """Temperature, clocks, power and clock-event reasons through NVML. Empty on devices without it."""

    def __init__(self, dev):
        self.nv = self.handle = None
        if getattr(dev, "type", None) == "cuda":
            nv = counters._nvml()
            if nv is not None:
                try:
                    self.handle, self.nv = counters._handle(nv, dev), nv
                except Exception:
                    self.handle = self.nv = None

    @property
    def available(self):
        return self.handle is not None

    def _call(self, fn_name, *const_names):
        """One NVML reading, or None when the call or a constant it needs is missing or fails."""
        fn = getattr(self.nv, fn_name, None)
        consts = [getattr(self.nv, c, None) for c in const_names]
        if fn is None or any(c is None for c in consts):
            return None
        return counters._try(fn, self.handle, *consts)

    def read(self):
        if not self.available:
            return {}
        s = {"temperature_c": self._call("nvmlDeviceGetTemperature", "NVML_TEMPERATURE_GPU"),
             "sm_clock_mhz": self._call("nvmlDeviceGetClockInfo", "NVML_CLOCK_SM"),
             "memory_clock_mhz": self._call("nvmlDeviceGetClockInfo", "NVML_CLOCK_MEM")}
        mw = self._call("nvmlDeviceGetPowerUsage")
        s["power_w"] = round(mw / 1000.0, 1) if mw is not None else None
        reasons = self._call("nvmlDeviceGetCurrentClocksEventReasons")
        if reasons is None:
            reasons = self._call("nvmlDeviceGetCurrentClocksThrottleReasons")
        s["clock_event_reasons"] = int(reasons) if reasons is not None else None
        return {k: v for k, v in s.items() if v is not None}


class Case:
    """One precision's inputs on the device, its verified first result, and what is needed to judge a result
    that differs from it."""

    def __init__(self, name, a, b, fast, first, int_view, bits, exact, ref=None, bound=None):
        self.name, self.a, self.b, self.fast = name, a, b, fast
        self.first, self.first_bits = first, first.view(int_view)
        self.int_view, self.bits, self.exact = int_view, bits, exact
        self.ref, self.bound = ref, bound

    def run(self):
        return screen.matmul(self.a, self.b, self.fast)

    def same(self, c):
        """Bit for bit, so a sign flip on a zero or a changed NaN is caught too."""
        return torch.equal(c.view(self.int_view), self.first_bits)

    def outside_bound(self, c):
        """True when a result sits outside the proven bound, or for int8 and FP8 is not the exact answer."""
        if self.exact:
            return not torch.equal(c, self.first)   # the first result equals the exact answer
        c64 = c.to("cpu").to(torch.float64)
        if not bool(torch.isfinite(c64).all()):
            return True
        return bool(((c64 - self.ref).abs() > self.bound).any())


def setup_case(step, dev, name, shape, seed, hw):
    """Prepare one precision: inputs on the device, the first result checked against the bound, and a few
    repeats to make sure the kernel gives the same bits. Returns (case or None, ok, verdict, message, seconds)."""
    dtype, out_dtype, int_view, bits, u_out = screen.DTYPES[name]
    m, k, n = shape
    if dtype == screen.FP8 and k > screen.FP8_EXACT_K:
        raise RuntimeError(f"FP8 exact check needs K <= {screen.FP8_EXACT_K}; got {k}")
    t0 = time.time()
    a, b, ref, abs_ab = screen.make_case(m, k, n, dtype, seed)
    exact = u_out is None
    g = None if exact else screen.accumulation_factor(name, k, dev.type)
    bound = screen.error_bound(ref, abs_ab, k, u_out, g=g,
                               flush_ab=None if exact else screen.flush_products(a, b),
                               out_floor=0.0 if exact else arith.subnormal_floor(out_dtype))
    del abs_ab
    a_dev, b_dev = a.to(dev), b.to(dev)
    if dtype == screen.FP8:
        b_dev = b_dev.t().contiguous().t()                 # the FP8 kernels take B column-major
    first = screen.matmul(a_dev, b_dev, name == "fp8fast").contiguous()
    if first.dtype != out_dtype:
        raise RuntimeError(f"device returned {first.dtype}, expected {out_dtype}")

    c64 = first.to("cpu").to(torch.float64)
    finite = torch.isfinite(c64)
    err = (c64 - ref).abs()
    outside = int(not bool(finite.all()) or bool((err[finite] > bound[finite]).any()))
    worst_ratio = 0.0
    if not exact and bool(finite.any()):
        worst_ratio = float((err[finite] / bound[finite]).max())
    case = Case(name, a_dev, b_dev, name == "fp8fast", first, int_view, bits, exact,
                ref=None if exact else ref, bound=None if exact else bound)
    differing = [c for c in (case.run() for _ in range(WARMUP_RUNS)) if not case.same(c)]
    differing_outside = sum(case.outside_bound(c) for c in differing)
    secs = time.time() - t0

    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    step.add_measurement(name="precision", value=name, hardware_info=hw)
    step.add_measurement(name="shape_mkn", value=f"{m}x{k}x{n}", hardware_info=hw)
    step.add_measurement(name="exact_check", value=exact, hardware_info=hw)
    if not exact:
        step.add_measurement(name="accumulation_factor", value=g, hardware_info=hw)
        step.add_measurement(name="worst_error_over_bound_ratio", value=worst_ratio, hardware_info=hw)
    step.add_measurement(name="first_result_outside_bound", value=outside, validators=eq0, hardware_info=hw)
    step.add_measurement(name="repeat_runs", value=WARMUP_RUNS, hardware_info=hw)
    step.add_measurement(name="repeat_runs_differing", value=len(differing), validators=eq0, hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)

    what = "the exact answer" if exact else "inside the bound"
    if outside:
        ok, verdict = False, "outside-error-bound"
        msg = (f"the first result is not {what}"
               + ("" if exact else f" (worst error {worst_ratio:.3g} times the bound)")
               + ", so it cannot serve as the soak reference; this precision is left out")
    elif differing and differing_outside:
        ok, verdict = False, "silent-data-corruption"
        msg = (f"{len(differing)} of {WARMUP_RUNS} repeats differ from the first result and {differing_outside} "
               f"of them are wrong; this precision is left out")
    elif differing:
        ok, verdict = False, "nondeterministic-kernel"
        msg = (f"{len(differing)} of {WARMUP_RUNS} repeats differ from the first result but stay inside the "
               f"bound, so the bit-for-bit check cannot judge this kernel; this precision is left out")
    else:
        ok, verdict = True, "first-result-verified"
        msg = f"first result {what}, {WARMUP_RUNS} repeats bit-identical"
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                       hardware_info=hw)
    return (case if ok else None), ok, verdict, msg, secs


def soak_window(step, dev, case, seconds, sample_s, inject, inject_model, rng, hw, telemetry, window):
    """Run one precision back to back for `seconds` and judge every run. Returns a dict that summarises the
    window, with 'interrupted' set when Ctrl-C ended it early."""
    readings = {key: [] for key, _ in TELEMETRY_KEYS}
    reasons = 0

    def sample():
        nonlocal reasons
        snap = telemetry.read()
        for key in readings:
            if key in snap:
                readings[key].append(snap[key])
        reasons |= snap.get("clock_event_reasons", 0)
        return snap

    inject_runs = set(rng.sample(range(2 * inject), inject)) if inject else set()
    models = screen.fault_models(inject_runs, inject_model)
    min_runs = max(1, 2 * inject)          # every planned fault is planted even in a short window
    runs, differing, checked, outside = 0, [], 0, 0
    injected, caught, sdc_planted, sdc_caught = 0, 0, 0, 0
    first_fail_temp, interrupted = None, False
    t0 = time.monotonic()
    deadline, next_sample = t0 + seconds, t0
    try:
        while runs < min_runs or time.monotonic() < deadline:
            if time.monotonic() >= next_sample:
                sample()
                next_sample = time.monotonic() + sample_s
            c = case.run()
            if _FAULT_HOOK is not None:
                _FAULT_HOOK(window, runs, c)
            planted = runs in inject_runs
            if planted:
                host = c.to("cpu").contiguous()
                screen.plant_fault(host, case.int_view, case.bits, rng, models[runs])
                c = host.to(dev)
                injected += 1
                sdc_planted += models[runs] == "gpu-sdc"
            if not case.same(c):
                if planted:
                    caught += 1
                    sdc_caught += models[runs] == "gpu-sdc"
                else:
                    differing.append(runs)
                    if first_fail_temp is None:
                        first_fail_temp = sample().get("temperature_c")
                    if checked < MAX_BOUND_CHECKS:
                        checked += 1
                        outside += case.outside_bound(c)
            runs += 1
    except KeyboardInterrupt:
        interrupted = True
    sample()
    secs = time.monotonic() - t0

    if inject:
        ok = injected > 0 and caught == injected and not differing
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg = f"{caught}/{injected} injected runs detected under load; {len(differing)} unexpected differing runs"
    elif not differing:
        ok, verdict = True, "no-silent-errors"
        msg = f"{runs} runs in {secs:.1f} s, all bit-identical to the verified first result"
    elif (not case.exact and checked and not outside
          and len(differing) >= max(screen.NONDET_MIN, screen.NONDET_SHARE * runs)):
        ok, verdict = False, "nondeterministic-kernel"
        msg = (f"{len(differing)}/{runs} runs differ but all {checked} checked stay inside the bound; "
               f"the bit-for-bit check cannot judge this kernel")
    else:
        ok, verdict = False, "silent-data-corruption"
        msg = (f"{len(differing)}/{runs} runs differ from the verified first result (first at run {differing[0]}); "
               f"{outside}/{checked} checked sit outside the bound")
        if first_fail_temp is not None:
            msg += f"; {first_fail_temp} C when the first was seen"
    if interrupted:
        msg += "; stopped early by Ctrl-C"

    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    step.add_measurement(name="precision", value=case.name, hardware_info=hw)
    step.add_measurement(name="runs", value=runs, hardware_info=hw)
    step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    step.add_measurement(name="differing_runs", value=len(differing), validators=eq0, hardware_info=hw)
    step.add_measurement(name="differing_runs_checked_against_bound", value=checked, hardware_info=hw)
    step.add_measurement(name="differing_runs_outside_bound", value=outside, hardware_info=hw)
    step.add_measurement(name="telemetry_available", value=telemetry.available, hardware_info=hw)
    for key, unit in TELEMETRY_KEYS:
        if readings[key]:
            step.add_measurement(name=f"{key}_min", value=min(readings[key]), unit=unit, hardware_info=hw)
            step.add_measurement(name=f"{key}_max", value=max(readings[key]), unit=unit, hardware_info=hw)
    if telemetry.available:
        step.add_measurement(name="clock_event_reasons", value=",".join(reason_names(reasons)) or "none",
                             hardware_info=hw)
    if first_fail_temp is not None:
        step.add_measurement(name="first_differing_run_temperature_c", value=first_fail_temp, unit="C",
                             hardware_info=hw)
    if inject:
        step.add_measurement(name="injected_runs", value=injected, hardware_info=hw)
        step.add_measurement(name="injected_runs_detected", value=caught,
                             validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=injected)],
                             hardware_info=hw)
        step.add_measurement(name="gpu_sdc_faults_planted", value=sdc_planted, hardware_info=hw)
        step.add_measurement(name="gpu_sdc_faults_detected", value=sdc_caught,
                             validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=sdc_planted)],
                             hardware_info=hw)
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                       hardware_info=hw)
    return {"ok": ok, "verdict": verdict, "msg": msg, "runs": runs, "differing": len(differing),
            "seconds": secs, "readings": readings, "reasons": reasons, "interrupted": interrupted}


def summarise(run, hw, windows, names, minutes, inject, interrupted, telemetry_on):
    """The soak_summary step: totals over every window, and the hottest, slowest and most power-hungry moments."""
    step = run.add_step("soak_summary")
    step.start()
    total_runs = sum(w["runs"] for w in windows)
    total_diff = sum(w["differing"] for w in windows)
    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    step.add_measurement(name="precisions", value=",".join(names) or "none", hardware_info=hw)
    step.add_measurement(name="minutes_planned", value=minutes, unit="min", hardware_info=hw)
    step.add_measurement(name="minutes_run", value=round(sum(w["seconds"] for w in windows) / 60.0, 3),
                         unit="min", hardware_info=hw)
    step.add_measurement(name="windows", value=len(windows), hardware_info=hw)
    step.add_measurement(name="runs", value=total_runs, hardware_info=hw)
    step.add_measurement(name="differing_runs", value=total_diff, validators=eq0, hardware_info=hw)
    step.add_measurement(name="interrupted", value=interrupted, hardware_info=hw)
    pooled = {key: [v for w in windows for v in w["readings"][key]] for key, _ in TELEMETRY_KEYS}
    for key, unit in TELEMETRY_KEYS:
        if pooled[key]:
            step.add_measurement(name=f"{key}_min", value=min(pooled[key]), unit=unit, hardware_info=hw)
            step.add_measurement(name=f"{key}_max", value=max(pooled[key]), unit=unit, hardware_info=hw)
    if telemetry_on:
        seen = 0
        for w in windows:
            seen |= w["reasons"]
        step.add_measurement(name="clock_event_reasons", value=",".join(reason_names(seen)) or "none",
                             hardware_info=hw)
    failed = [w for w in windows if not w["ok"]]
    ok = bool(windows) and not failed
    if not windows:
        verdict, msg = "not-run", "no precision passed its setup, so nothing was soaked"
    elif inject:
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg = f"{len(windows) - len(failed)}/{len(windows)} windows caught every planted fault"
    elif ok:
        verdict = "no-silent-errors"
        msg = f"{total_runs} runs in {len(windows)} windows, every one bit-identical to its verified first result"
    else:
        verdict = failed[0]["verdict"]
        msg = f"{len(failed)}/{len(windows)} windows failed, {total_diff} differing runs in all"
    if interrupted:
        msg += "; stopped early by Ctrl-C"
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                       hardware_info=hw)
    step.end(status=tv.TestStatus.COMPLETE)
    return ok, verdict, msg


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite, hours under load (probe 7)")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--minutes", type=float, default=60, help="how long to keep the chip under load, after setup")
    p.add_argument("--window-s", type=float, default=60,
                   help="seconds per window; each window is one precision and one OCP step")
    p.add_argument("--sample-s", type=float, default=5, help="seconds between telemetry readings")
    p.add_argument("--shape", default=None,
                   help="one MxKxN (default 8192x4096x8192 on CUDA, 1024x1024x1024 on MPS, 256x256x256 on CPU)")
    p.add_argument("--dtypes", default=None,
                   help="comma list from fp32,fp16,bf16,int8,fp8,fp8fast, one per window in rotation "
                        "(default bf16,fp16,fp32,int8,fp8 on CUDA, fp16,fp32 on MPS, fp32 on CPU)")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="faults to inject per window (self-test)")
    p.add_argument("--inject-model", default="mixed", choices=screen.INJECT_MODELS,
                   help="self-test faults: one random bit, a gate-level-shaped fault, or both alternating")
    p.add_argument("--out", default="soak.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    if not args.minutes > 0 or not args.window_s >= 0.1 or not args.sample_s > 0 or args.inject < 0:
        p.error("need --minutes > 0, --window-s >= 0.1, --sample-s > 0 and --inject >= 0")
    dev = screen.pick_device(args.device)
    try:
        shapes = screen.parse_shapes(args.shape or DEFAULT_SHAPE.get(dev.type, DEFAULT_SHAPE["cpu"]))
    except ValueError as e:
        p.error(str(e))
    if len(shapes) != 1:
        p.error("--shape takes exactly one MxKxN")
    shape = shapes[0]
    names = [d.strip() for d in (args.dtypes or DEFAULT_DTYPES.get(dev.type, "fp32")).split(",") if d.strip()]
    bad = [d for d in names if d not in screen.DTYPES]
    if bad or not names:
        p.error(f"check --dtypes: unknown {bad}, choose from {','.join(screen.DTYPES)}")

    screen.lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = screen.FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(
        name="ai-chip-integrity-soak",
        version=VERSION,
        parameters={"device": str(dev), "shape": "x".join(map(str, shape)), "dtypes": ",".join(names),
                    "minutes": args.minutes, "window_s": args.window_s, "sample_s": args.sample_s,
                    "seed": args.seed, "inject": args.inject, "inject_model": args.inject_model,
                    "repeat_runs": WARMUP_RUNS, "bound_model": arith.MODEL_ID},
    )
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=screen.device_label(dev))
    screen.add_software_infos(dut, dev)
    telemetry = Telemetry(dev)

    label = "x".join(map(str, shape))
    print(f"Device: {screen.device_label(dev)} ({dev})  shape={label}  precisions={','.join(names)}  "
          f"minutes={args.minutes:g}  window={args.window_s:g} s  inject={args.inject}  "
          f"telemetry={'NVML' if telemetry.available else 'none'}")
    all_ok, interrupted, windows, soaked = True, False, [], []
    run.start(dut=dut)
    before = counters.snapshot(dev)
    try:
        try:
            cases = []
            for name in names:
                step = run.add_step(f"soak_setup_{name}")
                step.start()
                try:
                    case, ok, verdict, msg, secs = setup_case(step, dev, name, shape, args.seed, hw)
                    step.end(status=tv.TestStatus.COMPLETE)
                except (RuntimeError, TypeError, NotImplementedError) as e:
                    step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                    step.end(status=tv.TestStatus.SKIP)
                    print(f"  setup  {name:7s}  SKIPPED  {str(e)[:100]}")
                    continue
                all_ok &= ok
                print(f"  setup  {name:7s}  {'PASS' if ok else 'FAIL'}  {verdict}  ({secs:.1f}s)  {msg}")
                if case is not None:
                    cases.append(case)
                    soaked.append(name)

            deadline = time.monotonic() + args.minutes * 60
            w = 0
            while cases and not interrupted and time.monotonic() < deadline:
                case = cases[w % len(cases)]
                step = run.add_step(f"soak_window_{w + 1:04d}_{case.name}")
                step.start()
                res = soak_window(step, dev, case, min(args.window_s, deadline - time.monotonic()), args.sample_s,
                                  args.inject, args.inject_model, rng, hw, telemetry, w)
                step.end(status=tv.TestStatus.COMPLETE)
                windows.append(res)
                interrupted = res["interrupted"]
                r = res["readings"]
                state = ""
                if r["temperature_c"]:
                    state += f"  {min(r['temperature_c'])}-{max(r['temperature_c'])} C"
                if r["sm_clock_mhz"]:
                    state += f"  {min(r['sm_clock_mhz'])}-{max(r['sm_clock_mhz'])} MHz"
                if r["power_w"]:
                    state += f"  up to {max(r['power_w'])} W"
                if res["reasons"]:
                    state += "  " + ",".join(reason_names(res["reasons"]))
                print(f"  window {w + 1:4d} {case.name:7s}  {'PASS' if res['ok'] else 'FAIL'}  {res['verdict']}  "
                      f"runs={res['runs']}{state}")
                w += 1
        except KeyboardInterrupt:              # Ctrl-C during setup or between windows: keep what was measured
            interrupted = True
        ok, verdict, msg = summarise(run, hw, windows, soaked, args.minutes, args.inject, interrupted,
                                     telemetry.available)
        all_ok &= ok
        print(f"  {'summary':>12s}  {'PASS' if ok else 'FAIL'}  {verdict}  {msg}")
    finally:
        try:
            all_ok &= counters.record(run, hw, before, counters.snapshot(dev), probe_ok=all_ok, injected=args.inject > 0)
        except Exception as e:                       # the cross-check must never cost the run its result file
            print(f"  error counters could not be recorded: {e}")
        run.end(status=tv.TestStatus.COMPLETE,
                result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()

    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
