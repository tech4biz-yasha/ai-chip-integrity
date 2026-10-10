"""
counters.py  -  AI Chip Integrity Suite, error counter cross-check (probe 11) v0.1.0

Every probe reads the chip's own error counters at the start and the end of its run and records them in an
extra step, device_error_counters. Then it asks the question a silent error raises: did the hardware notice?

  counters-quiet             no counter moved and the probe found nothing wrong
  silent-error-confirmed     the probe found wrong answers and every counter stayed quiet: silent in the
                             strict sense, the case ECC and the error logs cannot see
  hardware-noticed           the probe found wrong answers and a counter moved too
  corrected-during-run       a corrected-error counter moved but every answer was right: the hardware caught
                             something, which on a healthy chip is worth watching
  uncorrected-during-run     an uncorrected-error counter moved: a hardware fault even if the answers held
  self-test-run              faults were planted in software, so no cross-check is drawn; counters still kept
  counters-unavailable       no NVIDIA management library, or a device without these counters (Apple, CPU)

On NVIDIA GPUs the counters come from NVML (the nvidia-ml-py package): volatile ECC counts since the driver
loaded, corrected and uncorrected, remapped memory rows (A100 and newer) and retired pages (older parts), plus
temperature, SM clock and power for context. Counters a device does not support are left out rather than
reported as zero.
"""

import importlib

import ocptv.output as tv

VERSION = "0.1.0"
STEP_NAME = "device_error_counters"
DELTA_KEYS = ["ecc_corrected_volatile", "ecc_uncorrected_volatile", "remapped_rows_corrected",
              "remapped_rows_uncorrected", "retired_pages_single_bit", "retired_pages_double_bit"]
UNCORRECTED_KEYS = {"ecc_uncorrected_volatile", "remapped_rows_uncorrected", "retired_pages_double_bit"}
CONTEXT_KEYS = ["temperature_c", "sm_clock_mhz", "power_w"]


def _nvml():
    """The NVML bindings, or None when they or the driver are missing."""
    try:
        nv = importlib.import_module("pynvml")
        nv.nvmlInit()
        return nv
    except Exception:
        return None


def _handle(nv, dev):
    """The NVML handle for a torch CUDA device, matched by UUID first because CUDA and NVML can number
    devices differently when CUDA_VISIBLE_DEVICES is set."""
    import torch
    index = dev.index if dev.index is not None else torch.cuda.current_device()
    try:
        uuid = str(torch.cuda.get_device_properties(index).uuid)
        return nv.nvmlDeviceGetHandleByUUID(uuid if uuid.startswith("GPU-") else "GPU-" + uuid)
    except Exception:
        return nv.nvmlDeviceGetHandleByIndex(index)


def _try(fn, *args):
    try:
        return fn(*args)
    except Exception:
        return None


def snapshot(dev):
    """Read every counter this device supports. Returns None when no counters can be read."""
    if getattr(dev, "type", None) != "cuda":
        return None
    nv = _nvml()
    if nv is None:
        return None
    try:
        h = _handle(nv, dev)
    except Exception:
        return None
    s = {}
    mode = _try(nv.nvmlDeviceGetEccMode, h)
    if mode is not None:
        s["ecc_enabled"] = bool(mode[0])
    s["ecc_corrected_volatile"] = _try(nv.nvmlDeviceGetTotalEccErrors, h, nv.NVML_MEMORY_ERROR_TYPE_CORRECTED, nv.NVML_VOLATILE_ECC)
    s["ecc_uncorrected_volatile"] = _try(nv.nvmlDeviceGetTotalEccErrors, h, nv.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, nv.NVML_VOLATILE_ECC)
    rows = _try(nv.nvmlDeviceGetRemappedRows, h)
    if rows is not None:
        s["remapped_rows_corrected"], s["remapped_rows_uncorrected"] = int(rows[0]), int(rows[1])
        s["remap_pending"], s["remap_failure"] = bool(rows[2]), bool(rows[3])
    sbe = _try(nv.nvmlDeviceGetRetiredPages, h, nv.NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS)
    dbe = _try(nv.nvmlDeviceGetRetiredPages, h, nv.NVML_PAGE_RETIREMENT_CAUSE_DOUBLE_BIT_ECC_ERROR)
    s["retired_pages_single_bit"] = len(sbe) if sbe is not None else None
    s["retired_pages_double_bit"] = len(dbe) if dbe is not None else None
    s["temperature_c"] = _try(nv.nvmlDeviceGetTemperature, h, nv.NVML_TEMPERATURE_GPU)
    s["sm_clock_mhz"] = _try(nv.nvmlDeviceGetClockInfo, h, nv.NVML_CLOCK_SM)
    mw = _try(nv.nvmlDeviceGetPowerUsage, h)
    s["power_w"] = round(mw / 1000.0, 1) if mw is not None else None
    s = {k: v for k, v in s.items() if v is not None}
    return s if any(k in s for k in DELTA_KEYS) else None


def judge(before, after, probe_ok, injected):
    """Cross-check the counters against the probe. Returns (verdict, ok, message, deltas)."""
    if before is None or after is None:
        return "counters-unavailable", True, "no error counters on this device or no NVIDIA management library", {}
    deltas = {k: after[k] - before[k] for k in DELTA_KEYS if k in before and k in after}
    moved = {k: d for k, d in deltas.items() if d > 0}
    flags = [k for k in ("remap_pending", "remap_failure") if after.get(k) and not before.get(k)]
    summary = ", ".join(f"{k} +{d}" for k, d in moved.items()) + (", " if moved and flags else "") + ", ".join(flags)
    if injected:
        return "self-test-run", True, "faults were planted in software, so no cross-check is drawn" + (
            f"; counters moved: {summary}" if summary else "; no counter moved"), deltas
    if any(k in UNCORRECTED_KEYS for k in moved) or "remap_failure" in flags:
        return "uncorrected-during-run", False, f"the hardware recorded uncorrected errors during the run: {summary}", deltas
    if not probe_ok and not moved and not flags:
        return ("silent-error-confirmed", False,
                "the probes found wrong answers and every error counter stayed quiet: the hardware did not notice", deltas)
    if not probe_ok:
        return "hardware-noticed", False, f"the probes found wrong answers and the counters moved: {summary}", deltas
    if moved or flags:
        return "corrected-during-run", True, f"every answer was right, but the hardware corrected errors: {summary}", deltas
    return "counters-quiet", True, "no error counter moved during the run", deltas


def record(run, hw, before, after, probe_ok, injected):
    """Write the device_error_counters step. Returns False only when the counters show a hardware fault the
    probes' own verdicts do not already carry."""
    verdict, ok, msg, deltas = judge(before, after, probe_ok, injected)
    step = run.add_step(STEP_NAME)
    step.start()
    step.add_measurement(name="counters_available", value=before is not None and after is not None, hardware_info=hw)
    if before is not None and after is not None:
        if "ecc_enabled" in after:
            step.add_measurement(name="ecc_enabled", value=after["ecc_enabled"], hardware_info=hw)
        for k, d in deltas.items():
            step.add_measurement(name=f"{k}_before", value=before[k], hardware_info=hw)
            step.add_measurement(name=f"{k}_after", value=after[k], hardware_info=hw)
            step.add_measurement(name=f"{k}_delta", value=d, hardware_info=hw)
        for k in ("remap_pending", "remap_failure"):
            if k in after:
                step.add_measurement(name=k, value=after[k], hardware_info=hw)
        for k in CONTEXT_KEYS:
            for when, snap in (("before", before), ("after", after)):
                if k in snap:
                    step.add_measurement(name=f"{k}_{when}", value=snap[k], hardware_info=hw)
    kind = tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL
    if verdict == "counters-unavailable":
        kind = tv.DiagnosisType.UNKNOWN
    step.add_diagnosis(kind, verdict=verdict, message=msg, hardware_info=hw)
    step.end(status=tv.TestStatus.COMPLETE)
    print(f"  {'error counters':>16s}  {verdict}  {msg}")
    return ok or verdict != "uncorrected-during-run"
