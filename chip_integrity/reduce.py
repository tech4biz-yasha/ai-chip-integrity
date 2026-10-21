#!/usr/bin/env python3
"""
reduce.py  -  AI Chip Integrity Suite, reductions (probe 5) v0.1.0

Checks whether the chip adds numbers up correctly. Adding is everywhere in AI: every matrix multiply is a
chain of additions, every softmax adds up a row, every norm adds up squares, and training adds gradients
across devices. Probes 1 to 4 only see the adders inside a matrix multiply. A chip also has separate
reduction paths for plain sums over long vectors, running sums, means, and sums down the rows and columns of
a matrix, and those run through different kernels and different silicon. This probe hands the chip long
lists of numbers, asks for those results, and checks every one of them two ways.

  exact     whole numbers from {-1, 0, 1} arranged so that every partial sum, in any order, stays a small
            integer that the output type holds exactly. The true answer is one exact number and any
            difference at all is a fault (or an accumulator narrower than the FP32 the arithmetic model
            assumes, which is also a finding).
  bounded   random numbers, rounded to the input type, with Higham's summation bound: for a sum of n terms
            |fl(s) - s| <= gamma(n-1) * sum|x| no matter in what order or tree the chip adds, plus the
            rounding of the result into its type and the float64 reference's own error. The bound holds for
            every order, which is why the chip's kernel never has to be known; it is loose for very long
            sums, which is why the exact case exists.

The reference is float64 recursive summation with its own error term in the bound (gamma(n) at u = 2^-53),
which is stricter than Kahan summation in the input type and the same reference every other probe uses.
Every result is also repeated and compared bit for bit, as in probe 1.

  no-silent-errors          every run bit-identical, every result inside its bound or exact
  silent-data-corruption    runs differed and a result was wrong: an intermittent fault
  outside-error-bound       runs agree but a result was wrong: a faulty unit, or an accumulator narrower
                            than the arithmetic model assumes
  nondeterministic-kernel   runs differed but every result stayed inside its bound; reported as a finding

--inject N adds N planted-fault runs per step after the clean ones: one result moved eight bounds away from
its float64 answer (or by one whole unit in the exact case), at a random element for the running sum and the
row and column sums, which the check must catch and locate.

Output: OCP Test & Validation JSON, one object per line, written to --out.
"""

import argparse
import hashlib
import os
import random
import socket
import sys
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch  # noqa: E402
import ocptv.output as tv  # noqa: E402

from chip_integrity import arith  # noqa: E402
from chip_integrity import counters  # noqa: E402
from chip_integrity import screen  # noqa: E402

VERSION = "0.1.0"
DTYPES = ("fp32", "fp16", "bf16")
DEFAULT_DTYPES = {"cuda": "fp32,fp16,bf16", "mps": "fp16,fp32", "cpu": "fp32"}
DEFAULT_N = {"cuda": 1 << 23, "mps": 1 << 22, "cpu": 1 << 20}          # vector length, a power of two
DEFAULT_SIDE = {"cuda": 4096, "mps": 2048, "cpu": 1024}               # matrix side for row and column sums
OPS = ("sum", "cumsum", "mean", "rowsum", "colsum")
EXACT_OPS = ("sum", "cumsum", "rowsum", "colsum")                     # mean needs no exact case: sum/2^k is exact
EXACT_TAIL = 128             # the exact sequence ends at this total; every prefix stays within [-1, EXACT_TAIL], exact even in bf16 (8-bit mantissa, integers to 256)
MAX_EXACT_N = 1 << 23        # partial sums below 2^23 are exact in an FP32 accumulator in any order
_FAULT_HOOK = None           # tests only: callable(step_name, run, result_tensor) that corrupts a result in place


def make_exact(n, gen):
    """Whole numbers from {-1, 0, 1}: pairs (+1, -1) or (-1, +1) and zeros, so every prefix sum is -1, 0 or 1,
    then a tail of +1 so the total is EXACT_TAIL. Every partial sum in any order is an integer below n."""
    pairs = n // 2 - EXACT_TAIL // 2
    choice = torch.randint(0, 3, (pairs,), generator=gen)
    first = torch.where(choice == 0, torch.tensor(1), torch.where(choice == 1, torch.tensor(-1), torch.tensor(0)))
    seq = torch.stack([first, -first], dim=1).reshape(-1)
    tail = torch.ones(n - seq.numel(), dtype=seq.dtype)
    x = torch.cat([seq[:n - EXACT_TAIL], tail[:EXACT_TAIL]]).to(torch.float64)
    assert x.numel() == n and float(x.sum()) == EXACT_TAIL
    return x


def make_random(n, gen):
    return torch.randn(n, generator=gen, dtype=torch.float64)


def sum_bound(ref, abs_sum, n, u_out, out_dtype):
    """|fl(s) - s| for a sum of n terms in any order: the probe 1 bound with sum|x| in place of |A||B|."""
    return screen.error_bound(ref, abs_sum, n, u_out, g=arith.ieee_dot(n), out_floor=arith.subnormal_floor(out_dtype))


class Case:
    """One operation, input kind and precision: the inputs on the device, the float64 reference and the bound."""

    def __init__(self, op, kind, name, n, side, seed, dev):
        self.op, self.kind, self.name, self.dev = op, kind, name, dev
        self.dtype, self.out_dtype = screen.DTYPES[name][0], screen.DTYPES[name][1]
        self.u_out = screen.DTYPES[name][4]
        gen = torch.Generator().manual_seed(seed)
        count = side * side if op in ("rowsum", "colsum") else n
        x64 = make_exact(count, gen) if kind == "exact" else make_random(count, gen).to(self.dtype).to(torch.float64)
        self.n = count
        if op in ("rowsum", "colsum"):
            x64 = x64.reshape(side, side)
            terms = side
            self.ref = x64.sum(dim=1) if op == "rowsum" else x64.sum(dim=0)
            abs_sum = x64.abs().sum(dim=1) if op == "rowsum" else x64.abs().sum(dim=0)
        elif op == "cumsum":
            terms = torch.arange(1, count + 1, dtype=torch.float64)
            self.ref = x64.cumsum(0)
            abs_sum = x64.abs().cumsum(0)
        else:
            terms = count
            self.ref = x64.sum().reshape(1)
            abs_sum = x64.abs().sum().reshape(1)
            if op == "mean":
                self.ref = self.ref / count
        self.terms = terms
        if kind == "exact":
            self.bound = torch.zeros_like(self.ref)
        elif op == "cumsum":
            g = terms * arith.U_RN / (1.0 - terms * arith.U_RN)                  # gamma(i) per prefix
            gref = terms * 2.0 ** -53 / (1.0 - terms * 2.0 ** -53)
            acc = (g + gref) * abs_sum + arith.underflow_floor(count)
            self.bound = acc + self.u_out * (self.ref.abs() + acc) + arith.subnormal_floor(self.out_dtype)
        elif op == "mean":
            self.bound = sum_bound(self.ref * count, abs_sum, count, self.u_out, self.out_dtype) / count \
                + self.u_out * self.ref.abs()
        else:
            self.bound = sum_bound(self.ref, abs_sum, terms, self.u_out, self.out_dtype)
        self.x = x64.to(self.dtype).to(dev)
        self.first = None

    def run(self):
        x = self.x
        if self.op == "sum":
            y = x.sum().reshape(1)
        elif self.op == "mean":
            y = x.mean().reshape(1)
        elif self.op == "cumsum":
            y = x.cumsum(0)
        elif self.op == "rowsum":
            y = x.sum(dim=1)
        else:
            y = x.sum(dim=0)
        return y

    def check(self, y):
        """(elements outside the bound or not exact, worst error over bound, index of the worst element)."""
        y64 = y.detach().to("cpu", torch.float64).reshape(self.ref.shape)
        err = (y64 - self.ref).abs()
        if self.kind == "exact":
            bad = (err > 0) | ~torch.isfinite(y64)
            score = err
        else:
            bad = (err > self.bound) | ~torch.isfinite(y64)
            score = err / self.bound.clamp_min(torch.finfo(torch.float64).tiny)   # error as a share of its own bound
        score = torch.where(torch.isfinite(score), score, torch.full_like(score, float("inf")))
        worst = float(score.max())
        idx = int(score.argmax()) if bad.any() else -1
        return int(bad.sum()), worst, idx

    def plant(self, y, rng):
        """Self-test: move one result eight bounds away from its float64 answer (one unit in the exact case)."""
        flat = y.detach().reshape(-1)
        e = rng.randrange(flat.numel())
        delta = 1.0 if self.kind == "exact" else 8.0 * float(self.bound.reshape(-1)[e])
        wrong = float(self.ref.reshape(-1)[e]) + delta
        flat[e] = torch.tensor(wrong, dtype=flat.dtype)
        return e


def digest(y):
    return hashlib.sha256(y.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def run_case(step, case, runs, inject, rng, hw, step_name):
    t0 = time.monotonic()
    first_hash, differing, outside, worst, worst_idx = None, 0, 0, 0.0, -1
    planted = []
    for r in range(runs + inject):
        y = case.run()
        if _FAULT_HOOK is not None:
            _FAULT_HOOK(step_name, r, y)
        where = case.plant(y, rng) if r >= runs else None
        h = digest(y)
        bad, w, idx = case.check(y)
        if where is None:
            if first_hash is None:
                first_hash = h
            elif h != first_hash:
                differing += 1
            outside += bad
            if w > worst:
                worst, worst_idx = w, idx
        else:
            planted.append({"index": where, "caught": bad > 0, "located": bad > 0 and idx == where})
    secs = time.monotonic() - t0
    exact = case.kind == "exact"
    if differing and outside:
        ok, verdict = False, "silent-data-corruption"
        msg = f"{differing} of {runs - 1} repeat runs differed and {outside} result(s) were " + ("not exact" if exact else "outside the bound")
    elif differing:
        ok, verdict = False, "nondeterministic-kernel"
        msg = f"{differing} of {runs - 1} repeat runs differed with every result " + ("exact" if exact else "inside its bound")
    elif outside:
        ok, verdict = False, "outside-error-bound"
        msg = (f"{outside} result(s) not exact, largest error {worst:.3g}" if exact
               else f"{outside} result(s) outside the bound, worst {worst:.3g}x")
    else:
        ok, verdict = True, "no-silent-errors"
        msg = f"{runs} runs bit-identical, every result " + ("exact" if exact else f"inside its bound (worst {worst:.3g}x)")
    caught = sum(1 for p in planted if p["caught"])
    located = sum(1 for p in planted if p["located"])
    if inject:
        ok = ok and caught == len(planted) == located
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg += f"; planted {len(planted)}, caught {caught}, located {located}"

    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    m = step.add_measurement
    m(name="operation", value=case.op, hardware_info=hw)
    m(name="input_kind", value=case.kind, hardware_info=hw)
    m(name="precision", value=case.name, hardware_info=hw)
    m(name="elements", value=case.n, hardware_info=hw)
    m(name="terms_per_result", value=int(case.terms) if not torch.is_tensor(case.terms) else int(case.terms.max()),
      hardware_info=hw)
    m(name="results", value=int(case.ref.numel()), hardware_info=hw)
    m(name="runs", value=runs, hardware_info=hw)
    m(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    m(name="repeat_runs_differing", value=differing, validators=eq0, hardware_info=hw)
    m(name="results_wrong", value=outside, validators=eq0, hardware_info=hw)
    if exact:
        m(name="largest_error", value=float(f"{worst:.6g}"), hardware_info=hw)
    else:
        m(name="accumulation_factor", value=float(f"{arith.ieee_dot(int(case.terms) if not torch.is_tensor(case.terms) else int(case.terms.max())):.6g}"),
          hardware_info=hw)
        m(name="worst_error_over_bound_ratio", value=float(f"{worst:.6g}"), hardware_info=hw)
    if worst_idx >= 0:
        m(name="worst_result_index", value=worst_idx, hardware_info=hw)
    if inject:
        m(name="injected_runs", value=len(planted), hardware_info=hw)
        m(name="injected_runs_detected", value=caught,
          validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=len(planted))], hardware_info=hw)
        m(name="injected_runs_located", value=located,
          validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=len(planted))], hardware_info=hw)
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                       hardware_info=hw)
    return {"ok": ok, "verdict": verdict, "msg": msg, "secs": secs, "differing": differing, "outside": outside,
            "planted": len(planted), "caught": caught}


def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite, reductions (probe 5)")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--n", type=int, default=None,
                   help="vector length, a power of two (default 8388608 on CUDA, 4194304 on MPS, 1048576 on CPU)")
    p.add_argument("--side", type=int, default=None,
                   help="matrix side for row and column sums (default 4096 on CUDA, 2048 on MPS, 1024 on CPU)")
    p.add_argument("--dtypes", default=None,
                   help="comma list from fp32,fp16,bf16 (default fp32,fp16,bf16 on CUDA, fp16,fp32 on MPS, fp32 on CPU)")
    p.add_argument("--ops", default=",".join(OPS), help="comma list from sum,cumsum,mean,rowsum,colsum")
    p.add_argument("--runs", type=int, default=3, help="clean runs per step; runs after the first are compared bit for bit")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="planted-fault runs per step (self-test)")
    p.add_argument("--out", default="reduce.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    dev = screen.pick_device(args.device)
    n = args.n or DEFAULT_N.get(dev.type, DEFAULT_N["cpu"])
    side = args.side or DEFAULT_SIDE.get(dev.type, DEFAULT_SIDE["cpu"])
    names = [d.strip() for d in (args.dtypes or DEFAULT_DTYPES.get(dev.type, "fp32")).split(",") if d.strip()]
    ops = [o.strip() for o in args.ops.split(",") if o.strip()]
    if args.runs < 2 or args.inject < 0:
        p.error("need --runs >= 2 and --inject >= 0")
    if n < 2 * EXACT_TAIL or n & (n - 1) or n > MAX_EXACT_N:
        p.error(f"--n must be a power of two between {2 * EXACT_TAIL} and {MAX_EXACT_N}")
    if side < 32 or side & (side - 1) or side * side > MAX_EXACT_N:
        p.error(f"--side must be a power of two from 32 with side*side <= {MAX_EXACT_N}")
    bad = [d for d in names if d not in DTYPES]
    if bad or not names:
        p.error(f"check --dtypes: unknown {bad}, choose from {','.join(DTYPES)}")
    bad = [o for o in ops if o not in OPS]
    if bad or not ops:
        p.error(f"check --ops: unknown {bad}, choose from {','.join(OPS)}")

    screen.lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = screen.FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(
        name="ai-chip-integrity-reduce",
        version=VERSION,
        parameters={"device": str(dev), "n": n, "side": side, "dtypes": ",".join(names), "ops": ",".join(ops),
                    "runs": args.runs, "seed": args.seed, "inject": args.inject, "exact_tail": EXACT_TAIL,
                    "bound_model": arith.MODEL_ID},
    )
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=screen.device_label(dev))
    screen.add_software_infos(dut, dev)

    print(f"Device: {screen.device_label(dev)} ({dev})  n={n}  side={side}  precisions={','.join(names)}  "
          f"ops={','.join(ops)}  runs={args.runs}  inject={args.inject}")
    all_ok, results = True, []
    run.start(dut=dut)
    before = counters.snapshot(dev)
    try:
        for name in names:
            for op in ops:
                for kind in (("exact", "bounded") if op in EXACT_OPS else ("bounded",)):
                    step_name = f"reduce_{op}_{kind}_{name}"
                    step = run.add_step(step_name)
                    step.start()
                    try:
                        case = Case(op, kind, name, n, side, args.seed, dev)
                        res = run_case(step, case, args.runs, args.inject, rng, hw, step_name)
                        step.end(status=tv.TestStatus.COMPLETE)
                    except (RuntimeError, TypeError, NotImplementedError) as e:
                        step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                        step.end(status=tv.TestStatus.SKIP)
                        print(f"  {step_name:28s} SKIPPED  {str(e)[:100]}")
                        continue
                    all_ok &= res["ok"]
                    results.append((step_name, res))
                    print(f"  {step_name:28s} {'PASS' if res['ok'] else 'FAIL'}  {res['verdict']}  ({res['secs']:.1f}s)  {res['msg']}")

        step = run.add_step("reduce_summary")
        step.start()
        eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
        step.add_measurement(name="steps", value=len(results), hardware_info=hw)
        step.add_measurement(name="precisions", value=",".join(names), hardware_info=hw)
        step.add_measurement(name="repeat_runs_differing", value=sum(r["differing"] for _, r in results),
                             validators=eq0, hardware_info=hw)
        step.add_measurement(name="results_wrong", value=sum(r["outside"] for _, r in results), validators=eq0,
                             hardware_info=hw)
        if args.inject:
            planted, caught = sum(r["planted"] for _, r in results), sum(r["caught"] for _, r in results)
            step.add_measurement(name="injected_runs", value=planted, hardware_info=hw)
            step.add_measurement(name="injected_runs_detected", value=caught,
                                 validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=planted)], hardware_info=hw)
        failed = [(s, r) for s, r in results if not r["ok"]]
        if not results:
            ok, verdict, msg = False, "not-run", "no step could run on this device"
        elif args.inject:
            ok = not failed
            verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
            msg = f"{len(results)} steps, " + ("every planted fault caught and located" if ok else
                                               f"failed: {', '.join(s for s, _ in failed)}")
        elif failed:
            ok, verdict = False, failed[0][1]["verdict"]
            msg = f"failed: {', '.join(s for s, _ in failed)}"
        else:
            ok, verdict = True, "no-silent-errors"
            msg = f"{len(results)} steps clean, {args.runs} runs each"
        all_ok &= ok
        step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                           hardware_info=hw)
        step.end(status=tv.TestStatus.COMPLETE)
        print(f"  {'summary':28s} {'PASS' if ok else 'FAIL'}  {verdict}  {msg}")
    finally:
        try:
            all_ok &= counters.record(run, hw, before, counters.snapshot(dev), probe_ok=all_ok, injected=args.inject > 0)
        except Exception as e:                       # the cross-check must never cost the run its result file
            print(f"  error counters could not be recorded: {e}")
        run.end(status=tv.TestStatus.COMPLETE, result=tv.TestResult.PASS if all_ok else tv.TestResult.FAIL)
        writer.close()

    print(f"Result: {'PASS' if all_ok else 'FAIL'}   OCP output: {args.out}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
