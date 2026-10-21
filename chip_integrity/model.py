#!/usr/bin/env python3
"""
model.py  -  AI Chip Integrity Suite, full model forward pass (probe 4) v0.1.0

Runs a small open language model on the chip and checks it three ways, with a float64 copy of the same model
on the CPU as the reference. The matrix multiplies are where a wrong answer is made, and a bound carried through
a whole model would be far too loose to catch anything, so every matrix multiply inside the model (the attention
projections, the MLP and the output head) is checked on its own: the hook sees the input the chip actually gave
it, computes the float64 answer from that input and the proven rounding bound, and every output element has to
sit inside the bound. The kernels between the multiplies (softmax, norms, activations, rotary embeddings) are
covered by probe 3 with their own bounds.

  1. Every matrix multiply inside the model against float64 with the proven bound, fed the chip's own input.
  2. The weights read back from the chip after every run and compared bit for bit with what was loaded.
  3. The whole run (every logit of the prompt and of each generated token) repeated and compared bit for bit.

The decision figures against float64 are recorded as measurements, not verdicts: how many prompt positions
pick a different token than float64 would, the smallest float64 margin at those positions (a near tie is a
legitimate place for a lower precision to differ), the first generated token that diverges, and the largest
logit difference at the decision position. Low precision can flip a near tie without any fault; the verdict
rests on the three checks above.

  no-silent-errors          every run bit-identical, every multiply inside its bound, every weight unchanged
  silent-data-corruption    runs differed, or a weight came back changed; an intermittent or memory fault
  outside-error-bound       runs agree but a multiply sat outside its bound: a faulty unit, or an arithmetic
                            model this chip does not follow
  nondeterministic-kernel   runs differed but every multiply stayed inside its bound, so the bit-for-bit check
                            cannot judge this chip and precision; reported as a finding

--inject N adds N runs with a planted fault after the clean ones: one bit flipped in the output of one random
matrix multiply, and one bit flipped in one random weight on the chip (restored afterwards), alternating. Each
must be caught by at least one of the three checks, and the file records whether the multiply check also
localised it to the right layer.

--model tiny builds a two-layer model of the same family from a seed, with no download, so the self-test runs
on a CPU in CI. The default model is downloaded from the Hugging Face Hub (pip install ai-chip-integrity[model]).

Output: OCP Test & Validation JSON, one object per line, written to --out.
"""

import argparse
import copy
import hashlib
import json
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
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B"
TINY = "tiny"
DEFAULT_DTYPES = {"cuda": "bf16,fp16,fp32", "mps": "fp16,fp32", "cpu": "fp32"}
FLOAT_DTYPES = ("fp32", "fp16", "bf16")
_FAULT_HOOK = None       # tests only: callable(dtype_name, run, model) called before a run; may corrupt the model

# A fixed English prompt set, scope stated: short general prompts of mixed length. --prompts replaces it.
PROMPTS = [
    "The capital of France is",
    "Water boils at a temperature of",
    "In 1969, the first people to walk on the Moon were",
    "A prime number is a whole number greater than one that",
    "The three primary colours of light are red, green and",
    "Photosynthesis is the process by which plants",
    "To bake bread you need flour, water, salt and",
    "The speed of light in a vacuum is about",
    "Shakespeare wrote plays such as Hamlet, Macbeth and",
    "The largest planet in our solar system is",
    "A triangle has three sides and the sum of its angles is",
    "The human heart pumps blood through the body using",
    "Rivers flow downhill because of",
    "The chemical symbol for gold is",
    "Computers store numbers in binary, which uses only the digits",
    "The seasons change during the year because the Earth's axis is",
]


# ----------------------------------------------------------------------------------------------- loading

def _transformers():
    try:
        import transformers
    except ImportError:
        raise SystemExit("this probe needs the transformers package: pip install 'ai-chip-integrity[model]'")
    return transformers


def tiny_model(seed):
    """A two-layer model of the Qwen2 family built from a seed: RMSNorm, SwiGLU, rotary embeddings, grouped
    query attention and an output head, the same pieces as the default model, small enough for a CPU."""
    tf = _transformers()
    cfg = tf.Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=512,
                         tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = tf.Qwen2ForCausalLM(cfg)
    return model.eval(), None


def load_model(name, attention, seed):
    """Returns (model on CPU in the checkpoint's dtype, tokenizer or None)."""
    if name == TINY:
        return tiny_model(seed)
    tf = _transformers()
    # torch_dtype="auto" loads the checkpoint in the dtype it was saved in; the newest transformers spell it
    # dtype= and warn, but torch_dtype= works on every version from 4.45 to 4.57.
    model = tf.AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto", attn_implementation=attention)
    tok = tf.AutoTokenizer.from_pretrained(name)
    return model.eval(), tok


def sig(x, digits=6):
    """A float to a few significant digits, so a 1e-7 difference does not round to 0.0."""
    return float(f"{x:.{digits}g}")


def tensor_bytes(t):
    return t.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def weights_sha256(model):
    """One hash over every parameter in name order, in the dtype it was loaded in: the pin for this row."""
    h = hashlib.sha256()
    for name, p in sorted(model.named_parameters()):
        h.update(name.encode())
        h.update(tensor_bytes(p))
    return h.hexdigest()


def tokenizer_sha256(tok):
    if tok is None:
        return "none"
    return hashlib.sha256(json.dumps(tok.get_vocab(), sort_keys=True).encode()).hexdigest()


def prompt_ids(tok, prompts, max_tokens, vocab, seed):
    """Token ids per prompt (1-D long tensors). Without a tokenizer (tiny model) prompts are random ids."""
    if tok is not None:
        ids = [tok(p, return_tensors="pt").input_ids[0, :max_tokens] for p in prompts]
        return [i for i in ids if i.numel() > 0]
    g = torch.Generator().manual_seed(seed)
    return [torch.randint(0, vocab, (int(torch.randint(4, max_tokens + 1, (1,), generator=g)),), generator=g)
            for _ in prompts]


def layer_label(module_name):
    parts = module_name.split(".")
    if "layers" in parts:
        return f"layer_{int(parts[parts.index('layers') + 1]):02d}"
    if module_name.endswith("lm_head"):
        return "head"
    return "other"


# ------------------------------------------------------------------------------------- the matrix multiplies

class LinearCheck:
    """Hooks every nn.Linear on the device model. For each call: the chip's own input and weight go to the CPU
    in float64, the reference and the proven bound are computed from them, and every output element is checked.
    Also the place where the self-test plants an activation flip."""

    def __init__(self, model, dev, name):
        self.dev, self.name = dev, name
        self.out_dtype = screen.DTYPES[name][1]
        self.u_out = screen.DTYPES[name][4]
        self.linears = [(n, m) for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        self.handles = [m.register_forward_hook(self._make_hook(n)) for n, m in self.linears]
        self.plant = None                    # (call index, rng) for the self-test
        self.reset()

    def reset(self):
        self.calls = 0
        self.outside = 0
        self.nonfinite_inputs = 0
        self.worst = 0.0
        self.by_layer = {}
        self.call_index = 0
        self.planted_at = None

    def remove(self):
        for h in self.handles:
            h.remove()

    def _make_hook(self, module_name):
        label = layer_label(module_name)

        def hook(module, inputs, output):
            idx = self.call_index
            self.call_index += 1
            if self.plant is not None and self.plant[0] == idx:
                c = output.detach().to("cpu").clone()
                int_view, bits = screen.DTYPES[self.name][2], screen.DTYPES[self.name][3]
                screen.flip_bit(c, int_view, bits, self.plant[1])
                output.detach().copy_(c.to(output.device))
                self.planted_at = (label, module_name)
            x = inputs[0].detach()
            x64 = x.to("cpu", torch.float64).reshape(-1, x.shape[-1])
            w64 = module.weight.detach().to("cpu", torch.float64)
            if not torch.isfinite(x64).all() or not torch.isfinite(w64).all():
                self.nonfinite_inputs += 1
                return
            k = x64.shape[1]
            ref = x64 @ w64.T
            abs_ab = x64.abs() @ w64.abs().T
            flush_ab = arith.flush_term(x64, x.dtype) @ w64.abs().T + x64.abs() @ arith.flush_term(w64, x.dtype).T
            if module.bias is not None:
                b64 = module.bias.detach().to("cpu", torch.float64)
                ref = ref + b64
                abs_ab = abs_ab + b64.abs()          # one more rounded term in the sum
                k = k + 1
            g = screen.accumulation_factor(self.name, k, self.dev.type)
            bound = screen.error_bound(ref, abs_ab, k, self.u_out, g=g, flush_ab=flush_ab,
                                       out_floor=arith.subnormal_floor(self.out_dtype))
            y64 = output.detach().to("cpu", torch.float64).reshape(ref.shape)
            err = (y64 - ref).abs()
            bad = (err > bound) | ~torch.isfinite(y64)
            outside = int(bad.sum())
            ratio = float((err / bound.clamp_min(torch.finfo(torch.float64).tiny)).max())
            self.calls += 1
            self.outside += outside
            self.worst = max(self.worst, ratio)
            entry = self.by_layer.setdefault(label, {"calls": 0, "outside": 0, "worst": 0.0})
            entry["calls"] += 1
            entry["outside"] += outside
            entry["worst"] = max(entry["worst"], ratio)

        return hook


# --------------------------------------------------------------------------------------- running the model

@torch.inference_mode()
def forward_greedy(model, ids, new_tokens, dev):
    """Prefill the prompt, then generate new_tokens greedily through the cache.
    Returns (prefill logits [T, V], generated token ids, sha256 of every logit of the run)."""
    h = hashlib.sha256()
    out = model(input_ids=ids.to(dev)[None], use_cache=True)
    logits = out.logits[0]
    h.update(tensor_bytes(logits))
    prefill = logits.detach().to("cpu")
    past = out.past_key_values
    tokens = []
    nxt = logits[-1].argmax()
    for _ in range(new_tokens):
        tokens.append(int(nxt))
        out = model(input_ids=nxt.reshape(1, 1), past_key_values=past, use_cache=True)
        past = out.past_key_values
        step_logits = out.logits[0, -1]
        h.update(tensor_bytes(step_logits))
        nxt = step_logits.argmax()
    return prefill, tokens, h.hexdigest()


def reference_for(ref_model, ids, new_tokens):
    """What the float64 model decides for this prompt: argmax and top-two margin at every prompt position,
    the logits at the decision position, and the greedy continuation."""
    prefill, tokens, _ = forward_greedy(ref_model, ids, new_tokens, torch.device("cpu"))
    top2 = prefill.topk(2, dim=-1).values
    return {"argmax": prefill.argmax(dim=-1), "margin": (top2[:, 0] - top2[:, 1]), "last": prefill[-1],
            "tokens": tokens}


def compare_to_reference(prefill, tokens, ref):
    """Decision measurements for one prompt: device run 0 against the float64 model."""
    dev_argmax = prefill.to(torch.float64).argmax(dim=-1)
    disagree = (dev_argmax != ref["argmax"])
    first_div = next((i for i, (a, b) in enumerate(zip(tokens, ref["tokens"])) if a != b), -1)
    return {
        "positions": int(disagree.numel()),
        "disagreeing": int(disagree.sum()),
        "min_margin": float(ref["margin"][disagree].min()) if disagree.any() else None,
        "max_logit_diff": float((prefill[-1].to(torch.float64) - ref["last"]).abs().max()),
        "first_divergence": first_div,
    }


def weights_changed(model, pristine):
    """(elements changed, first parameter name that changed) after reading every weight back from the chip."""
    changed, first = 0, None
    for name, p in model.named_parameters():
        a = p.detach().to("cpu")
        if not torch.equal(a, pristine[name]):
            changed += int((a != pristine[name]).sum()) or 1
            first = first or name
    return changed, first


def flip_weight(model, pristine, name_of_dtype, rng):
    """Flip one bit of one element of one random Linear weight on the chip. Returns the parameter name."""
    names = [n for n, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
    target = rng.choice(names) + ".weight"
    p = dict(model.named_parameters())[target]
    c = p.detach().to("cpu").clone()
    screen.flip_bit(c, screen.DTYPES[name_of_dtype][2], screen.DTYPES[name_of_dtype][3], rng)
    p.detach().copy_(c.to(p.device))
    return target


def restore_weights(model, pristine):
    for name, p in model.named_parameters():
        p.detach().copy_(pristine[name].to(p.device))


# ------------------------------------------------------------------------------------------------- one dtype

def forward_step(step, dev, name, cpu_model, prompts, refs, runs, new_tokens, inject, rng, hw):
    """All prompts, all runs and the planted faults for one precision. Returns the summary dict."""
    dt = screen.DTYPES[name][0]
    model = copy.deepcopy(cpu_model).to(dt)
    pristine = {n: p.detach().clone() for n, p in model.named_parameters()}
    model = model.to(dev)
    check = LinearCheck(model, dev, name)
    t0 = time.monotonic()
    run0 = {}                     # prompt index -> hash of run 0
    differing, clean_outside, clean_worst, nonfinite = 0, 0, 0.0, 0
    weight_changes, first_weight = 0, None
    decision = {"positions": 0, "disagreeing": 0, "min_margin": None, "max_logit_diff": 0.0, "first_divergence": [],
                "prompts_disagreeing": 0}
    by_layer = {}
    injected = []
    try:
        for r in range(runs + inject):
            planted, planted_layer_outside = None, 0
            if r >= runs:                                       # a planted run
                j = r - runs
                if j % 2 == 0:                                  # call_index is the calls of the last prompt run
                    check.plant = (rng.randrange(check.call_index or 1), rng)
                    planted = {"kind": "activation", "prompt": 0}
                else:
                    planted = {"kind": "weight", "param": flip_weight(model, pristine, name, rng), "prompt": 0}
            if _FAULT_HOOK is not None:
                _FAULT_HOOK(name, r, model)
            run_outside, run_hash_diff, run_worst, run_layer = 0, False, 0.0, None
            for i, ids in enumerate(prompts):
                check.reset()
                prefill, tokens, digest = forward_greedy(model, ids, new_tokens, dev)
                check.plant = None                              # a planted flip lands in prompt 0 only
                if planted is not None and i == 0:
                    run_layer = check.planted_at[0] if check.planted_at else (
                        layer_label(planted["param"]) if planted["kind"] == "weight" else None)
                    planted_layer_outside = check.by_layer.get(run_layer, {}).get("outside", 0) if run_layer else 0
                run_outside += check.outside
                run_worst = max(run_worst, check.worst)
                nonfinite += check.nonfinite_inputs
                if r == 0:
                    run0[i] = digest
                    d = compare_to_reference(prefill, tokens, refs[i])
                    decision["positions"] += d["positions"]
                    decision["disagreeing"] += d["disagreeing"]
                    decision["prompts_disagreeing"] += int(d["disagreeing"] > 0 or d["first_divergence"] >= 0)
                    if d["min_margin"] is not None:
                        decision["min_margin"] = d["min_margin"] if decision["min_margin"] is None else \
                            min(decision["min_margin"], d["min_margin"])
                    decision["max_logit_diff"] = max(decision["max_logit_diff"], d["max_logit_diff"])
                    decision["first_divergence"].append(d["first_divergence"])
                elif digest != run0[i]:
                    run_hash_diff = True
                if planted is None:
                    for label, e in check.by_layer.items():
                        agg = by_layer.setdefault(label, {"calls": 0, "outside": 0, "worst": 0.0})
                        agg["calls"] += e["calls"]
                        agg["outside"] += e["outside"]
                        agg["worst"] = max(agg["worst"], e["worst"])
            changed, which = weights_changed(model, pristine)
            if planted is None:
                differing += int(run_hash_diff)
                clean_outside += run_outside
                clean_worst = max(clean_worst, run_worst)
                if changed:
                    weight_changes += changed
                    first_weight = first_weight or which
            else:
                caught = run_hash_diff or run_outside > 0 or changed > 0
                localised = planted_layer_outside > 0 if planted["kind"] == "activation" else \
                    (changed > 0 and which == planted["param"])
                injected.append({"kind": planted["kind"], "layer": run_layer, "caught": caught, "localised": localised})
                if planted["kind"] == "weight":
                    restore_weights(model, pristine)
    finally:
        check.remove()
    secs = time.monotonic() - t0

    # verdict from the clean runs
    if weight_changes:
        ok, verdict = False, "silent-data-corruption"
        msg = f"{weight_changes} weight element(s) came back changed from the chip, first in {first_weight}"
    elif differing and clean_outside:
        ok, verdict = False, "silent-data-corruption"
        msg = (f"{differing} of {runs - 1} repeat runs differed and {clean_outside} multiply output(s) "
               f"sat outside the bound")
    elif differing:
        ok, verdict = False, "nondeterministic-kernel"
        msg = f"{differing} of {runs - 1} repeat runs differed with every multiply inside its bound"
    elif clean_outside:
        ok, verdict = False, "outside-error-bound"
        msg = f"{clean_outside} multiply output(s) sat outside the bound, worst {clean_worst:.3g}x"
    else:
        ok, verdict = True, "no-silent-errors"
        msg = (f"{len(prompts)} prompts x {runs} runs bit-identical, every multiply inside its bound "
               f"(worst {clean_worst:.3g}x), every weight unchanged")
    if nonfinite:
        msg += f"; {nonfinite} multiply call(s) had non-finite inputs and were not judged"
    caught = sum(1 for j in injected if j["caught"])
    if inject:
        ok = ok and caught == len(injected)
        verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
        msg += f"; planted {len(injected)}, caught {caught}, localised {sum(1 for j in injected if j['localised'])}"

    eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
    m = step.add_measurement
    m(name="precision", value=name, hardware_info=hw)
    m(name="prompts", value=len(prompts), hardware_info=hw)
    m(name="runs", value=runs, hardware_info=hw)
    m(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
    m(name="repeat_runs_differing", value=differing, validators=eq0, hardware_info=hw)
    m(name="linear_calls", value=sum(e["calls"] for e in by_layer.values()), hardware_info=hw)
    m(name="linear_calls_outside_bound", value=clean_outside, validators=eq0, hardware_info=hw)
    m(name="worst_error_over_bound_ratio", value=sig(clean_worst), hardware_info=hw)
    m(name="linear_calls_nonfinite_input", value=nonfinite, hardware_info=hw)
    m(name="weight_elements_changed", value=weight_changes, validators=eq0, hardware_info=hw)
    m(name="positions_compared_with_float64", value=decision["positions"], hardware_info=hw)
    m(name="positions_disagreeing_with_float64", value=decision["disagreeing"], hardware_info=hw)
    m(name="prompts_disagreeing_with_float64", value=decision["prompts_disagreeing"], hardware_info=hw)
    if decision["min_margin"] is not None:
        m(name="smallest_float64_margin_at_disagreement", value=sig(decision["min_margin"]), hardware_info=hw)
    m(name="max_logit_abs_diff_at_decision", value=sig(decision["max_logit_diff"]), hardware_info=hw)
    diverged = [d for d in decision["first_divergence"] if d >= 0]
    m(name="generated_tokens", value=new_tokens, hardware_info=hw)
    m(name="prompts_with_generated_divergence", value=len(diverged), hardware_info=hw)
    m(name="first_divergence_token_min", value=min(diverged) if diverged else -1, hardware_info=hw)
    for label in sorted(by_layer):
        m(name=f"{label}_calls_outside_bound", value=by_layer[label]["outside"], hardware_info=hw)
        m(name=f"{label}_worst_error_over_bound_ratio", value=sig(by_layer[label]["worst"]), hardware_info=hw)
    if inject:
        m(name="injected_runs", value=len(injected), hardware_info=hw)
        m(name="injected_runs_detected", value=caught,
          validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=len(injected))], hardware_info=hw)
        m(name="injected_runs_localised", value=sum(1 for j in injected if j["localised"]), hardware_info=hw)
        m(name="injected_kinds", value=",".join(j["kind"] for j in injected), hardware_info=hw)
    step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                       hardware_info=hw)
    del model
    return {"ok": ok, "verdict": verdict, "msg": msg, "secs": secs, "differing": differing,
            "outside": clean_outside, "changed": weight_changes, "injected": len(injected), "caught": caught}


# ------------------------------------------------------------------------------------------------------ main

def main(argv=None):
    p = argparse.ArgumentParser(description="AI Chip Integrity Suite, full model forward pass (probe 4)")
    p.add_argument("--device", default="auto", help="auto, cpu, mps, cuda or cuda:N")
    p.add_argument("--model", default=DEFAULT_MODEL,
                   help="Hugging Face model id, or tiny for a two-layer model built from the seed (no download)")
    p.add_argument("--dtypes", default=None,
                   help="comma list from fp32,fp16,bf16 for the chip (default bf16,fp16,fp32 on CUDA, fp16,fp32 on MPS, "
                        "fp32 on CPU); the reference is always float64 on the CPU")
    p.add_argument("--prompts", default=None, help="text file, one prompt per line (default: the built-in 16)")
    p.add_argument("--max-prompts", type=int, default=16, help="prompts to use from the set")
    p.add_argument("--max-tokens", type=int, default=64, help="prompt tokens kept per prompt")
    p.add_argument("--new-tokens", type=int, default=8, help="tokens generated greedily after each prompt")
    p.add_argument("--runs", type=int, default=3,
                   help="clean runs per precision; runs after the first are compared bit for bit")
    p.add_argument("--attention", default="sdpa", choices=["sdpa", "eager"], help="attention implementation")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--inject", type=int, default=0, help="planted-fault runs per precision (self-test)")
    p.add_argument("--out", default="model.jsonl", help="OCP JSON output file")
    args = p.parse_args(argv)

    if args.runs < 2 or args.inject < 0 or args.max_prompts < 1 or args.max_tokens < 1 or args.new_tokens < 0:
        p.error("need --runs >= 2, --inject >= 0, --max-prompts >= 1, --max-tokens >= 1 and --new-tokens >= 0")
    dev = screen.pick_device(args.device)
    names = [d.strip() for d in (args.dtypes or DEFAULT_DTYPES.get(dev.type, "fp32")).split(",") if d.strip()]
    bad = [d for d in names if d not in FLOAT_DTYPES]
    if bad or not names:
        p.error(f"check --dtypes: unknown {bad}, choose from {','.join(FLOAT_DTYPES)}")
    if args.prompts:
        try:
            with open(args.prompts, encoding="utf-8") as f:
                prompts_text = [line.strip() for line in f if line.strip()]
        except OSError as e:
            p.error(f"--prompts: {e}")
        if not prompts_text:
            p.error("--prompts: the file has no prompts")
    else:
        prompts_text = PROMPTS
    prompts_text = prompts_text[:args.max_prompts]

    screen.lock_down_math(dev)
    rng = random.Random(args.seed)
    writer = screen.FileWriter(args.out)
    tv.config(writer=writer)
    run = tv.TestRun(
        name="ai-chip-integrity-model",
        version=VERSION,
        parameters={"device": str(dev), "model": args.model, "dtypes": ",".join(names), "prompts": len(prompts_text),
                    "max_tokens": args.max_tokens, "new_tokens": args.new_tokens, "runs": args.runs,
                    "attention": args.attention, "seed": args.seed, "inject": args.inject,
                    "bound_model": arith.MODEL_ID},
    )
    dut = tv.Dut(id=socket.gethostname())
    hw = dut.add_hardware_info(name=screen.device_label(dev))
    screen.add_software_infos(dut, dev)

    print(f"Device: {screen.device_label(dev)} ({dev})  model={args.model}  precisions={','.join(names)}  "
          f"prompts={len(prompts_text)}  runs={args.runs}  inject={args.inject}")
    all_ok, results = True, []
    run.start(dut=dut)
    before = counters.snapshot(dev)
    try:
        step = run.add_step("model_load")
        step.start()
        t0 = time.monotonic()
        cpu_model, tok = load_model(args.model, args.attention, args.seed)
        cfg = cpu_model.config
        prompts = prompt_ids(tok, prompts_text, args.max_tokens, cfg.vocab_size, args.seed)
        ckpt_dtype = str(next(cpu_model.parameters()).dtype).replace("torch.", "")
        m = step.add_measurement
        m(name="model", value=args.model, hardware_info=hw)
        m(name="weights_sha256", value=weights_sha256(cpu_model), hardware_info=hw)
        m(name="tokenizer_vocab_sha256", value=tokenizer_sha256(tok), hardware_info=hw)
        m(name="parameters", value=sum(p_.numel() for p_ in cpu_model.parameters()), hardware_info=hw)
        m(name="layers", value=int(cfg.num_hidden_layers), hardware_info=hw)
        m(name="hidden_size", value=int(cfg.hidden_size), hardware_info=hw)
        m(name="vocab_size", value=int(cfg.vocab_size), hardware_info=hw)
        m(name="checkpoint_dtype", value=ckpt_dtype, hardware_info=hw)
        m(name="attention", value=args.attention, hardware_info=hw)
        m(name="transformers_version", value=_transformers().__version__, hardware_info=hw)
        m(name="prompts", value=len(prompts), hardware_info=hw)
        m(name="prompt_tokens", value=sum(int(i.numel()) for i in prompts), hardware_info=hw)
        m(name="seconds", value=round(time.monotonic() - t0, 3), unit="s", hardware_info=hw)
        step.add_diagnosis(tv.DiagnosisType.PASS, verdict="model-pinned",
                           message=f"{args.model}: {cfg.num_hidden_layers} layers, {len(prompts)} prompts",
                           hardware_info=hw)
        step.end(status=tv.TestStatus.COMPLETE)
        print(f"  load      PASS  model-pinned  {cfg.num_hidden_layers} layers, {len(prompts)} prompts")

        step = run.add_step("model_reference_float64")
        step.start()
        t0 = time.monotonic()
        ref_model = copy.deepcopy(cpu_model).to(torch.float64)
        refs = [reference_for(ref_model, ids, args.new_tokens) for ids in prompts]
        del ref_model
        secs = time.monotonic() - t0
        step.add_measurement(name="prompts", value=len(prompts), hardware_info=hw)
        step.add_measurement(name="seconds", value=round(secs, 3), unit="s", hardware_info=hw)
        step.add_diagnosis(tv.DiagnosisType.PASS, verdict="reference-computed",
                           message=f"float64 on the CPU, {len(prompts)} prompts in {secs:.1f}s", hardware_info=hw)
        step.end(status=tv.TestStatus.COMPLETE)
        print(f"  reference PASS  reference-computed  ({secs:.1f}s)")

        for name in names:
            step = run.add_step(f"model_forward_{name}")
            step.start()
            try:
                res = forward_step(step, dev, name, cpu_model, prompts, refs, args.runs, args.new_tokens,
                                   args.inject, rng, hw)
                step.end(status=tv.TestStatus.COMPLETE)
            except (RuntimeError, TypeError, NotImplementedError) as e:
                step.add_error(symptom="not-supported-on-device", message=str(e)[:500])
                step.end(status=tv.TestStatus.SKIP)
                print(f"  {name:9s} SKIPPED  {str(e)[:100]}")
                continue
            all_ok &= res["ok"]
            results.append((name, res))
            print(f"  {name:9s} {'PASS' if res['ok'] else 'FAIL'}  {res['verdict']}  ({res['secs']:.1f}s)  "
                  f"{res['msg']}")

        step = run.add_step("model_summary")
        step.start()
        eq0 = [tv.Validator(type=tv.ValidatorType.EQUAL, value=0)]
        step.add_measurement(name="model", value=args.model, hardware_info=hw)
        step.add_measurement(name="precisions", value=",".join(n for n, _ in results) or "none", hardware_info=hw)
        step.add_measurement(name="prompts", value=len(prompts), hardware_info=hw)
        step.add_measurement(name="runs_per_precision", value=args.runs, hardware_info=hw)
        step.add_measurement(name="repeat_runs_differing", value=sum(r["differing"] for _, r in results),
                             validators=eq0, hardware_info=hw)
        step.add_measurement(name="linear_calls_outside_bound", value=sum(r["outside"] for _, r in results),
                             validators=eq0, hardware_info=hw)
        step.add_measurement(name="weight_elements_changed", value=sum(r["changed"] for _, r in results),
                             validators=eq0, hardware_info=hw)
        if args.inject:
            planted, caught = sum(r["injected"] for _, r in results), sum(r["caught"] for _, r in results)
            step.add_measurement(name="injected_runs", value=planted, hardware_info=hw)
            step.add_measurement(name="injected_runs_detected", value=caught,
                                 validators=[tv.Validator(type=tv.ValidatorType.EQUAL, value=planted)],
                                 hardware_info=hw)
        failed = [(n, r) for n, r in results if not r["ok"]]
        if not results:
            ok, verdict, msg = False, "not-run", "no precision could run on this device"
        elif args.inject:
            ok = not failed
            verdict = "injection-self-test-pass" if ok else "injection-self-test-fail"
            msg = f"{len(results)} precision(s), " + ("every planted fault caught" if ok else
                                                      f"failed on {', '.join(n for n, _ in failed)}")
        elif failed:
            ok, verdict = False, failed[0][1]["verdict"]
            msg = f"failed on {', '.join(n for n, _ in failed)}: {failed[0][1]['msg']}"
        else:
            ok, verdict = True, "no-silent-errors"
            msg = f"{len(results)} precision(s) clean on {len(prompts)} prompts x {args.runs} runs"
        all_ok &= ok
        step.add_diagnosis(tv.DiagnosisType.PASS if ok else tv.DiagnosisType.FAIL, verdict=verdict, message=msg,
                           hardware_info=hw)
        step.end(status=tv.TestStatus.COMPLETE)
        print(f"  {'summary':9s} {'PASS' if ok else 'FAIL'}  {verdict}  {msg}")
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
