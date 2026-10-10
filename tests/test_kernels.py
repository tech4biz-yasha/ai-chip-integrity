"""
Tests for kernels.py. Run from the repo root:  python -m pytest -q
All tests run on the CPU so they work on any machine.
"""

import json
import os
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import kernels  # noqa: E402

BASE = ["--device", "cpu", "--size", "128", "--iters", "4"]


def strict_load(path):
    def no_constants(tok):
        raise ValueError(f"non-standard JSON constant {tok}")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line, parse_constant=no_constants) for line in f if line.strip()]


def by_step(objs):
    names, diag, meas = {}, {}, {}
    for o in objs:
        sa = o.get("testStepArtifact")
        if not sa:
            continue
        sid = sa["testStepId"]
        if "testStepStart" in sa:
            names[sid] = sa["testStepStart"]["name"]
        if names.get(sid) == "device_error_counters":      # probe 11's cross-check, tested in test_counters.py
            continue
        if "diagnosis" in sa:
            diag[names[sid]] = sa["diagnosis"]["verdict"]
        if "measurement" in sa:
            meas.setdefault(names[sid], {})[sa["measurement"]["name"]] = sa["measurement"]["value"]
    return diag, meas


@pytest.fixture(autouse=True)
def clear_hook():
    kernels._FAULT_HOOK = None
    yield
    kernels._FAULT_HOOK = None


def test_all_kernels_all_precisions_pass(tmp_path):
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 0
    assert len(diag) == 15
    for name, v in diag.items():
        assert v == "no-silent-errors", name
        assert meas[name]["reference_check_failed_runs"] == 0
        assert meas[name]["repeat_check_failed_runs"] == 0
        assert 0 < meas[name]["worst_error_over_bound_ratio"] <= 1.0, name


def test_bounds_are_tight_for_fp32():
    """An fp32 bound that is loose by orders of magnitude would hide real faults."""
    for name in kernels.KERNELS:
        _, ref, bound = kernels.make_case(name, torch.float32, 1, 128)
        rel = bound.max() / ref.abs().max()
        # attention carries the worst-case QK^T rounding through softmax, so its bound is wider
        assert rel < (1e-3 if name == "attention" else 1e-4), (name, float(rel))


@pytest.mark.parametrize("seed", range(6))
def test_injected_bit_flips_are_all_caught(tmp_path, seed):
    out = tmp_path / "k.jsonl"
    rc = kernels.main(["--device", "cpu", "--size", "64", "--iters", "6", "--inject", "5",
                       "--seed", str(seed), "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 0
    for name, v in diag.items():
        assert v == "injection-self-test-pass", name
        assert meas[name]["injected_runs_detected"] == 5


def test_intermittent_fault_is_silent_data_corruption(tmp_path):
    def hook(i, c):
        if i == 2:
            c.view(-1)[9] += 0.5
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "gelu", "--dtypes", "fp32", "--out", str(out)])
    diag, _ = by_step(strict_load(out))
    assert rc == 1
    assert diag["gelu_fp32"] == "silent-data-corruption"


def test_one_ulp_intermittent_change_is_still_caught(tmp_path):
    def hook(i, c):
        if i == 3:
            v = c.view(torch.int32).view(-1)
            v[5] = v[5] ^ 1
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "softmax", "--dtypes", "fp32", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 1
    assert meas["softmax_fp32"]["reference_check_failed_runs"] == 0
    assert meas["softmax_fp32"]["repeat_check_failed_runs"] == 1
    assert diag["softmax_fp32"] == "silent-data-corruption"


def test_systematic_error_is_outside_bound(tmp_path):
    def hook(i, c):
        c.view(-1)[0] += 0.25
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "layernorm", "--dtypes", "fp32", "--out", str(out)])
    diag, _ = by_step(strict_load(out))
    assert rc == 1
    assert diag["layernorm_fp32"] == "outside-error-bound"


def test_low_precision_internal_math_is_outside_bound(tmp_path):
    """Softmax computed in fp16 internally violates the FP32-accumulation assumption and must be reported."""
    def hook(i, c):
        # replace the output with a softmax computed entirely in fp16 (CPU path), then back to fp32
        x = kernels.make_case("softmax", torch.float32, 1234, 128)[0][0]
        c.copy_(torch.softmax(x.half(), dim=1).float())
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "softmax", "--dtypes", "fp32", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 1
    assert diag["softmax_fp32"] == "outside-error-bound"
    assert meas["softmax_fp32"]["worst_error_over_bound_ratio"] > 10


def test_tanh_gelu_is_outside_bound(tmp_path):
    """The tanh approximation differs from erf-GELU by up to about 5e-4: it must not pass as erf-GELU."""
    def hook(i, c):
        x = kernels.make_case("gelu", torch.float32, 1234, 128)[0][0]
        c.copy_(torch.nn.functional.gelu(x, approximate="tanh"))
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "gelu", "--dtypes", "fp32", "--out", str(out)])
    diag, _ = by_step(strict_load(out))
    assert rc == 1
    assert diag["gelu_fp32"] == "outside-error-bound"


def test_nan_output_is_flagged(tmp_path):
    def hook(i, c):
        if i == 1:
            c.view(-1)[0] = float("nan")
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "attention", "--dtypes", "fp32", "--out", str(out)])
    diag, meas = by_step(strict_load(out))
    assert rc == 1
    assert meas["attention_fp32"]["worst_nonfinite_values"] == 1
    assert diag["attention_fp32"] == "silent-data-corruption"


def test_inaccurate_sin_is_outside_bound(tmp_path):
    """A fast sin with 1e-4 error, as fast-math libraries have at large angles, must not pass as a correct RoPE."""
    def hook(i, c):
        (x, theta), _, _ = kernels.case_rope(128, torch.float32, 1234)
        half = kernels.ROPE_DIM // 2
        emb = torch.cat((theta, theta), dim=-1)
        rot = torch.cat((-x[:, half:], x[:, :half]), dim=-1)
        c.copy_(x * emb.cos() + rot * (emb.sin() + 1e-4))
    kernels._FAULT_HOOK = hook
    out = tmp_path / "k.jsonl"
    rc = kernels.main(BASE + ["--kernels", "rope", "--dtypes", "fp32", "--out", str(out)])
    diag, _ = by_step(strict_load(out))
    assert rc == 1
    assert diag["rope_fp32"] == "outside-error-bound"


def test_rope_angles_reach_every_position():
    """Angles are position times frequency, so the largest is rows - 1 radians: range reduction is exercised."""
    (_, theta), _, _ = kernels.case_rope(4096, torch.float32, 1)
    assert theta.dtype == torch.float32
    assert float(theta.max()) == 4095.0


def test_function_check_measures_the_device_library(tmp_path):
    for name in ("softmax", "gelu", "rope"):
        args = kernels.make_case(name, torch.float32, 1, 128)[0]
        label, err, allowance = kernels.function_check(name, args, torch.device("cpu"))
        assert 0.0 <= err < allowance, (name, label, err)
    out = tmp_path / "k.jsonl"
    kernels.main(BASE + ["--kernels", "rope", "--dtypes", "fp32", "--out", str(out)])
    _, meas = by_step(strict_load(out))
    assert "device_sincos_abs_worst_error" in meas["rope_fp32"]


def test_bad_args_rejected(tmp_path):
    with pytest.raises(SystemExit):
        kernels.main(BASE + ["--kernels", "conv", "--out", str(tmp_path / "k.jsonl")])


@pytest.mark.skipif(not os.environ.get("OCP_SCHEMA_DIR"), reason="set OCP_SCHEMA_DIR to ocp-diag-core/json_spec/output")
def test_output_matches_official_ocp_schema(tmp_path):
    jsonschema = pytest.importorskip("jsonschema")
    referencing = pytest.importorskip("referencing")
    schema_dir = pathlib.Path(os.environ["OCP_SCHEMA_DIR"])
    resources = []
    for p in schema_dir.glob("*.json"):
        doc = json.loads(p.read_text())
        resources.append((doc["$id"], referencing.Resource.from_contents(doc)))
    registry = referencing.Registry().with_resources(resources)
    validator = jsonschema.Draft202012Validator(json.loads((schema_dir / "root.json").read_text()), registry=registry)
    out = tmp_path / "k.jsonl"
    kernels.main(BASE + ["--inject", "2", "--out", str(out)])
    for obj in strict_load(out):
        validator.validate(obj)


def attention_inputs(dtype=torch.float32):
    g = torch.Generator().manual_seed(3)
    return tuple(torch.randn(4, 32, 64, generator=g).to(dtype) for _ in range(3))


def test_cuda_attention_path_adds_a_batch_and_records_the_kernel(monkeypatch):
    """The path NVIDIA GPUs take: fused kernels need 4-D inputs, which the first H100 run showed when every
    fp16 and bf16 attention step was refused with 3-D inputs. Forced onto the CPU, it must return the same
    shape and values as the default kernel and record which kernel ran."""
    from torch.nn.attention import SDPBackend
    q, k, v = attention_inputs()
    monkeypatch.setattr(kernels, "attention_backends",
                        lambda dtype: [("efficient", SDPBackend.EFFICIENT_ATTENTION), ("math", SDPBackend.MATH)])
    kernels.ATTENTION_BACKEND.clear()
    out = kernels.attention(q, k, v, choose=True)
    assert out.shape == q.shape
    assert torch.allclose(out, torch.nn.functional.scaled_dot_product_attention(q, k, v), atol=1e-6)
    assert kernels.ATTENTION_BACKEND[torch.float32] in ("efficient", "math")


def test_cuda_attention_path_refuses_loudly_when_no_kernel_fits(monkeypatch):
    """With only a kernel that cannot run, the step must be refused with every reason, never silently fall
    back to an unchecked path."""
    from torch.nn.attention import SDPBackend
    q, k, v = attention_inputs()
    monkeypatch.setattr(kernels, "attention_backends", lambda dtype: [("cudnn", SDPBackend.CUDNN_ATTENTION)]
                        if hasattr(SDPBackend, "CUDNN_ATTENTION") else [("efficient", SDPBackend.EFFICIENT_ATTENTION)])
    with pytest.raises(RuntimeError, match="no allowed attention kernel"):
        kernels.attention(q, k, v, choose=True)


def test_fused_only_for_half_precisions():
    names = lambda dt: [n for n, _ in kernels.attention_backends(dt)]
    assert "math" not in names(torch.float16) and "math" not in names(torch.bfloat16)
    assert names(torch.float32)[-1] == "math" and names(torch.float16)[0] == "flash"


def test_attention_step_records_its_kernel(tmp_path):
    out = tmp_path / "k.jsonl"
    kernels.main(["--device", "cpu", "--kernels", "attention", "--iters", "3", "--dtypes", "fp32", "--out", str(out)])
    text = out.read_text()
    assert '"name": "attention_backend", "value": "default"' in text
