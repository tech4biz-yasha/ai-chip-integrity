"""
The gpu-sdc self-test faults must follow Tung et al., "The Anatomy of Silent Data Corruption: GPU Error Pattern
Study and Modeling Guidance" (DSN 2026, arXiv:2605.04213), and every one of them must be caught.
Run from the repo root:  python -m pytest -q
"""

import json
import random

import pytest
import torch

from chip_integrity import screen

VIEWS = {torch.float32: (torch.int32, 32), torch.float16: (torch.int16, 16),
         torch.bfloat16: (torch.int16, 16), torch.int32: (torch.int32, 32)}


def plant_many(dtype, trials, seed=7, zeros=False):
    """Plant `trials` faults into fresh tensors; return (before, after) bit views for each plant."""
    int_view, bits = VIEWS[dtype]
    rng, gen, pairs = random.Random(seed), torch.Generator().manual_seed(seed), []
    for _ in range(trials):
        if zeros:
            c = torch.zeros(300, dtype=dtype)
        elif dtype.is_floating_point:
            c = (torch.rand(300, generator=gen) + 0.5).to(dtype)        # no zeros, so every outcome shows
        else:
            c = torch.randint(1, 1 << 20, (300,), generator=gen, dtype=dtype)
        before = c.view(int_view).clone()
        screen.plant_gpu_sdc(c, int_view, bits, rng)
        pairs.append((before, c.view(int_view).clone()))
    return pairs


def popcount(x, bits):
    return bin(int(x) & ((1 << bits) - 1)).count("1")


def test_outcome_mix_is_the_papers_and_sums_to_100():
    mix = dict(screen.SDC_OUTCOMES)
    assert mix == {"nullified": 50.68, "nan": 0.39, "+inf": 0.49, "-inf": 0.13, "bit-flips": 48.31}
    assert sum(mix.values()) == pytest.approx(100.0)


@pytest.mark.parametrize("dtype", list(VIEWS))
@pytest.mark.parametrize("zeros", [False, True])
def test_every_plant_changes_at_least_one_element(dtype, zeros):
    for before, after in plant_many(dtype, 200, zeros=zeros):
        assert int((before != after).sum()) >= 1


@pytest.mark.parametrize("dtype", list(VIEWS))
def test_faults_sit_in_one_warp_aligned_chunk_at_even_spacing(dtype):
    for before, after in plant_many(dtype, 300):
        hit = (before != after).nonzero().flatten().tolist()
        assert len({i // screen.WARP for i in hit}) == 1             # one 32-element warp-aligned chunk
        assert all((b - a) % 2 == 0 for a, b in zip(hit, hit[1:]))   # periods are 2, 4, 8 or 16
        assert len(hit) <= screen.WARP // min(screen.SDC_PERIODS)


def test_outcome_shares_match_the_measurements():
    counts = {"nullified": 0, "special": 0, "bit-flips": 0}
    for before, after in plant_many(torch.float32, 1500):
        for b, a in zip(before[before != after].tolist(), after[before != after].tolist()):
            value = torch.tensor(a, dtype=torch.int32).view(torch.float32).item()
            counts["nullified" if a == 0 else "special" if not torch.isfinite(torch.tensor(value)) else "bit-flips"] += 1
    total = sum(counts.values())
    assert total > 5000
    assert counts["nullified"] / total == pytest.approx(0.5068, abs=0.03)
    assert counts["bit-flips"] / total == pytest.approx(0.4831, abs=0.03)
    assert counts["special"] / total < 0.03


def test_bit_flips_are_mostly_multi_bit_and_favour_low_bits():
    flips, positions = [], []
    for before, after in plant_many(torch.float32, 1500):
        for b, a in zip(before[before != after].tolist(), after[before != after].tolist()):
            value = torch.tensor(a, dtype=torch.int32).view(torch.float32).item()
            if a != 0 and torch.isfinite(torch.tensor(value)):
                x = (b ^ a) & 0xFFFFFFFF
                flips.append(popcount(x, 32))
                positions += [p for p in range(32) if x >> p & 1]
    single = sum(f == 1 for f in flips) / len(flips)
    assert single < 0.40                                            # the paper: single-bit flips are under 40%
    assert single == pytest.approx(screen.SINGLE_BIT_SHARE, abs=0.04)
    assert max(flips) <= screen.MAX_FLIPPED_BITS
    assert sum(positions) / len(positions) < 14                     # low bits flip more often than high bits


def test_integer_outputs_get_only_zeros_and_bit_flips():
    for before, after in plant_many(torch.int32, 400):
        for b, a in zip(before[before != after].tolist(), after[before != after].tolist()):
            assert a == 0 or 1 <= popcount(b ^ a, 32) <= screen.MAX_FLIPPED_BITS


def test_mixed_alternates_single_bits_and_gpu_sdc_faults():
    runs = {12, 3, 20, 7, 9}
    assert screen.fault_models(runs, "mixed") == {3: "bit", 7: "gpu-sdc", 9: "bit", 12: "gpu-sdc", 20: "bit"}
    assert set(screen.fault_models(runs, "gpu-sdc").values()) == {"gpu-sdc"}
    assert set(screen.fault_models(runs, "bit").values()) == {"bit"}


@pytest.mark.parametrize("model,expected", [("gpu-sdc", 5), ("mixed", 2), ("bit", 0)])
def test_compute_self_test_catches_every_gpu_sdc_fault(tmp_path, model, expected):
    out = tmp_path / "s.jsonl"
    rc = screen.main(["--device", "cpu", "--size", "48", "--iters", "10", "--dtypes", "fp32,fp16,bf16,int8",
                      "--inject", "5", "--inject-model", model, "--out", str(out)])
    assert rc == 0
    meas = [json.loads(line)["testStepArtifact"]["measurement"] for line in out.read_text().splitlines()
            if '"measurement"' in line]
    value = lambda name: [m["value"] for m in meas if m["name"] == name]
    assert value("injected_runs") == value("injected_runs_detected") == [5] * 4
    assert value("gpu_sdc_faults_planted") == value("gpu_sdc_faults_detected") == [expected] * 4
