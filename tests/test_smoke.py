"""Fast checks of the sweep pipeline and the diagnostics (the first run downloads MAGIC)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import torch

from resnet_theory.anatomy import rebalanced_swiglu_cost
from resnet_theory.diagnostics import diagnose, greedy_affine_scan
from resnet_theory.figures import make_figures
from resnet_theory.model import ResidualNet, block_cost, residual_cost
from resnet_theory.sweep import load_config, rediagnose, run_sweep

ROOT = Path(__file__).resolve().parents[1]


def tiny_config(tmp_path: Path, name: str, block_scale: float = 0.3) -> dict:
    cfg = load_config(ROOT / "configs" / "smoke.yaml")
    cfg.update(name=name, output_dir=str(tmp_path))
    cfg["data"]["path"] = str(ROOT / "data" / "magic04.data")
    cfg["model"]["init"]["block_scale"] = block_scale
    cfg["sweep"] = {"depths": [2, 4], "seeds": [0]}
    cfg["optimizer"].update(adam_steps=30, warmup_steps=5, lbfgs_steps=10)
    return cfg


def test_cost_definition():
    torch.manual_seed(0)
    model = ResidualNet(10, 3, 6, {"block_scale": 0.3, "head_scale": 0.3})
    block = model.blocks[0]
    expected = 0.5 * sum(float(p.detach().square().sum()) for p in block.parameters())
    assert abs(float(block_cost(block)) - expected) < 1e-6
    assert abs(float(residual_cost(model)) - sum(float(block_cost(b)) for b in model.blocks)) < 1e-5


def test_depth_scaled_init_variance():
    torch.manual_seed(0)
    model = ResidualNet(10, 64, 6, {"block_scale": 1.0, "head_scale": 1.0, "depth_scaling": True})
    std = torch.cat([b.U.weight.flatten() for b in model.blocks]).std()
    assert abs(float(std) - (1.0 / (10 * 64)) ** 0.5) < 0.003


def test_sweep_runs_and_is_deterministic(tmp_path):
    first = run_sweep(tiny_config(tmp_path, "a"))
    second = run_sweep(tiny_config(tmp_path, "b"))
    s1, s2 = pd.read_csv(first / "run_summary.csv"), pd.read_csv(second / "run_summary.csv")
    assert list(s1["L"]) == [2, 4]
    assert (s1["total_residual_cost"] > 0).all()
    pd.testing.assert_series_equal(s1["total_residual_cost"], s2["total_residual_cost"])
    layers = pd.read_csv(first / "layer_metrics.csv")
    assert len(layers) == 2 + 4
    run = first / "runs" / "L4_seed0"
    before = json.loads((run / "summary.json").read_text())
    after = rediagnose(tiny_config(tmp_path, "a"), run)
    assert after["val_acc"] == before["val_acc"]
    assert after["total_residual_cost"] == before["total_residual_cost"]

    make_figures([("small", first), ("other", second)], tmp_path / "figures")
    for stem in ("layer_cost_profiles", "depth_scaling", "accuracy"):
        assert (tmp_path / "figures" / f"{stem}.png").exists()
    table = pd.read_csv(tmp_path / "figures" / "depth_summary.csv")
    assert set(table["init"]) == {"small", "other"}


def test_identity_network_is_one_affine_section():
    torch.manual_seed(0)
    model = ResidualNet(10, 8, 6, {"block_scale": 0.0, "head_scale": 0.3})
    X = torch.randn(64, 10)
    cfg = load_config(ROOT / "configs" / "smoke.yaml")["diagnostics"]
    layers, sections, summary = diagnose(model, X, torch.randn(64, 10), cfg, seed=0)
    assert summary["best_affine_section_length"] == 8
    assert summary["num_dead_blocks"] == 8
    assert (layers["c_l"] == 0).all()


def test_greedy_scan_splits_at_nonaffine_flagged_block():
    # Block 5 is flagged and not affine on its own, every section avoiding it is affine.
    def evaluate(start, end):
        ok = not (start <= 5 < end)
        return {"section_start": start, "section_end": end, "length": end - start, "max_jump_score": 0.0,
                "val_relative_rmse": 0.0, "probe_relative_rmse": 0.0,
                "classification": "affine_section" if ok else "non_affine"}

    scores = [0.0] * 12
    scores[5] = 10.0
    cfg = {"min_section_fraction": 0.1, "jump_score_threshold": 2.5}
    sections = greedy_affine_scan(12, scores, cfg, evaluate)
    assert [(s["section_start"], s["section_end"]) for s in sections] == [(0, 5), (6, 12)]


def test_rebalancing_does_not_increase_cost():
    torch.manual_seed(0)
    d, m = 8, 16
    g = torch.rand(d) * 0.2
    actual, minimal = rebalanced_swiglu_cost(g, torch.randn(m, d) * 2, torch.randn(m, d) * 2, torch.randn(d, m))
    assert minimal <= actual
    assert minimal > 0
