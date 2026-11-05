"""Depth sweep: train one configuration for several depths and seeds, then diagnose.

Every run writes to ``<output_dir>/<name>/runs/L<depth>_seed<seed>/``:
``model.pt`` (state dict and metadata), ``layers.csv``, ``sections.csv``,
``summary.json`` and ``train_log.csv``. Runs that already have a ``summary.json``
are skipped, so a sweep can be split over several processes and resumed.
``collect`` concatenates the runs into ``run_summary.csv``, ``layer_metrics.csv``
and ``section_metrics.csv`` in the sweep directory.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml

from resnet_theory.data import load_magic
from resnet_theory.diagnostics import diagnose
from resnet_theory.model import ResidualNet, load_state_dict
from resnet_theory.train import evaluate, seed_everything, train

DEFAULT_DIAGNOSTICS = {
    "tracked_points": 128,  # training and validation points whose trajectories are recorded
    "pairwise_points": 128,
    "affine_ridge": 1.0e-8,
    "probe_radii": [0.01, 0.03, 0.1],
    "probes_per_point": 4,
    "tau_rel": 0.03,
    "tau_probe_rel": 0.05,
    "min_section_fraction": 0.1,  # minimal section length max(3, round(0.1 L))
    "jump_score_threshold": 2.5,
    "dead_block_threshold": 1.0e-3,
}
DEFAULT_OPTIMIZER = {
    "adam_steps": 3000,
    "adam_lr": 1.0e-3,
    "warmup_steps": 100,
    "min_lr_ratio": 0.05,
    "lbfgs_steps": 2000,
    "lbfgs_history": 100,
    "lbfgs_tolerance_grad": 1.0e-7,
    "lbfgs_tolerance_change": 1.0e-9,
}


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open() as f:
        cfg = yaml.safe_load(f)
    cfg["optimizer"] = {**DEFAULT_OPTIMIZER, **cfg.get("optimizer", {})}
    cfg["diagnostics"] = {**DEFAULT_DIAGNOSTICS, **cfg.get("diagnostics", {})}
    cfg.setdefault("output_dir", "outputs")
    return cfg


def sweep_dir(cfg: dict[str, Any]) -> Path:
    return Path(cfg["output_dir"]) / cfg["name"]


def run_name(depth: int, seed: int) -> str:
    return f"L{depth}_seed{seed}"


def _build(cfg: dict[str, Any], depth: int, dim: int) -> ResidualNet:
    return ResidualNet(dim, depth, int(cfg["model"]["width"]), cfg["model"]["init"])


def _diagnose_and_save(cfg, model, data, depth, seed, final_eval, out: Path, extra: dict[str, Any]) -> dict[str, Any]:
    n = int(cfg["diagnostics"]["tracked_points"])
    layers, sections, diag = diagnose(model, data.X_train[:n], data.X_val[:n], cfg["diagnostics"], seed)
    summary = {
        "sweep": cfg["name"], "run": run_name(depth, seed), "L": depth, "seed": seed,
        "width": int(cfg["model"]["width"]), "lambda_res": float(cfg["objective"]["lambda_res"]),
        "lambda_out": float(cfg["objective"]["lambda_out"]),
        **{f"train_{k}": v for k, v in final_eval["train"].items()},
        **{f"val_{k}": v for k, v in final_eval["val"].items()},
        **diag, **extra,
    }
    for df in (layers, sections):
        df.insert(0, "seed", seed)
        df.insert(0, "L", depth)
        df.insert(0, "run", run_name(depth, seed))
    layers.to_csv(out / "layers.csv", index=False)
    sections.to_csv(out / "sections.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def run_one(cfg: dict[str, Any], depth: int, seed: int) -> dict[str, Any]:
    """Train and diagnose one network. The seed sets the data split and the initialisation."""
    out = sweep_dir(cfg) / "runs" / run_name(depth, seed)
    out.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)
    data = load_magic(cfg["data"]["path"], float(cfg["data"]["val_fraction"]), seed)
    model = _build(cfg, depth, data.dim)
    lam_res, lam_out = float(cfg["objective"]["lambda_res"]), float(cfg["objective"]["lambda_out"])
    start = time.time()
    log = train(model, data.X_train, data.y_train, lam_res, lam_out, cfg["optimizer"])
    seconds = time.time() - start
    pd.DataFrame(log).to_csv(out / "train_log.csv", index=False)
    torch.save({"state_dict": model.state_dict(), "depth": depth, "seed": seed, "dim": data.dim}, out / "model.pt")
    final_eval = {"train": evaluate(model, data.X_train, data.y_train), "val": evaluate(model, data.X_val, data.y_val)}
    return _diagnose_and_save(cfg, model, data, depth, seed, final_eval, out, {"train_seconds": seconds})


def rediagnose(cfg: dict[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Recompute all tables of a trained run from its ``model.pt``, without training."""
    run_dir = Path(run_dir)
    state = torch.load(run_dir / "model.pt", map_location="cpu")
    depth, seed = int(state["depth"]), int(state["seed"])
    old = run_dir / "summary.json"
    extra = {k: v for k, v in json.loads(old.read_text()).items() if k == "train_seconds"} if old.exists() else {}
    seed_everything(seed)
    data = load_magic(cfg["data"]["path"], float(cfg["data"]["val_fraction"]), seed)
    model = _build(cfg, depth, data.dim)
    load_state_dict(model, state["state_dict"])
    final_eval = {"train": evaluate(model, data.X_train, data.y_train), "val": evaluate(model, data.X_val, data.y_val)}
    return _diagnose_and_save(cfg, model, data, depth, seed, final_eval, run_dir, extra)


def run_sweep(cfg: dict[str, Any], depths: list[int] | None = None, seeds: list[int] | None = None) -> Path:
    root = sweep_dir(cfg)
    root.mkdir(parents=True, exist_ok=True)
    with (root / "config.yaml").open("w") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    for depth in depths or cfg["sweep"]["depths"]:
        for seed in seeds or cfg["sweep"]["seeds"]:
            if (root / "runs" / run_name(depth, seed) / "summary.json").exists():
                print(f"{cfg['name']} L={depth} seed={seed}: done, skipped")
                continue
            s = run_one(cfg, int(depth), int(seed))
            print(f"{cfg['name']} L={depth} seed={seed}: val acc {s['val_acc']:.4f}, total cost "
                  f"{s['total_residual_cost']:.3f}, N_eff {s['effective_num_layers']:.2f}, "
                  f"longest affine section {s['best_affine_section_length']} ({s['train_seconds']:.0f}s)")
    collect(root)
    return root


def collect(root: str | Path) -> None:
    root = Path(root)
    runs = sorted((root / "runs").glob("*/summary.json"))
    if not runs:
        return
    summaries = [json.loads(p.read_text()) for p in runs]
    pd.DataFrame(summaries).sort_values(["L", "seed"]).to_csv(root / "run_summary.csv", index=False)
    for name, out in (("layers.csv", "layer_metrics.csv"), ("sections.csv", "section_metrics.csv")):
        tables = [pd.read_csv(p.parent / name) for p in runs]
        tables = [t for t in tables if not t.empty]
        if tables:
            pd.concat(tables, ignore_index=True).sort_values(["L", "seed"], kind="stable").to_csv(root / out, index=False)
