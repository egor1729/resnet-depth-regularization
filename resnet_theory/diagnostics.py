"""Per-layer diagnostics and the affine-section scan of a trained network.

All measurements use the trajectories z_0, ..., z_L of a fixed set of tracked training
and validation points.

Affine-section scan. For a section of blocks s, ..., e-1, an affine map z_s -> A z_s + q
is fitted to the training points by least squares, and the relative error

    |Z_e - A Z_s - q 1^T|_F / |Z_e - mean(Z_e)|_F

is measured on training points, validation points and noisy probes around the training
points (the probes are pushed through the real blocks). A section is accepted as affine
if the error is below ``tau_rel`` on training and validation points and below
``tau_probe_rel`` on the probes (``weak_affine_section`` if the probe error is at most
twice that). The scan is greedy: it first splits the network at blocks that are flagged
by a robust outlier score and are not affine on their own, then repeatedly takes the
longest accepted section of the longest remaining segment.
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from resnet_theory.model import ResidualNet, block_cost

EPS = 1e-12
ACCEPTED = ("affine_section", "weak_affine_section")


# ---------------------------------------------------------------------------
# Affine fits


def fit_affine(X: torch.Tensor, Y: torch.Tensor, ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Least squares Y ~ X A^T + q, with a small ridge on A (not on q)."""
    Xd, Yd = X.double(), Y.double()
    X_aug = torch.cat([Xd, torch.ones((Xd.shape[0], 1), dtype=Xd.dtype)], dim=1)
    gram = X_aug.T @ X_aug
    penalty = ridge * torch.eye(gram.shape[0], dtype=gram.dtype)
    penalty[-1, -1] = 0.0
    theta = torch.linalg.solve(gram + penalty, X_aug.T @ Yd)
    return theta[:-1].T, theta[-1]


def affine_errors(X: torch.Tensor, Y: torch.Tensor, A: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    residual = Y.double() - (X.double() @ A.T + q)
    sse = residual.square().sum()
    sst = (Y.double() - Y.double().mean(dim=0, keepdim=True)).square().sum()
    return {
        "rmse": float(torch.sqrt(torch.linalg.norm(residual, dim=1).square().mean())),
        "relative_rmse": float(torch.sqrt(sse) / (torch.sqrt(sst) + EPS)),
    }


# ---------------------------------------------------------------------------
# Per-layer metrics


@torch.no_grad()
def trajectory(model: ResidualNet, X: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model.features(X, return_trajectory=True)


def pairwise_log_distortion(prev: torch.Tensor, nxt: torch.Tensor) -> dict[str, float]:
    """Largest change of log pairwise distances caused by one block."""
    def dist(Z: torch.Tensor) -> torch.Tensor:
        D = torch.cdist(Z.double(), Z.double())
        return 0.5 * (D + D.T)

    D0, D1 = dist(prev), dist(nxt)
    mask = ~torch.eye(D0.shape[0], dtype=torch.bool)
    log_ratio = torch.log((D1[mask] + EPS) / (D0[mask] + EPS))
    return {
        "pairwise_distortion_max": float(torch.abs(log_ratio).max()),
        "pairwise_contraction_max": float(torch.clamp(-log_ratio, min=0).max()),
        "pairwise_expansion_max": float(torch.clamp(log_ratio, min=0).max()),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    finite = np.isfinite(x)
    out = np.zeros_like(x)
    if finite.any():
        med = np.nanmedian(x[finite])
        mad = np.nanmedian(np.abs(x[finite] - med))
        out[finite] = (x[finite] - med) / (mad + EPS)
    return out


def jump_scores(layers: pd.DataFrame) -> np.ndarray:
    """Robust outlier score of each block within one network (sum of robust z-scores of
    log c_l, log max displacement, log alpha_op and the pairwise distortion). It is only
    used to pick candidate split points for the affine-section scan."""
    pieces = []
    for col, log in (("c_l", True), ("disp_max", True), ("alpha_op", True), ("pairwise_distortion_max", False)):
        v = layers[col].to_numpy(dtype=float)
        pieces.append(robust_z(np.log(np.clip(v, EPS, None)) if log else v))
    return np.sum(pieces, axis=0)


@torch.no_grad()
def layer_metrics(model: ResidualNet, traj: torch.Tensor, ridge: float, pairwise_points: int) -> pd.DataFrame:
    L = len(model.blocks)
    rows = []
    for layer, block in enumerate(model.blocks):
        z0, z1 = traj[layer], traj[layer + 1]
        disp = torch.linalg.norm((z1 - z0).double(), dim=1)
        U, V = block.U, block.V
        A, q = fit_affine(z0, z1, ridge)
        c = float(block_cost(block))
        rows.append({
            "layer": layer,
            "p": layer / max(1, L),
            "c_l": c,
            "L_c_l": L * c,
            "cost_U": 0.5 * float(U.weight.square().sum() + U.bias.square().sum()),
            "cost_V": 0.5 * float(V.weight.square().sum() + V.bias.square().sum()),
            "alpha_op": float(torch.linalg.matrix_norm(V.weight, ord=2) * torch.linalg.matrix_norm(U.weight, ord=2)),
            "alpha_frob": float(torch.linalg.norm(V.weight) * torch.linalg.norm(U.weight)),
            "disp_mean": float(disp.mean()),
            "disp_max": float(disp.max()),
            "disp_rms": float(torch.sqrt(disp.square().mean())),
            "one_step_affine_rmse": affine_errors(z0, z1, A, q)["rmse"],
            **pairwise_log_distortion(z0[:pairwise_points], z1[:pairwise_points]),
        })
    df = pd.DataFrame(rows)
    df["jump_score"] = jump_scores(df)
    return df


# ---------------------------------------------------------------------------
# Affine-section scan


def make_probes(base: torch.Tensor, radii: list[float], per_point: int, seed: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(int(seed))
    probes = []
    for radius in radii:
        noise = torch.randn((base.shape[0], per_point, base.shape[1]), generator=gen, dtype=base.dtype)
        probes.append((base[:, None, :] + float(radius) * noise).reshape(-1, base.shape[1]))
    return torch.cat(probes, dim=0)


def classify(train: float, val: float, probe: float, tau_rel: float, tau_probe_rel: float) -> str:
    if train <= tau_rel and val <= tau_rel:
        if probe <= tau_probe_rel:
            return "affine_section"
        if probe <= 2.0 * tau_probe_rel:
            return "weak_affine_section"
        return "training_only_affine"
    if train <= tau_rel:
        return "training_only_affine"
    return "non_affine"


Evaluate = Callable[[int, int], dict[str, Any]]


def greedy_affine_scan(L: int, jump_score: np.ndarray, cfg: dict[str, Any], evaluate: Evaluate) -> list[dict[str, Any]]:
    """Greedy search for long accepted sections; ``evaluate(start, end)`` returns a row
    with at least ``classification``, ``probe_relative_rmse``, ``val_relative_rmse``."""
    if "min_section_length" in cfg:
        min_length = int(cfg["min_section_length"])
    else:
        min_length = max(3, round(float(cfg["min_section_fraction"]) * L))
    min_length = max(1, min(min_length, L))
    max_evaluations = int(cfg.get("max_evaluations", 2048))

    # Split points: flagged blocks that are not affine on their own.
    flagged = [layer for layer in range(L) if jump_score[layer] >= float(cfg["jump_score_threshold"])]
    split = [layer for layer in flagged if evaluate(layer, layer + 1)["classification"] not in ACCEPTED]
    pending, start = [], 0
    for layer in split:
        if layer - start >= min_length:
            pending.append((start, layer))
        start = max(start, layer + 1)
    if L - start >= min_length:
        pending.append((start, L))

    def rank(row: dict[str, Any]):
        return (0 if row["classification"] == "affine_section" else 1, row["probe_relative_rmse"],
                row["val_relative_rmse"], row["max_jump_score"], row["section_start"])

    sections: list[dict[str, Any]] = []
    evaluations = 0
    while pending and evaluations < max_evaluations:
        seg = max(pending, key=lambda s: s[1] - s[0])
        pending.remove(seg)
        best = None
        for length in range(seg[1] - seg[0], min_length - 1, -1):
            accepted = []
            for s in range(seg[0], seg[1] - length + 1):
                if evaluations >= max_evaluations:
                    break
                row = evaluate(s, s + length)
                evaluations += 1
                if row["classification"] in ACCEPTED:
                    accepted.append(row)
            else:
                if accepted:
                    best = sorted(accepted, key=rank)[0]
                    break
                continue
            break  # evaluation budget exhausted
        if best is None:
            continue
        sections.append(best)
        a, b = best["section_start"], best["section_end"]
        if a - seg[0] >= min_length:
            pending.append((seg[0], a))
        if seg[1] - b >= min_length:
            pending.append((b, seg[1]))
    return sorted(sections, key=lambda r: (r["section_start"], r["section_end"], r["classification"]))


def affine_section_scan(
    model: ResidualNet,
    train_traj: torch.Tensor,
    val_traj: torch.Tensor,
    layers: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
) -> pd.DataFrame:
    L = train_traj.shape[0] - 1
    c = layers["c_l"].to_numpy(dtype=float)
    js = layers["jump_score"].to_numpy(dtype=float)
    ridge = float(cfg["affine_ridge"])

    @torch.no_grad()
    def evaluate(start: int, end: int) -> dict[str, Any]:
        A, q = fit_affine(train_traj[start], train_traj[end], ridge)
        train = affine_errors(train_traj[start], train_traj[end], A, q)
        val = affine_errors(val_traj[start], val_traj[end], A, q)
        probe_x = make_probes(train_traj[start], list(cfg["probe_radii"]), int(cfg["probes_per_point"]),
                              seed=seed + 1009 * start + end)
        model.eval()
        probe = affine_errors(probe_x, model.forward_range(start, end, probe_x), A, q)
        s = torch.linalg.svdvals(A)
        return {
            "section_start": start,
            "section_end": end,
            "length": end - start,
            "p_start": start / L,
            "p_end": end / L,
            "section_cost": float(c[start:end].sum()),
            "max_cost": float(c[start:end].max()),
            "max_jump_score": float(js[start:end].max()),
            "train_relative_rmse": train["relative_rmse"],
            "val_relative_rmse": val["relative_rmse"],
            "probe_relative_rmse": probe["relative_rmse"],
            "probe_rmse": probe["rmse"],
            "classification": classify(train["relative_rmse"], val["relative_rmse"], probe["relative_rmse"],
                                       float(cfg["tau_rel"]), float(cfg["tau_probe_rel"])),
            "operator_norm_A": float(s.max()),
            "singular_values_A": ",".join(f"{float(v):.8g}" for v in s),
        }

    return pd.DataFrame(greedy_affine_scan(L, js, cfg, evaluate), columns=SECTION_COLUMNS)


SECTION_COLUMNS = [
    "section_start", "section_end", "length", "p_start", "p_end", "section_cost", "max_cost", "max_jump_score",
    "train_relative_rmse", "val_relative_rmse", "probe_relative_rmse", "probe_rmse", "classification",
    "operator_norm_A", "singular_values_A",
]


# ---------------------------------------------------------------------------
# Summary of one trained network


def cost_concentration(costs: np.ndarray) -> dict[str, float]:
    """Top share, top-3 share, N_eff = (sum c)^2 / sum c^2 and the position of the largest block."""
    c = np.clip(np.asarray(costs, dtype=float), 0.0, None)
    total = float(c.sum())
    if c.size == 0 or total <= 0.0:
        return {k: float("nan") for k in ("top1_cost_share", "top3_cost_share", "effective_num_layers", "argmax_layer_p")}
    ordered = sorted(c, reverse=True)
    return {
        "top1_cost_share": float(ordered[0] / total),
        "top3_cost_share": float(sum(ordered[:3]) / total),
        "effective_num_layers": float(total**2 / float((c**2).sum())),
        "argmax_layer_p": float(int(c.argmax()) / c.size),
    }


def diagnose(model: ResidualNet, X_train: torch.Tensor, X_val: torch.Tensor, cfg: dict[str, Any], seed: int):
    """Layer table, affine-section table and scalar summary of a trained network.

    ``X_train`` and ``X_val`` are the tracked points.
    """
    train_traj = trajectory(model, X_train)
    val_traj = trajectory(model, X_val)
    layers = layer_metrics(model, train_traj, float(cfg["affine_ridge"]), int(cfg["pairwise_points"]))
    sections = affine_section_scan(model, train_traj, val_traj, layers, cfg, seed)

    layers["section_label"] = ""
    for _, row in sections.iterrows():
        in_section = (layers["layer"] >= row["section_start"]) & (layers["layer"] < row["section_end"])
        layers.loc[in_section, "section_label"] = row["classification"]

    c = layers["c_l"].to_numpy(dtype=float)
    summary = {
        "total_residual_cost": float(c.sum()),
        "max_layer_cost": float(c.max()),
        "mean_layer_cost": float(c.mean()),
        **cost_concentration(c),
        "num_dead_blocks": int((c < float(cfg["dead_block_threshold"])).sum()),
        "best_affine_section_length": int(sections["length"].max()) if len(sections) else 0,
        "affine_section_coverage": int(sections["length"].sum()) if len(sections) else 0,
        "num_affine_sections": int(len(sections)),
    }
    return layers, sections, summary
