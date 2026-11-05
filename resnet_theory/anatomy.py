"""Weight-level anatomy of pretrained transformer blocks.

- Norm gains and raw MLP weight sizes (what decoupled weight decay acts on).
- MLP preactivation statistics.
- Absolute write sizes of the attention and MLP branches against the stream norm.
- The Jacobian of the MLP branch h -> mlp(norm(h)) at sampled tokens (its operator norm
  is a lower bound for the Lipschitz constant of the branch) and its linearisation error.
- The MLP cost after minimising over the rescaling symmetries of a SwiGLU block, which
  leave the function unchanged (rebalancing).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from resnet_theory.open_weights import Adapter

EPS = 1e-12


def _gain_stats(ln: torch.nn.Module) -> dict[str, float]:
    g = ln.weight.detach().double().abs()
    return {"median": float(g.median()), "rms": float(g.square().mean().sqrt()), "max": float(g.max())}


@torch.no_grad()
def _capture_preacts(model, adapter: Adapter, input_ids: torch.Tensor, skip_first: int = 1):
    """MLP preactivations [N, m] per (layer, name) and attention writes [N, d] per layer."""
    pre: dict[tuple[int, str], list[torch.Tensor]] = {}
    attn: dict[int, list[torch.Tensor]] = {}
    handles = []

    def rows(x: torch.Tensor) -> torch.Tensor:
        return x[:, skip_first:].reshape(-1, x.shape[-1]).detach().to(torch.float16 if x.shape[-1] > 2048 else torch.float32)

    for layer, block in enumerate(adapter.blocks(model)):
        for name, module in adapter.preacts(block).items():
            handles.append(module.register_forward_hook(
                lambda m, a, out, key=(layer, name): pre.setdefault(key, []).append(rows(out))))
        handles.append(adapter.attn(block).register_forward_hook(
            lambda m, a, out, key=layer: attn.setdefault(key, []).append(rows(out[0] if isinstance(out, tuple) else out))))
    try:
        for start in range(0, input_ids.shape[0], 4):
            model(input_ids[start:start + 4], use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return {k: torch.cat(v).float() for k, v in pre.items()}, {k: torch.cat(v) for k, v in attn.items()}


def _preact_stats(z: torch.Tensor, prefix: str) -> dict[str, float]:
    return {
        f"{prefix}_rms": float(z.square().mean().sqrt()),
        f"{prefix}_frac_lt_0.5": float((z.abs() < 0.5).float().mean()),
        f"{prefix}_frac_gt_2": float((z.abs() > 2.0).float().mean()),
    }


def _jacobian_stats(adapter: Adapter, block, H: torch.Tensor, radii: tuple[float, ...], directions: int, seed: int):
    """Jacobian and linearisation error of the MLP branch h -> mlp(norm(h)) at tokens H [n, d]."""
    ln, mlp = adapter.mlp_ln(block), adapter.mlp(block)

    def f(h: torch.Tensor) -> torch.Tensor:
        return mlp(ln(h[None, None]))[0, 0]

    gen = torch.Generator().manual_seed(seed)
    ops, curv = [], {r: [] for r in radii}
    for h in H:
        h = h.float()
        J = torch.autograd.functional.jacobian(f, h)
        ops.append(float(torch.linalg.svdvals(J.double())[0]))
        with torch.no_grad():
            f0 = f(h)
            for r in radii:
                for _ in range(directions):
                    u = torch.randn(h.shape, generator=gen)
                    delta = r * h.norm() * u / u.norm()
                    df = f(h + delta) - f0
                    curv[r].append(float((df - J @ delta).norm() / (df.norm() + EPS)))
    return {"mlp_jac_op_median": float(np.median(ops)),
            **{f"mlp_linearisation_error_r{r}": float(np.median(curv[r])) for r in radii}}


def rebalanced_swiglu_cost(g, W_gate, W_up, W_down, iterations: int = 50) -> tuple[float, float]:
    """Actual and symmetry-minimal 0.5 * (|g|^2 + |W_gate|^2 + |W_up|^2 + |W_down|^2).

    The function of a SwiGLU block is unchanged under (i) gain g_j -> c_j g_j with the
    j-th input columns of W_gate and W_up divided by c_j, and (ii) the k-th row of W_up
    times a_k with the k-th column of W_down divided by a_k. Alternating exact
    minimisation over the two families, each step in closed form.
    """
    g2 = g.double().square()
    cols_g = W_gate.double().square().sum(0)
    up = W_up.double().square()
    rows_down = W_down.double().square().sum(0)
    actual = 0.5 * float(g2.sum() + cols_g.sum() + up.sum() + rows_down.sum())
    c = torch.ones_like(g2)
    a = torch.ones_like(rows_down)
    for _ in range(iterations):
        cols = cols_g + (up * a[:, None] ** 2).sum(0)
        c = (cols / g2.clamp_min(1e-30)) ** 0.25
        up_rows = (up / c[None, :] ** 2).sum(1)
        a = (rows_down / up_rows.clamp_min(1e-30)) ** 0.25
    up_scaled = up * a[:, None] ** 2 / c[None, :] ** 2
    minimal = 0.5 * float((g2 * c**2).sum() + (cols_g / c**2).sum() + up_scaled.sum() + (rows_down / a**2).sum())
    return actual, minimal


def layer_anatomy(
    model,
    adapter: Adapter,
    input_ids: torch.Tensor,
    traj: dict[str, torch.Tensor],
    keep: torch.Tensor,
    preact_sequences: int = 4,
    jacobian_tokens: int = 32,
    radii: tuple[float, ...] = (0.01, 0.05, 0.2),
    directions: int = 2,
) -> pd.DataFrame:
    """One row per block. ``traj`` and ``keep`` come from ``residual_trajectory`` and
    ``massive_token_mask`` on the same ``input_ids``."""
    T = input_ids.shape[1] - 1
    pre, attn = _capture_preacts(model, adapter, input_ids[:preact_sequences])
    keep_pre = keep[: preact_sequences * T]
    candidates = torch.nonzero(keep).flatten()
    jac_idx = candidates[torch.randperm(len(candidates), generator=torch.Generator().manual_seed(0))[:jacobian_tokens]]

    rows = []
    blocks = adapter.blocks(model)
    for layer, block in enumerate(blocks):
        h = traj["h"][layer][keep].double()
        mlp_out = traj["mlp_out"][layer][keep].double()
        a = attn[layer][keep_pre].double()
        h_pre = traj["h"][layer][: preact_sequences * T][keep_pre].double()
        raw = adapter.raw_mlp(block)
        row: dict[str, Any] = {
            "layer": layer,
            "stream_norm_median": float(h.norm(dim=1).median()),
            "attn_write_rel": float((a.norm(dim=1) / h_pre.norm(dim=1)).median()),
            "mlp_write_rel": float((mlp_out.norm(dim=1) / h.norm(dim=1)).median()),
            "raw_mlp_cost": 0.5 * float(sum(W.detach().double().square().sum() for W in raw)),
        }
        for tag, ln in (("attn_gain", adapter.attn_ln(block)), ("mlp_gain", adapter.mlp_ln(block))):
            row.update({f"{tag}_{k}": v for k, v in _gain_stats(ln).items()})
        for name in adapter.preacts(block):
            row.update(_preact_stats(pre[(layer, name)][keep_pre], name))
        row.update(_jacobian_stats(adapter, block, traj["mlp_in"][layer][jac_idx], radii, directions, seed=layer))
        if adapter.family == "llama":
            mlp = block.mlp
            actual, minimal = rebalanced_swiglu_cost(adapter.mlp_ln(block).weight.detach(), mlp.gate_proj.weight.detach(),
                                                     mlp.up_proj.weight.detach(), mlp.down_proj.weight.detach())
            col_norm = torch.sqrt(mlp.gate_proj.weight.detach().double().square().sum(0)
                                  + mlp.up_proj.weight.detach().double().square().sum(0))
            row.update({"mlp_cost_with_gain": actual, "mlp_cost_rebalanced": minimal,
                        "removable_share": 1.0 - minimal / actual,
                        "gain_over_column_norm_median": float((adapter.mlp_ln(block).weight.detach().double().abs() / col_norm).median())})
        rows.append(row)
    return pd.DataFrame(rows)
