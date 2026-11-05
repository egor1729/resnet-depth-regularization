"""Diagnostics of pretrained (open-weight) transformers, from forward passes and weights only.

The residual stream h_0, ..., h_L of a pre-norm transformer (h_l is the input of block l,
h_L the output of the last block, before the final norm) is recorded on windows of
WikiText-103 validation text. Each token gives one trajectory. The first position
(attention sink) is dropped, and tokens with massive activations (stream norm above
4 times the layer median at some layer) are excluded.

Measured per block:
- the affine-fit error of the one-block map h_l -> h_{l+1}, normalised by the spread of
  the increment h_{l+1} - h_l (the share of the block's work that no affine map
  explains), fitted on training sequences and evaluated on held-out sequences;
- the contraction and expansion of the fitted map (sums of negative and positive log
  singular values), after dividing out the growth of the median stream norm;
- the folded MLP weight cost (norm gains multiplied into the following matrices);
- absolute stream norms, step sizes and path length (norm escape).

Supported model families: Llama-style SwiGLU models (SmolLM2, PleIAs Baguettotron and
Monad) and GPT-NeoX (Pythia). Needs ``transformers`` and ``datasets``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
import torch

from resnet_theory.diagnostics import fit_affine, greedy_affine_scan, pairwise_log_distortion, robust_z

EPS = 1e-12


# ---------------------------------------------------------------------------
# Model adapters


@dataclass
class MLPWeights:
    """y = W_out phi(W_in x + b_in) + b_out on the normalised input x (gains folded in).
    For SwiGLU blocks y = W_out (silu(W_gate x) * (W_in x))."""

    W_in: torch.Tensor
    b_in: torch.Tensor
    W_out: torch.Tensor
    b_out: torch.Tensor
    W_gate: torch.Tensor | None = None


@dataclass
class Adapter:
    family: str
    blocks: Callable[[torch.nn.Module], list[torch.nn.Module]]
    mlp: Callable[[torch.nn.Module], torch.nn.Module]
    mlp_ln: Callable[[torch.nn.Module], torch.nn.Module]
    attn: Callable[[torch.nn.Module], torch.nn.Module]
    attn_ln: Callable[[torch.nn.Module], torch.nn.Module]
    weights: Callable[[torch.nn.Module], MLPWeights]
    preacts: Callable[[torch.nn.Module], dict[str, torch.nn.Module]]  # layers producing MLP preactivations
    raw_mlp: Callable[[torch.nn.Module], list[torch.Tensor]]  # MLP matrices before folding


def _fold_ln(ln: torch.nn.Module, W: torch.Tensor, b: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
    """Fold y = W (g * x_hat + beta) + b into W' x_hat + b'. W: [out, in]."""
    g = ln.weight.detach().double()
    bias = getattr(ln, "bias", None)
    beta = bias.detach().double() if bias is not None else torch.zeros_like(g)
    W = W.detach().double()
    b0 = b.detach().double() if b is not None else torch.zeros(W.shape[0], dtype=torch.float64)
    return W * g[None, :], W @ beta + b0


def _llama_weights(block: torch.nn.Module) -> MLPWeights:
    mlp, ln = block.mlp, block.post_attention_layernorm
    W_in, b_in = _fold_ln(ln, mlp.up_proj.weight, mlp.up_proj.bias)
    W_gate, _ = _fold_ln(ln, mlp.gate_proj.weight, mlp.gate_proj.bias)
    W_out = mlp.down_proj.weight.detach().double()
    b_out = mlp.down_proj.bias.detach().double() if mlp.down_proj.bias is not None else torch.zeros(W_out.shape[0], dtype=torch.float64)
    return MLPWeights(W_in, b_in, W_out, b_out, W_gate=W_gate)


def _neox_weights(block: torch.nn.Module) -> MLPWeights:
    W_in, b_in = _fold_ln(block.post_attention_layernorm, block.mlp.dense_h_to_4h.weight, block.mlp.dense_h_to_4h.bias)
    return MLPWeights(W_in, b_in, block.mlp.dense_4h_to_h.weight.detach().double(), block.mlp.dense_4h_to_h.bias.detach().double())


_LLAMA = Adapter(
    "llama", lambda m: list(m.model.layers), lambda b: b.mlp, lambda b: b.post_attention_layernorm,
    lambda b: b.self_attn, lambda b: b.input_layernorm, _llama_weights,
    preacts=lambda b: {"gate": b.mlp.gate_proj, "up": b.mlp.up_proj},
    raw_mlp=lambda b: [b.mlp.gate_proj.weight, b.mlp.up_proj.weight, b.mlp.down_proj.weight],
)
ADAPTERS: dict[str, Adapter] = {
    "llama": _LLAMA,
    "gpt_neox": Adapter(
        "gpt_neox", lambda m: list(m.gpt_neox.layers), lambda b: b.mlp, lambda b: b.post_attention_layernorm,
        lambda b: b.attention, lambda b: b.input_layernorm, _neox_weights,
        preacts=lambda b: {"in": b.mlp.dense_h_to_4h},
        raw_mlp=lambda b: [b.mlp.dense_h_to_4h.weight, b.mlp.dense_4h_to_h.weight],
    ),
}


def load_model(name: str, revision: str | None = None) -> tuple[torch.nn.Module, Any, Adapter]:
    """Load a Hugging Face causal LM in float32. ``revision="init"`` builds a freshly
    initialised model with the same configuration (default Hugging Face initialisation,
    seed 0), a stand-in for a step-0 checkpoint when none is published."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if revision == "init":
        torch.manual_seed(0)
        model = AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(name), dtype=torch.float32)
        revision = None
    else:
        kwargs = {"revision": revision} if revision else {}
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32, **kwargs)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(name, **({"revision": revision} if revision else {}))
    family = model.config.model_type
    if family not in ADAPTERS:
        raise ValueError(f"Unsupported model family {family!r}; supported: {sorted(ADAPTERS)}")
    return model, tokenizer, ADAPTERS[family]


def load_token_batches(tokenizer: Any, num_sequences: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    """Random non-overlapping windows of the WikiText-103 validation set, in text order.
    The same seed selects the same windows for every model, up to tokenisation."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="validation")
    ids = tokenizer("".join(row["text"] for row in ds), return_tensors="pt", truncation=False)["input_ids"][0]
    order = torch.randperm(ids.numel() // seq_len, generator=torch.Generator().manual_seed(seed))[:num_sequences]
    return torch.stack([ids[i * seq_len:(i + 1) * seq_len] for i in order.sort().values])


# ---------------------------------------------------------------------------
# Forward passes with hooks


def _first(output: Any) -> torch.Tensor:
    return output[0] if isinstance(output, (tuple, list)) else output


@torch.no_grad()
def residual_trajectory(
    model: torch.nn.Module,
    adapter: Adapter,
    input_ids: torch.Tensor,
    batch_size: int = 8,
    perturb: tuple[int, float, int] | None = None,
    skip_first: int = 1,
) -> dict[str, torch.Tensor]:
    """Residual stream ``h`` [L+1, N, d], MLP inputs ``mlp_in`` [L, N, d] (stream entering
    the MLP norm) and MLP outputs ``mlp_out`` [L, N, d], tokens flattened over (sequence,
    position) without the first ``skip_first`` positions.

    ``perturb=(layer, radius, seed)`` adds isotropic noise of relative size ``radius`` to
    every token but the first at the input of ``layer`` and records only ``h``.
    """
    blocks = adapter.blocks(model)
    L = len(blocks)
    B, T = input_ids.shape
    N = B * (T - skip_first)
    store: dict[str, torch.Tensor] = {}
    cursor = {"row": 0}
    gen = torch.Generator().manual_seed(perturb[2]) if perturb else None

    def put(key: str, layers: int, idx: int, x: torch.Tensor) -> None:
        if key not in store:
            store[key] = torch.empty((layers, N, x.shape[-1]), dtype=torch.float32)
        rows = x[:, skip_first:].reshape(-1, x.shape[-1])
        store[key][idx, cursor["row"]:cursor["row"] + rows.shape[0]] = rows

    def pre_hook(idx: int):
        def hook(module, args, kwargs):
            x = args[0] if args else kwargs["hidden_states"]
            if perturb is not None and idx == perturb[0]:
                noise = torch.randn(x.shape, generator=gen, dtype=x.dtype)
                noise[:, 0] = 0.0
                x = x + perturb[1] * x.norm(dim=-1, keepdim=True) / np.sqrt(x.shape[-1]) * noise
                if args:
                    args = (x,) + tuple(args[1:])
                else:
                    kwargs["hidden_states"] = x
            put("h", L + 1, idx, x)
            return args, kwargs
        return hook

    handles = []
    for idx, block in enumerate(blocks):
        handles.append(block.register_forward_pre_hook(pre_hook(idx), with_kwargs=True))
        if perturb is None:
            handles.append(adapter.mlp_ln(block).register_forward_pre_hook(
                lambda m, args, idx=idx: put("mlp_in", L, idx, args[0])))
            handles.append(adapter.mlp(block).register_forward_hook(
                lambda m, args, out, idx=idx: put("mlp_out", L, idx, _first(out))))
    handles.append(blocks[-1].register_forward_hook(lambda m, args, out: put("h", L + 1, L, _first(out))))
    try:
        for start in range(0, B, batch_size):
            chunk = input_ids[start:start + batch_size]
            model(chunk, use_cache=False)
            cursor["row"] += chunk.shape[0] * (T - skip_first)
    finally:
        for h in handles:
            h.remove()
    return store


def massive_token_mask(traj: dict[str, torch.Tensor], factor: float = 4.0) -> tuple[torch.Tensor, np.ndarray]:
    """Tokens whose stream norm exceeds ``factor`` times the layer median at some layer,
    and the fraction of such tokens after each block."""
    norms = traj["h"].norm(dim=-1)
    ratio = norms / norms.median(dim=1, keepdim=True).values
    return (ratio > factor).any(dim=0), (ratio > factor).double().mean(dim=1)[1:].numpy()


# ---------------------------------------------------------------------------
# Affine fits with increment normalisation


def affine_fit_errors(X: torch.Tensor, Y: torch.Tensor, A: torch.Tensor, q: torch.Tensor) -> dict[str, float]:
    """Fit error relative to the spread of the output Y and of the increment Y - X.

    On a residual stream the output is mostly the input, so the output-normalised error is
    small for almost any map. The increment-normalised error is the share of the map's own
    work that no affine map explains (0: affine, 1: no better than a constant).
    """
    res = Y - (X @ A.T + q)
    sse = res.square().sum()
    sst_out = (Y - Y.mean(0, keepdim=True)).square().sum()
    D = Y - X
    sst_inc = (D - D.mean(0, keepdim=True)).square().sum()
    return {
        "rmse": float(torch.sqrt(sse / Y.shape[0])),
        "relative_rmse_output": float(torch.sqrt(sse) / (torch.sqrt(sst_out) + EPS)),
        "relative_rmse_increment": float(torch.sqrt(sse) / (torch.sqrt(sst_inc) + EPS)),
    }


def fit_section(X: torch.Tensor, Y: torch.Tensor, rel_ridge: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Least squares with a ridge proportional to the spread of X."""
    ridge = rel_ridge * float((X - X.mean(0)).square().sum(1).mean())
    return fit_affine(X, Y, ridge=ridge * X.shape[0])


# ---------------------------------------------------------------------------
# Per-block metrics


def weight_metrics(model: torch.nn.Module, adapter: Adapter) -> pd.DataFrame:
    rows = []
    blocks = adapter.blocks(model)
    for layer, block in enumerate(blocks):
        w = adapter.weights(block)
        u_norm = torch.sqrt(w.W_in.square().sum(1) + w.b_in.square())
        v_norm = w.W_out.norm(dim=0)
        cost = w.W_in.square().sum() + w.b_in.square().sum() + w.W_out.square().sum() + w.b_out.square().sum()
        alpha = torch.linalg.matrix_norm(w.W_out, ord=2) * torch.linalg.matrix_norm(w.W_in, ord=2)
        if w.W_gate is not None:
            # alpha_op is only a scale proxy for SwiGLU blocks, not a Lipschitz bound.
            cost = cost + w.W_gate.square().sum()
            alpha = alpha * torch.linalg.matrix_norm(w.W_gate, ord=2)
        rows.append({
            "layer": layer,
            "p": layer / len(blocks),
            "c_l": 0.5 * float(cost),  # folded MLP cost
            "alpha_op": float(alpha),
            "balance_log_ratio_median": float(torch.log(u_norm / (v_norm + EPS)).median()),
        })
    return pd.DataFrame(rows)


def activation_metrics(traj: dict[str, torch.Tensor], train_idx: torch.Tensor, val_idx: torch.Tensor,
                       rel_ridge: float, pairwise_points: int = 256) -> pd.DataFrame:
    H, M_in, M_out = traj["h"], traj["mlp_in"], traj["mlp_out"]
    rows = []
    pair_idx = val_idx[:pairwise_points]
    for layer in range(H.shape[0] - 1):
        X, Y = H[layer].double(), H[layer + 1].double()
        A, q = fit_section(X[train_idx], Y[train_idx], rel_ridge)
        val = affine_fit_errors(X[val_idx], Y[val_idx], A, q)
        train = affine_fit_errors(X[train_idx], Y[train_idx], A, q)
        # MLP branch alone, as a function of the residual stream entering it.
        m_in, m_out = M_in[layer].double(), M_out[layer].double()
        Am, qm = fit_section(m_in[train_idx], m_out[train_idx], rel_ridge)
        f = m_out[val_idx]
        nl_mlp = float((f - (m_in[val_idx] @ Am.T + qm)).norm() / ((f - f.mean(0)).norm() + EPS))
        # Pre-norm blocks only see the direction of the stream, so a common rescaling is a
        # gauge: divide out the growth of the median stream norm before splitting the
        # log singular values into contraction and expansion.
        gauge = float(X[train_idx].norm(dim=1).median() / Y[train_idx].norm(dim=1).median())
        logs = torch.log(torch.linalg.svdvals(A).clamp_min(EPS))
        logs_g = logs + np.log(gauge)
        x_pair = X[pair_idx] / X[pair_idx].norm(dim=1).mean()
        y_pair = Y[pair_idx] / Y[pair_idx].norm(dim=1).mean()
        step = (Y - X).norm(dim=1)
        rel_step = step / (X.norm(dim=1) + EPS)
        rows.append({
            "layer": layer,
            "abs_disp_median": float(step.median()),
            "rel_disp_mean": float(rel_step.mean()),
            "rel_disp_max": float(rel_step.max()),
            "stream_norm_median": float(X.norm(dim=1).median()),
            "one_step_nl_increment": val["relative_rmse_increment"],
            "one_step_nl_increment_train": train["relative_rmse_increment"],
            "one_step_rel_rmse_output": val["relative_rmse_output"],
            "nl_mlp": nl_mlp,
            "kappa_minus": float(torch.clamp(logs, max=0.0).sum()),
            "kappa_plus": float(torch.clamp(logs, min=0.0).sum()),
            "kappa_minus_gauge": float(torch.clamp(logs_g, max=0.0).sum()),
            "kappa_plus_gauge": float(torch.clamp(logs_g, min=0.0).sum()),
            "log_stream_growth": -float(np.log(gauge)),
            **{k.replace("pairwise", "pairwise_normed"): v for k, v in pairwise_log_distortion(x_pair, y_pair).items()},
        })
    return pd.DataFrame(rows)


def norm_escape_profile(traj: dict[str, torch.Tensor], keep: torch.Tensor | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Absolute stream norms and step sizes.

    A Neural-ODE limit needs bounded states and a bounded path length sum_l |h_{l+1} - h_l|.
    In norm escape the stream norm grows with depth, so that steps of fixed absolute size
    are small only relative to the stream. Rows are block inputs. The summary uses the
    median stream norm |h_1| after block 0 as unit.
    """
    H = traj["h"] if keep is None else traj["h"][:, keep]
    L = H.shape[0] - 1
    rows = []
    for layer in range(L + 1):
        X = H[layer].double()
        mu = X.mean(0)
        row = {
            "layer": layer,
            "stream_norm_median": float(X.norm(dim=1).median()),
            "stream_norm_q99": float(torch.quantile(X.norm(dim=1).float(), 0.99)),
            "spread_rms": float((X - mu).square().sum(1).mean().sqrt()),
        }
        if layer < L:
            D = H[layer + 1].double() - X
            step = D.norm(dim=1)
            row.update({
                "abs_disp_median": float(step.median()),
                # Share of the step along the current stream direction (pure growth when 1).
                "radial_share_mean": float(((D * X).sum(1) / (step * X.norm(dim=1) + EPS)).mean()),
            })
        rows.append(row)
    df = pd.DataFrame(rows)
    bulk = df.iloc[1:L]
    norms = df["stream_norm_median"]
    path = float(df["abs_disp_median"].iloc[1:L].sum())
    net = float((H[L].double() - H[1].double()).norm(dim=1).median())
    summary = {
        "norm_after_block0": float(norms.iloc[1]),
        "norm_ratio_bulk": float(norms.iloc[L - 1] / norms.iloc[1]),
        "norm_ratio_final": float(norms.iloc[L] / norms.iloc[1]),
        "spread_ratio_final": float(df["spread_rms"].iloc[L] / df["spread_rms"].iloc[1]),
        "norm_growth_exponent": float(np.polyfit(np.log(bulk["layer"].to_numpy(float)), np.log(bulk["stream_norm_median"]), 1)[0]) if len(bulk) > 2 else np.nan,
        "abs_disp_bulk_median": float(bulk["abs_disp_median"].median()),
        "path_length_bulk": path,
        "straightness": net / path if path > 0 else np.nan,
        "radial_share_bulk_median": float(bulk["radial_share_mean"].median()),
    }
    return df, summary


def jump_scores(layers: pd.DataFrame) -> np.ndarray:
    """Robust outlier score of each block, used to choose split points of the affine scan."""
    pieces = []
    for col, log in (("c_l", True), ("rel_disp_max", True), ("alpha_op", True), ("pairwise_normed_distortion_max", False)):
        v = layers[col].to_numpy(dtype=float)
        pieces.append(robust_z(np.log(np.clip(v, EPS, None)) if log else v))
    return np.sum(pieces, axis=0)


# ---------------------------------------------------------------------------
# Affine sections


def section_error_matrix(traj: dict[str, torch.Tensor], train_idx: torch.Tensor, val_idx: torch.Tensor,
                         rel_ridge: float) -> pd.DataFrame:
    """Train and validation affine-fit errors of every section [a, b)."""
    H = traj["h"]
    L = H.shape[0] - 1
    rows = []
    for a in range(L):
        X = H[a].double()
        for b in range(a + 1, L + 1):
            Y = H[b].double()
            A, q = fit_section(X[train_idx], Y[train_idx], rel_ridge)
            val = affine_fit_errors(X[val_idx], Y[val_idx], A, q)
            train = affine_fit_errors(X[train_idx], Y[train_idx], A, q)
            rows.append({"start": a, "end": b, "length": b - a, **val, **{f"train_{k}": v for k, v in train.items()}})
    return pd.DataFrame(rows)


def affine_section_scan(model, adapter, input_ids, traj, train_idx, val_idx, jump_score, cfg, keep) -> pd.DataFrame:
    """Greedy scan of diagnostics.greedy_affine_scan with increment-normalised errors.

    Probes perturb the stream at the section start (all tokens but the first) and run the
    real model forward, so attention sees a consistent perturbed context.
    """
    H = traj["h"]
    L = H.shape[0] - 1
    probe_ids = input_ids[: int(cfg["probe_sequences"])]
    probe_keep = keep[: probe_ids.shape[0] * (probe_ids.shape[1] - 1)]
    tau_rel, tau_probe = float(cfg["tau_rel"]), float(cfg["tau_probe_rel"])
    cache: dict[int, torch.Tensor] = {}

    def probe_states(start: int) -> torch.Tensor:
        if start not in cache:
            pt = residual_trajectory(model, adapter, probe_ids, perturb=(start, float(cfg["probe_radius"]), 1000 + start))
            cache[start] = pt["h"][:, probe_keep]
        return cache[start]

    def evaluate(start: int, end: int) -> dict[str, Any]:
        from resnet_theory.diagnostics import classify

        X, Y = H[start].double(), H[end].double()
        A, q = fit_section(X[train_idx], Y[train_idx], float(cfg["rel_ridge"]))
        train = affine_fit_errors(X[train_idx], Y[train_idx], A, q)["relative_rmse_increment"]
        val = affine_fit_errors(X[val_idx], Y[val_idx], A, q)["relative_rmse_increment"]
        probe = val
        if max(train, val) <= tau_rel:  # probes cost a forward pass, skip them when the fit already fails
            P = probe_states(start)
            probe = affine_fit_errors(P[start].double(), P[end].double(), A, q)["relative_rmse_increment"]
        return {"section_start": start, "section_end": end, "length": end - start,
                "max_jump_score": float(jump_score[start:end].max()), "train_relative_rmse": train,
                "val_relative_rmse": val, "probe_relative_rmse": probe,
                "classification": classify(train, val, probe, tau_rel, tau_probe)}

    scan_cfg = {"min_section_length": int(cfg["min_section_length"]), "max_evaluations": 4096,
                "jump_score_threshold": float(cfg["jump_threshold"])}
    return pd.DataFrame(greedy_affine_scan(L, jump_score, scan_cfg, evaluate))


# ---------------------------------------------------------------------------
# Driver


DEFAULT_CFG = {
    "val_fraction": 0.25,  # share of sequences held out
    "rel_ridge": 1e-6,
    "massive_factor": 4.0,
    "tau_rel": 0.3,
    "tau_probe_rel": 0.5,
    "probe_sequences": 8,
    "probe_radius": 0.05,
    "jump_threshold": 4.0,
}


def analyse_residual_stream(model, adapter, ids: torch.Tensor, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    """Per-block table, section error matrix and affine sections on the windows ``ids``."""
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    traj = residual_trajectory(model, adapter, ids)
    T = ids.shape[1] - 1
    n_train_seq = ids.shape[0] - max(1, int(round(cfg["val_fraction"] * ids.shape[0])))
    massive, massive_frac = massive_token_mask(traj, cfg["massive_factor"])
    keep = ~massive
    tokens = torch.arange(ids.shape[0] * T)
    # Split by sequence, so that held-out tokens do not share context with training tokens.
    train_idx = tokens[(tokens < n_train_seq * T) & keep]
    val_idx = tokens[(tokens >= n_train_seq * T) & keep]
    layers = weight_metrics(model, adapter).merge(activation_metrics(traj, train_idx, val_idx, cfg["rel_ridge"]), on="layer")
    layers["massive_token_frac"] = massive_frac
    layers["jump_score"] = jump_scores(layers)
    L = len(layers)
    cfg.setdefault("min_section_length", max(2, round(0.1 * L)))

    matrix = section_error_matrix(traj, train_idx, val_idx, cfg["rel_ridge"])
    # The scan accepts nothing unless some long enough section passes on training and
    # held-out tokens. The matrix already has these errors, so the scan is skipped otherwise.
    candidates = matrix[(matrix["length"] >= cfg["min_section_length"])
                        & (matrix["relative_rmse_increment"] <= cfg["tau_rel"])
                        & (matrix["train_relative_rmse_increment"] <= cfg["tau_rel"])]
    if candidates.empty:
        sections = pd.DataFrame(columns=["section_start", "section_end", "length", "classification"])
    else:
        sections = affine_section_scan(model, adapter, ids, traj, train_idx, val_idx,
                                       layers["jump_score"].to_numpy(), cfg, keep)
    return {"layers": layers, "matrix": matrix, "sections": sections,
            "massive_token_frac": float(massive.double().mean())}


def concentration(values) -> dict[str, float]:
    x = np.clip(np.asarray(values, dtype=float), 0.0, None)
    total = x.sum()
    if total <= 0:
        return {"top_share": np.nan, "n_eff_over_L": np.nan}
    return {"top_share": float(x.max() / total), "n_eff_over_L": float(total**2 / np.square(x).sum()) / len(x)}
