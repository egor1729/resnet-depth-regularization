"""Residual-stream and weight diagnostics of open-weight transformers (no training).

    python scripts/run_open_weights.py EleutherAI/pythia-160m EleutherAI/pythia-160m@step0
    python scripts/run_open_weights.py PleIAs/Baguettotron PleIAs/Baguettotron@init

``name@revision`` selects a checkpoint (a branch or tag of the Hugging Face repository),
``name@init`` a fresh initialisation of the same configuration. Per model it writes
``<tag>__layers.csv`` (affine-fit errors, contraction, folded weight costs per block),
``<tag>__matrix.csv`` (fit errors of every section), ``<tag>__sections.csv`` (accepted
affine sections), ``<tag>__norm_escape.csv``, ``<tag>__anatomy.csv`` (gains, Jacobians,
rebalanced costs) and ``<tag>__summary.csv``. ``scripts/open_weights_table.py`` collects
the summaries.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_theory.anatomy import layer_anatomy  # noqa: E402
from resnet_theory.open_weights import (  # noqa: E402
    analyse_residual_stream,
    concentration,
    load_model,
    load_token_batches,
    massive_token_mask,
    norm_escape_profile,
    residual_trajectory,
)


def summarise(spec: str, layers: pd.DataFrame, sections: pd.DataFrame, matrix: pd.DataFrame,
              escape: dict, anatomy: pd.DataFrame | None, massive_frac: float) -> dict:
    L = len(layers)
    bulk = layers.iloc[1:L - 1]
    last = layers.iloc[L - 1]
    unit = escape["norm_after_block0"]
    multi = matrix[matrix["length"] >= 2]
    out = {
        "model": spec, "L": L, "d": None,
        # Table of the thesis: bulk = all blocks but the first and the last.
        "affine_error_bulk": float(bulk["one_step_nl_increment"].median()),
        "norm_growth_bulk": escape["norm_ratio_bulk"],
        "step_bulk": escape["abs_disp_bulk_median"] / unit,
        "path_bulk": escape["path_length_bulk"] / unit,
        # Further quantities quoted in the text.
        "affine_error_bulk_train": float(bulk["one_step_nl_increment_train"].median()),
        "affine_error_mlp_bulk": float(bulk["nl_mlp"].median()),
        "best_multiblock_section_error": float(multi["relative_rmse_increment"].min()) if len(multi) else np.nan,
        "num_affine_sections": int(len(sections)),
        "norm_growth_total": escape["norm_ratio_final"],
        "norm_growth_exponent": escape["norm_growth_exponent"],
        "straightness": escape["straightness"],
        "radial_share_bulk": escape["radial_share_bulk_median"],
        "last_block_affine_error": float(last["one_step_nl_increment"]),
        "last_block_most_affine": bool(layers["one_step_nl_increment"].idxmin() == L - 1),
        "last_block_contraction_over_bulk": float(last["kappa_minus_gauge"] / bulk["kappa_minus_gauge"].median()),
        "last_block_expansion_over_bulk": float(last["kappa_plus_gauge"] / bulk["kappa_plus_gauge"].median()),
        "massive_token_frac": massive_frac,
        **{f"folded_mlp_cost_{k}": v for k, v in concentration(layers["c_l"]).items()},
        "folded_mlp_cost_bulk_median": float(bulk["c_l"].median()),
    }
    if anatomy is not None:
        ab = anatomy.iloc[1:L - 1]
        out.update({
            "raw_mlp_cost_bulk_median": float(ab["raw_mlp_cost"].median()),
            "mlp_gain_bulk_median": float(ab["mlp_gain_median"].median()),
            "attn_gain_bulk_median": float(ab["attn_gain_median"].median()),
            "mlp_jacobian_bulk_median": float(ab["mlp_jac_op_median"].median()),
            "mlp_jacobian_bulk_sum": float(ab["mlp_jac_op_median"].sum()),
            "mlp_write_rel_bulk_median": float(ab["mlp_write_rel"].median()),
        })
        if "removable_share" in ab:
            out.update({"removable_share_bulk_median": float(ab["removable_share"].median()),
                        "rebalanced_mlp_cost_bulk_median": float(ab["mlp_cost_rebalanced"].median())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("models", nargs="+", help="Hugging Face ids, optionally with @revision or @init.")
    parser.add_argument("--out", default="outputs/open_weights")
    parser.add_argument("--sequences", type=int, default=64, help="Windows for the affine fits (a quarter held out).")
    parser.add_argument("--norm-sequences", type=int, default=32, help="Windows for norm escape and anatomy.")
    parser.add_argument("--seq-len", type=int, default=256)
    parser.add_argument("--jacobian-tokens", type=int, default=32)
    parser.add_argument("--no-anatomy", action="store_true", help="Skip gains, Jacobians and rebalancing.")
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.set_grad_enabled(True)

    for spec in args.models:
        start = time.time()
        name, _, revision = spec.partition("@")
        tag = spec.replace("/", "__").replace("@", "__")
        model, tokenizer, adapter = load_model(name, revision or None)

        ids = load_token_batches(tokenizer, args.sequences, args.seq_len)
        res = analyse_residual_stream(model, adapter, ids)
        res["layers"].insert(0, "model", spec)
        res["layers"].to_csv(out / f"{tag}__layers.csv", index=False)
        res["matrix"].to_csv(out / f"{tag}__matrix.csv", index=False)
        res["sections"].to_csv(out / f"{tag}__sections.csv", index=False)

        ids = load_token_batches(tokenizer, args.norm_sequences, args.seq_len)
        traj = residual_trajectory(model, adapter, ids)
        keep = ~massive_token_mask(traj)[0]
        escape_layers, escape = norm_escape_profile(traj, keep)
        escape_layers.to_csv(out / f"{tag}__norm_escape.csv", index=False)
        anatomy = None
        if not args.no_anatomy:
            anatomy = layer_anatomy(model, adapter, ids, traj, keep, jacobian_tokens=args.jacobian_tokens)
            anatomy.to_csv(out / f"{tag}__anatomy.csv", index=False)

        summary = summarise(spec, res["layers"], res["sections"], res["matrix"], escape, anatomy, res["massive_token_frac"])
        summary["d"] = int(traj["h"].shape[-1])
        pd.DataFrame([summary]).to_csv(out / f"{tag}__summary.csv", index=False)
        print(f"{spec}: affine error {summary['affine_error_bulk']:.2f}, norm growth {summary['norm_growth_bulk']:.1f}, "
              f"step {summary['step_bulk']:.2f}, path {summary['path_bulk']:.0f} ({time.time() - start:.0f}s)")


if __name__ == "__main__":
    main()
