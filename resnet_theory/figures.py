"""Figures and depth tables that combine sweeps from different initialisations."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

DEPTH_COLUMNS = [
    "train_acc", "val_acc", "train_loss", "val_loss", "total_residual_cost", "max_layer_cost", "top1_cost_share",
    "top3_cost_share", "effective_num_layers", "argmax_layer_p", "num_dead_blocks", "best_affine_section_length",
    "affine_section_coverage",
]


def load_sweeps(runs: list[tuple[str, Path]]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``runs``: (label, sweep directory) pairs. A label may repeat (e.g. extra seeds)."""
    summaries, layers = [], []
    for label, root in runs:
        summaries.append(pd.read_csv(Path(root) / "run_summary.csv").assign(init=label))
        layers.append(pd.read_csv(Path(root) / "layer_metrics.csv").assign(init=label))
    return pd.concat(summaries, ignore_index=True), pd.concat(layers, ignore_index=True)


def depth_table(summary: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in DEPTH_COLUMNS if c in summary]
    grouped = summary.groupby(["init", "L"])[cols]
    table = grouped.mean().add_suffix("_mean").join(grouped.std().add_suffix("_std"))
    table["n_seeds"] = summary.groupby(["init", "L"]).size()
    return table.reset_index()


def _save(fig, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=160)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)


def _depth_axis(ax, depths) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(depths)
    ax.set_xticklabels([str(L) for L in depths])
    ax.set_xlabel("depth L")


def plot_profiles(layers: pd.DataFrame, inits: list[str], out_dir: Path) -> None:
    """Depth-scaled layer cost L c_l against l/L, median over seeds, one panel per init."""
    depths = sorted(layers["L"].unique())
    colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(depths)))
    fig, axes = plt.subplots(1, len(inits), figsize=(4.2 * len(inits), 3.6), sharey=True, squeeze=False)
    for ax, init in zip(axes[0], inits):
        sub = layers[layers["init"] == init]
        med = sub.groupby(["L", "layer"], as_index=False).agg(p=("p", "median"), L_c_l=("L_c_l", "median"))
        for color, L in zip(colors, depths):
            line = med[med["L"] == L]
            ax.plot(line["p"], line["L_c_l"], marker="o", markersize=2.5, linewidth=1.2, color=color, label=f"L={L}")
        ax.set_title(init)
        ax.set_xlabel("layer position l/L")
        ax.set_yscale("log")
        ax.grid(alpha=0.25, which="both")
    axes[0][0].set_ylabel(r"$L\cdot c_\ell$ (median over seeds)")
    axes[0][-1].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "layer_cost_profiles")


def plot_depth_scaling(summary: pd.DataFrame, inits: list[str], out_dir: Path) -> None:
    """Total cost, largest layer cost, top share and N_eff against L (median and range over seeds)."""
    panels = [
        ("total_residual_cost", r"total cost $\sum_\ell c_\ell$", "linear"),
        ("max_layer_cost", r"largest layer cost $\max_\ell c_\ell$", "log"),
        ("top1_cost_share", "cost share of the largest layer", "linear"),
        ("effective_num_layers", r"effective no. of layers $(\sum c)^2/\sum c^2$", "log"),
    ]
    depths = sorted(summary["L"].unique())
    fig, axes = plt.subplots(1, len(panels), figsize=(16, 3.6))
    for ax, (col, title, yscale) in zip(axes, panels):
        for k, init in enumerate(inits):
            stats = summary[summary["init"] == init].groupby("L")[col].agg(["median", "min", "max"]).reindex(depths)
            ax.plot(depths, stats["median"], marker="o", color=f"C{k}", label=init)
            ax.fill_between(depths, stats["min"], stats["max"], color=f"C{k}", alpha=0.15, linewidth=0)
        if col == "max_layer_cost":
            ref = summary[summary["L"] == depths[0]][col].median()
            ax.plot(depths, [ref * depths[0] / L for L in depths], "k--", linewidth=1, label=r"$\propto 1/L$")
        if col == "effective_num_layers":
            ax.plot(depths, depths, "k--", linewidth=1, label="= L (flat)")
        _depth_axis(ax, depths)
        ax.set_yscale(yscale)
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25, which="both")
    axes[1].legend(fontsize=8)
    axes[3].legend(fontsize=8)
    fig.tight_layout()
    _save(fig, out_dir, "depth_scaling")


def plot_accuracy(summary: pd.DataFrame, inits: list[str], out_dir: Path) -> None:
    depths = sorted(summary["L"].unique())
    fig, ax = plt.subplots(figsize=(5, 3.6))
    for k, init in enumerate(inits):
        sub = summary[summary["init"] == init].groupby("L")[["train_acc", "val_acc"]].mean().reindex(depths)
        ax.plot(depths, sub["train_acc"], marker="o", color=f"C{k}", label=f"{init} train")
        ax.plot(depths, sub["val_acc"], marker="s", linestyle="--", color=f"C{k}", label=f"{init} val")
    _depth_axis(ax, depths)
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, out_dir, "accuracy")


def make_figures(runs: list[tuple[str, Path]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    inits = list(dict.fromkeys(label for label, _ in runs))
    summary, layers = load_sweeps(runs)
    summary.to_csv(out_dir / "run_summary_all.csv", index=False)
    depth_table(summary).to_csv(out_dir / "depth_summary.csv", index=False)
    plot_profiles(layers, inits, out_dir)
    plot_depth_scaling(summary, inits, out_dir)
    plot_accuracy(summary, inits, out_dir)
