"""Recompute the tables of a finished sweep from its model checkpoints, without training.

    python scripts/rediagnose.py configs/magic_lam2e-3_small.yaml

Useful after changing a diagnostic setting in the config (affine-scan thresholds, probe
radii, ...). Overwrites layers.csv, sections.csv and summary.json of every run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_theory.sweep import collect, load_config, rediagnose, sweep_dir  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--output-dir", help="Overrides output_dir of the config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    root = sweep_dir(cfg)
    for ckpt in sorted((root / "runs").glob("*/model.pt")):
        rediagnose(cfg, ckpt.parent)
        print(ckpt.parent.name, "done")
    collect(root)


if __name__ == "__main__":
    main()
