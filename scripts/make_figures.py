"""Combine sweeps of the three initialisations into the figures and depth tables.

    python scripts/make_figures.py --out outputs/figures_lam2e-3 \
        small=outputs/magic_lam2e-3_small depth_scaled=outputs/magic_lam2e-3_depth_scaled \
        near_zero=outputs/magic_lam2e-3_near_zero

Writes layer_cost_profiles, depth_scaling and accuracy (PNG and PDF),
depth_summary.csv (mean and std over seeds per init and depth) and run_summary_all.csv.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_theory.figures import make_figures  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("runs", nargs="+", help="label=sweep_directory pairs")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = [(item.split("=", 1)[0], Path(item.split("=", 1)[1])) for item in args.runs]
    make_figures(runs, Path(args.out))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
