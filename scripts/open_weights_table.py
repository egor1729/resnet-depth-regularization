"""Collect the summaries written by run_open_weights.py into one table.

    python scripts/open_weights_table.py outputs/open_weights

Writes summary_all.csv and prints the columns of the thesis table (affine error, norm
growth, step and path, over the bulk of all blocks but the first and the last).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

TABLE = ["model", "L", "affine_error_bulk", "norm_growth_bulk", "step_bulk", "path_bulk"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", default="outputs/open_weights")
    args = parser.parse_args()
    root = Path(args.out_dir)
    df = pd.concat([pd.read_csv(p) for p in sorted(root.glob("*__summary.csv"))], ignore_index=True)
    df.to_csv(root / "summary_all.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        print(df[TABLE].round(2).to_string(index=False))
        print()
        print(df.set_index("model").drop(columns=TABLE[1:]).T.to_string(float_format=lambda v: f"{v:.3g}"))


if __name__ == "__main__":
    main()
