"""Train and diagnose a depth sweep.

    python scripts/run_sweep.py configs/magic_lam2e-3_small.yaml
    python scripts/run_sweep.py configs/magic_lam2e-3_small.yaml --depths 64 --seeds 2

Finished runs are skipped, so several processes with different --depths/--seeds can
share one sweep directory. The sweep tables are rebuilt from all finished runs at the end.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from resnet_theory.sweep import load_config, run_sweep  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config")
    parser.add_argument("--depths", type=int, nargs="+", help="Subset of the configured depths.")
    parser.add_argument("--seeds", type=int, nargs="+", help="Subset of the configured seeds.")
    parser.add_argument("--output-dir", help="Overrides output_dir of the config.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    root = run_sweep(cfg, depths=args.depths, seeds=args.seeds)
    print(f"Wrote {root}")


if __name__ == "__main__":
    main()
