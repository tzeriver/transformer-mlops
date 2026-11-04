"""Evaluate the saved ESOL GNN artifact; exits non-zero if the gate fails.

Recomputes scaffold-split test RMSE from the raw CSV so the reported number
is reproducible from the artifact alone, not just trusted from training logs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gnn.pipeline import evaluate_saved

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=REPO_ROOT / "artifacts" / "gnn")
    p.add_argument("--csv", type=Path, default=REPO_ROOT / "data" / "esol.csv")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-rmse", type=float, default=1.5)
    args = p.parse_args()

    rmse = evaluate_saved(args.artifacts / "model.pt", args.csv, seed=args.seed)
    print(f"scaffold test RMSE {rmse:.4f} (gate: <= {args.max_rmse})")
    if rmse > args.max_rmse:
        print("QUALITY GATE FAILED")
        raise SystemExit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
