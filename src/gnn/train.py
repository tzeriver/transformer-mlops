"""Train the ESOL GNN track.

Trains on the scaffold split (the saved artifact) and, for contrast, an
identical model on a random split. Both runs are tracked in MLflow; the
scaffold model is then put through the champion/challenger registry gate.

Saves artifacts/gnn/model.pt + metrics.json. Never touches the transformer
track's artifacts/model.pt or metrics.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gnn.pipeline import DEFAULT_CONFIG, prepare, promote_champion, train_one

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "esol.csv"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "gnn"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    p.add_argument("--hidden-dim", type=int, default=DEFAULT_CONFIG["hidden_dim"])
    p.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    p.add_argument("--batch-size", type=int, default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--seed", type=int, default=DEFAULT_CONFIG["seed"])
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking + registry")
    args = p.parse_args()

    cfg = {
        **DEFAULT_CONFIG,
        "epochs": args.epochs,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "batch_size": args.batch_size,
        "seed": args.seed,
    }
    data = prepare(args.csv, seed=args.seed)
    print(
        f"ESOL: {data['n_rows']} rows -> {data['n_graphs']} graphs "
        f"({data['n_skipped']} invalid SMILES skipped)",
        flush=True,
    )
    scaffold = train_one(
        data["scaffold"], cfg, out=args.out, log_mlflow=not args.no_mlflow
    )
    random_res = train_one(
        data["random"], cfg, out=None, log_mlflow=not args.no_mlflow, split_name="random"
    )
    metrics = scaffold["metrics"]
    metrics["random_test_rmse"] = random_res["metrics"]["test_rmse"]
    metrics["random_val_rmse"] = random_res["metrics"]["val_rmse"]
    (Path(args.out) / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        f"scaffold test RMSE {metrics['test_rmse']:.4f} | "
        f"random test RMSE {metrics['random_test_rmse']:.4f} "
        f"(gap shows the scaffold split is the honest number)",
        flush=True,
    )
    if not args.no_mlflow and scaffold["run_id"]:
        decision = promote_champion(scaffold["run_id"], metrics["val_rmse"])
        print(f"registry: {decision}", flush=True)
    print(f"saved artifacts -> {args.out}")


if __name__ == "__main__":
    main()
