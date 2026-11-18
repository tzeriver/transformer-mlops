"""Prefect flow for the ESOL GNN track: data prep -> train -> evaluate -> register.

Mirrors the Gap 2 plan's flow shape (task-level retries, observable runs),
applied to the GNN track instead of the drift check. Run locally with:

    python flows/gnn_flow.py

Defaults train the served model (the msm-desc variant: mean+sum+max pooling
plus RDKit descriptors, same as CI). Override with the same options as
src/gnn/train.py, e.g. ``--pool mean --no-descriptors --variant baseline``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from prefect import flow, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.model import POOLS
from src.gnn.pipeline import (
    DEFAULT_CONFIG,
    evaluate_saved,
    prepare,
    promote_champion,
    train_one,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "esol.csv"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "gnn"
# The served model (README "Status", CI's GNN training step).
SERVED_VARIANT = "msm-desc"
SERVED_MODEL = {"pool": "mean_sum_max", "descriptors": True, "residual": False}


@task(retries=2)
def prepare_data(csv_path: Path, seed: int) -> dict:
    data = prepare(csv_path, seed=seed)
    print(
        f"prepared {data['n_graphs']} graphs "
        f"({data['n_skipped']} invalid SMILES skipped)"
    )
    return data


@task(retries=2)
def train(data: dict, cfg: dict, out: Path, variant: str | None) -> dict:
    provenance = {"variant": variant, "data_sha256": data["data_sha256"]}
    result = train_one(data["scaffold"], cfg, out=out, log_mlflow=True, **provenance)
    random_res = train_one(
        data["random"], cfg, out=None, log_mlflow=True, split_name="random", **provenance
    )
    result["metrics"]["random_test_rmse"] = random_res["metrics"]["test_rmse"]
    result["metrics"]["random_val_rmse"] = random_res["metrics"]["val_rmse"]
    (Path(out) / "metrics.json").write_text(
        json.dumps(result["metrics"], indent=2)
    )
    return result


@task(retries=2)
def evaluate(result: dict, csv_path: Path, seed: int) -> float:
    rmse = evaluate_saved(Path(result["out"]) / "model.pt", csv_path, seed=seed)
    print(f"flow evaluation: scaffold test RMSE {rmse:.4f}")
    return rmse


@task(retries=2)
def register(result: dict) -> str:
    decision = promote_champion(result["run_id"], result["metrics"]["val_rmse"])
    print(f"registry: {decision}")
    return decision


@flow(name="esol-gnn-pipeline")
def gnn_flow(
    csv_path: Path = DEFAULT_CSV,
    out: Path = DEFAULT_OUT,
    seed: int = 42,
    epochs: int = 100,
    pool: str = SERVED_MODEL["pool"],
    descriptors: bool = SERVED_MODEL["descriptors"],
    residual: bool = SERVED_MODEL["residual"],
    num_layers: int = DEFAULT_CONFIG["num_layers"],
    variant: str | None = SERVED_VARIANT,
) -> dict:
    cfg = {
        **DEFAULT_CONFIG,
        "seed": seed,
        "epochs": epochs,
        "pool": pool,
        "descriptors": descriptors,
        "residual": residual,
        "num_layers": num_layers,
    }
    data = prepare_data(csv_path, seed)
    result = train(data, cfg, out, variant)
    test_rmse = evaluate(result, csv_path, seed)
    decision = register(result)
    return {"test_rmse": test_rmse, "registry": decision, "metrics": result["metrics"]}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--pool", choices=POOLS, default=SERVED_MODEL["pool"])
    p.add_argument(
        "--descriptors",
        action=argparse.BooleanOptionalAction,
        default=SERVED_MODEL["descriptors"],
    )
    p.add_argument(
        "--residual", action=argparse.BooleanOptionalAction, default=SERVED_MODEL["residual"]
    )
    p.add_argument("--num-layers", type=int, default=DEFAULT_CONFIG["num_layers"])
    p.add_argument("--variant", type=str, default=SERVED_VARIANT)
    args = p.parse_args()
    state = gnn_flow(
        csv_path=args.csv,
        out=args.out,
        seed=args.seed,
        epochs=args.epochs,
        pool=args.pool,
        descriptors=args.descriptors,
        residual=args.residual,
        num_layers=args.num_layers,
        variant=args.variant,
    )
    print(state)


if __name__ == "__main__":
    main()
