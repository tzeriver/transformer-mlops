"""Prefect flow for the ESOL GNN track: data prep -> train -> evaluate -> register.

Mirrors the Gap 2 plan's flow shape (task-level retries, observable runs),
applied to the GNN track instead of the drift check. Run locally with:

    python flows/gnn_flow.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from prefect import flow, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.gnn.pipeline import evaluate_saved, prepare, promote_champion, train_one

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CSV = REPO_ROOT / "data" / "esol.csv"
DEFAULT_OUT = REPO_ROOT / "artifacts" / "gnn"


@task(retries=2)
def prepare_data(csv_path: Path, seed: int) -> dict:
    data = prepare(csv_path, seed=seed)
    print(
        f"prepared {data['n_graphs']} graphs "
        f"({data['n_skipped']} invalid SMILES skipped)"
    )
    return data


@task(retries=2)
def train(data: dict, cfg: dict, out: Path) -> dict:
    result = train_one(data["scaffold"], cfg, out=out, log_mlflow=True)
    random_res = train_one(
        data["random"], cfg, out=None, log_mlflow=True, split_name="random"
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
) -> dict:
    from src.gnn.pipeline import DEFAULT_CONFIG

    cfg = {**DEFAULT_CONFIG, "seed": seed, "epochs": epochs}
    data = prepare_data(csv_path, seed)
    result = train(data, cfg, out)
    test_rmse = evaluate(result, csv_path, seed)
    decision = register(result)
    return {"test_rmse": test_rmse, "registry": decision, "metrics": result["metrics"]}


if __name__ == "__main__":
    state = gnn_flow()
    print(state)
