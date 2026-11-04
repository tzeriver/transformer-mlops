"""GNN training pipeline: prepare -> train -> evaluate -> register.

Mirrors src/train.py's style (a train function + argparse main in train.py),
extended with MLflow tracking and a champion/challenger registry promotion
step per the Gap 2 plan: a new model only becomes champion when it beats the
current champion's validation RMSE.
"""

from __future__ import annotations

import json
import math
import random
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.gnn.data import featurize_rows, load_esol_rows, random_split, scaffold_split
from src.gnn.featurize import ATOM_DIM, BOND_DIM, collate_graphs
from src.gnn.model import GCNRegressor

MODEL_NAME = "esol-gnn"
# MLflow 3.x refuses the plain file store (./mlruns) unless explicitly opted
# in, so use the Gap 2 plan's own upgrade path: SQLite backend store, with
# artifacts still landing in ./mlruns. Zero extra services to run.
TRACKING_URI = "sqlite:///mlflow.db"
DEFAULT_CONFIG = {
    "hidden_dim": 64,
    "num_layers": 2,
    "dropout": 0.1,
    "lr": 1e-3,
    "batch_size": 32,
    "epochs": 100,
    "seed": 42,
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def prepare(csv_path: Path, seed: int = 42) -> dict:
    """Load + featurize ESOL; return graphs and both splits."""
    rows = load_esol_rows(Path(csv_path))
    graphs, skipped = featurize_rows(rows)
    return {
        "n_rows": len(rows),
        "n_graphs": len(graphs),
        "n_skipped": skipped,
        "scaffold": scaffold_split(graphs, seed=seed),
        "random": random_split(graphs, seed=seed),
    }


def _batches(graphs, batch_size: int, shuffle: bool, rng: random.Random):
    items = list(graphs)
    if shuffle:
        rng.shuffle(items)
    for i in range(0, len(items), batch_size):
        yield collate_graphs(items[i : i + batch_size])


@torch.no_grad()
def _predict_rmse(model, graphs, y_mean: float, y_std: float, batch_size: int = 256) -> float:
    model.eval()
    sq, n = 0.0, 0
    for x, edge_index, edge_attr, batch, y in _batches(
        graphs, batch_size, False, random.Random(0)
    ):
        pred = model(x, edge_index, edge_attr, batch) * y_std + y_mean
        sq += float(((pred - y) ** 2).sum())
        n += len(y)
    return math.sqrt(sq / n)


def train_one(
    splits: dict,
    cfg: dict,
    out: Path | None = None,
    log_mlflow: bool = True,
    split_name: str = "scaffold",
    verbose: bool = True,
) -> dict:
    """Train on one split; return metrics (+ artifact paths, mlflow run id)."""
    set_seed(cfg["seed"])
    train_graphs, val_graphs, test_graphs = (
        splits["train"],
        splits["val"],
        splits["test"],
    )
    ys = torch.tensor([g.y for g in train_graphs])
    y_mean, y_std = float(ys.mean()), float(ys.std())
    model = GCNRegressor(
        atom_dim=ATOM_DIM,
        bond_dim=BOND_DIM,
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
    )
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    loss_fn = nn.MSELoss()
    rng = random.Random(cfg["seed"])

    run_id = None
    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment("esol-gnn")
        run = mlflow.start_run(run_name=f"{split_name}-seed{cfg['seed']}")
        run_id = run.info.run_id
        mlflow.log_params({**cfg, "split": split_name})
        mlflow.set_tag("split", split_name)

    try:
        best_rmse, best_state = float("inf"), None
        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            total_loss, n = 0.0, 0
            for x, edge_index, edge_attr, batch, y in _batches(
                train_graphs, cfg["batch_size"], True, rng
            ):
                opt.zero_grad()
                pred = model(x, edge_index, edge_attr, batch)
                loss = loss_fn(pred, (y - y_mean) / y_std)
                loss.backward()
                opt.step()
                total_loss += loss.item() * len(y)
                n += len(y)
            val_rmse = _predict_rmse(model, val_graphs, y_mean, y_std)
            if log_mlflow:
                mlflow.log_metric("train_loss", total_loss / n, step=epoch)
                mlflow.log_metric("val_rmse", val_rmse, step=epoch)
            if verbose and (epoch == 1 or epoch % 10 == 0 or epoch == cfg["epochs"]):
                print(f"epoch {epoch:03d}  train loss {total_loss / n:.4f} | val RMSE {val_rmse:.4f}", flush=True)
            if val_rmse < best_rmse:
                best_rmse = val_rmse
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        test_rmse = _predict_rmse(model, test_graphs, y_mean, y_std)
        metrics = {
            "split": split_name,
            "val_rmse": round(best_rmse, 4),
            "test_rmse": round(test_rmse, 4),
            "n_train": len(train_graphs),
            "n_val": len(val_graphs),
            "n_test": len(test_graphs),
            **{k: cfg[k] for k in ("hidden_dim", "num_layers", "dropout", "lr", "batch_size", "epochs", "seed")},
        }

        if out is not None:
            out = Path(out)
            out.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "state_dict": best_state,
                    "config": {
                        "hidden_dim": cfg["hidden_dim"],
                        "num_layers": cfg["num_layers"],
                        "dropout": cfg["dropout"],
                    },
                    "y_mean": y_mean,
                    "y_std": y_std,
                    "atom_dim": ATOM_DIM,
                    "bond_dim": BOND_DIM,
                },
                out / "model.pt",
            )
            (out / "metrics.json").write_text(json.dumps(metrics, indent=2))

        if log_mlflow:
            # NB: per-epoch "val_rmse" is logged with step=epoch, and MLflow's
            # "latest" value is the highest-step one — so the final best value
            # goes under its own key, otherwise the epoch-100 value shadows it.
            mlflow.log_metrics({"best_val_rmse": best_rmse, "test_rmse": test_rmse})
            if out is not None:
                mlflow.log_artifact(str(out / "model.pt"))
                mlflow.log_artifact(str(out / "metrics.json"))
            # MLflow 3 defaults pytorch serialization to pt2 (traced graph),
            # which needs an input example; pickle keeps the classic
            # weights + class-by-reference format, loadable in this repo.
            mlflow.pytorch.log_model(
                model, name="model", serialization_format="pickle"
            )
    finally:
        if log_mlflow:
            mlflow.end_run()

    if verbose:
        print(f"[{split_name}] val RMSE {best_rmse:.4f} | test RMSE {test_rmse:.4f}")
    return {"metrics": metrics, "run_id": run_id, "out": str(out) if out else None}


def promote_champion(run_id: str, val_rmse: float) -> str:
    """Register the run's model; champion only if it beats the champion.

    Returns a human-readable promotion decision. First-ever model becomes
    champion; a beaten champion is demoted to challenger; a losing model is
    registered as challenger.
    """
    import mlflow
    from mlflow.exceptions import MlflowException
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient()
    version = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    new_version = str(version.version)  # MLflow 3 versions are ints; aliases want str
    try:
        champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
    except MlflowException:
        # No champion alias yet (first model ever registered).
        champion = None
    if champion is None:
        client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
        return f"promoted to champion (first model, v{new_version})"
    champ_run = client.get_run(champion.run_id)
    # Prefer best_val_rmse (final best); fall back to val_rmse for older runs.
    champ_rmse = champ_run.data.metrics.get(
        "best_val_rmse", champ_run.data.metrics.get("val_rmse", float("inf"))
    )
    if val_rmse < champ_rmse:
        client.set_registered_model_alias(
            MODEL_NAME, "challenger", str(champion.version)
        )
        client.set_registered_model_alias(MODEL_NAME, "champion", new_version)
        return (
            f"promoted to champion (v{new_version}, val RMSE {val_rmse:.4f} < "
            f"champion v{champion.version} {champ_rmse:.4f})"
        )
    client.set_registered_model_alias(MODEL_NAME, "challenger", new_version)
    return (
        f"registered as challenger (v{new_version}, val RMSE {val_rmse:.4f} did "
        f"not beat champion v{champion.version} {champ_rmse:.4f})"
    )


@torch.no_grad()
def evaluate_saved(
    artifact_path: Path, csv_path: Path, seed: int = 42, split: str = "scaffold"
) -> float:
    """Recompute test RMSE of a saved artifact from the raw CSV."""
    bundle = torch.load(artifact_path, map_location="cpu", weights_only=False)
    data = prepare(Path(csv_path), seed=seed)
    graphs = data[split]["test"]
    model = GCNRegressor(
        atom_dim=bundle["atom_dim"],
        bond_dim=bundle["bond_dim"],
        hidden_dim=bundle["config"]["hidden_dim"],
        num_layers=bundle["config"]["num_layers"],
        dropout=bundle["config"]["dropout"],
    )
    model.load_state_dict(bundle["state_dict"])
    return _predict_rmse(model, graphs, bundle["y_mean"], bundle["y_std"])
