"""Train the tiny transformer classifier.

Saves artifacts/model.pt (weights + vocab + classes + config) and
artifacts/metrics.json (validation / test accuracy, run config).

MLflow tracking mirrors the GNN track (src/gnn/pipeline.py): same
sqlite:///mlflow.db store, so both tracks show up in one `mlflow ui`.
Per-epoch val_accuracy is logged with step=epoch; the final best value
goes under its own key (best_val_accuracy), because MLflow's "latest"
value for a metric is the highest-step one and would otherwise shadow it.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import CLASSES, make_datasets
from src.model import TinyTransformerClassifier

# Same store as the GNN track (src/gnn/pipeline.py TRACKING_URI):
# MLflow 3.x refuses the plain file store (./mlruns) unless explicitly
# opted in, so SQLite backend store, artifacts still in ./mlruns.
TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT = "transformer"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_model(
    epochs: int = 10,
    lr: float = 3e-3,
    batch_size: int = 64,
    seed: int = 42,
    n_per_class: int = 1200,
    out: Path = Path("artifacts"),
    verbose: bool = True,
    log_mlflow: bool = True,
) -> dict:
    set_seed(seed)
    loaders, vocab, _ = make_datasets(
        n_per_class=n_per_class, seed=seed, batch_size=batch_size
    )
    model = TinyTransformerClassifier(vocab_size=len(vocab), num_classes=len(CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    run_id = None
    if log_mlflow:
        import mlflow

        mlflow.set_tracking_uri(TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT)
        run = mlflow.start_run(run_name=f"seed{seed}")
        run_id = run.info.run_id
        mlflow.log_params(
            {
                "epochs": epochs,
                "lr": lr,
                "batch_size": batch_size,
                "seed": seed,
                "n_per_class": n_per_class,
                "d_model": 64,
                "nhead": 4,
                "num_layers": 2,
            }
        )

    try:
        def run_epoch(loader, training: bool):
            model.train(training)
            total_loss, correct, n = 0.0, 0, 0
            for x, y in loader:
                if training:
                    opt.zero_grad()
                logits = model(x)
                loss = loss_fn(logits, y)
                if training:
                    loss.backward()
                    opt.step()
                total_loss += loss.item() * len(x)
                correct += (logits.argmax(dim=1) == y).sum().item()
                n += len(x)
            return total_loss / n, correct / n

        best_acc, best_state = 0.0, None
        for epoch in range(1, epochs + 1):
            tr_loss, tr_acc = run_epoch(loaders["train"], True)
            va_loss, va_acc = run_epoch(loaders["val"], False)
            if log_mlflow:
                import mlflow

                mlflow.log_metric("train_loss", tr_loss, step=epoch)
                mlflow.log_metric("train_accuracy", tr_acc, step=epoch)
                mlflow.log_metric("val_loss", va_loss, step=epoch)
                mlflow.log_metric("val_accuracy", va_acc, step=epoch)
            if verbose:
                print(
                    f"epoch {epoch:02d}  train loss {tr_loss:.3f} acc {tr_acc:.3f} | "
                    f"val loss {va_loss:.3f} acc {va_acc:.3f}",
                    flush=True,
                )
            if va_acc > best_acc:
                best_acc = va_acc
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()
        correct, n = 0, 0
        with torch.no_grad():
            for x, y in loaders["test"]:
                correct += (model(x).argmax(dim=1) == y).sum().item()
                n += len(x)
        test_acc = correct / n

        out = Path(out)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": best_state,
                "vocab": vocab,
                "classes": CLASSES,
                "config": {"d_model": 64, "nhead": 4, "num_layers": 2},
            },
            out / "model.pt",
        )
        metrics = {
            "val_accuracy": round(best_acc, 4),
            "test_accuracy": round(test_acc, 4),
            "epochs": epochs,
            "lr": lr,
            "seed": seed,
            "n_per_class": n_per_class,
        }
        (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
        if log_mlflow:
            import mlflow

            mlflow.log_metrics(
                {"best_val_accuracy": best_acc, "test_accuracy": test_acc}
            )
            mlflow.log_artifact(str(out / "model.pt"))
            mlflow.log_artifact(str(out / "metrics.json"))
    finally:
        if log_mlflow:
            import mlflow

            mlflow.end_run()

    if verbose:
        print(f"saved artifacts -> {out} | test accuracy {test_acc:.3f}")
    metrics["mlflow_run_id"] = run_id
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-class", type=int, default=1200)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    p.add_argument("--no-mlflow", action="store_true", help="skip MLflow tracking")
    args = p.parse_args()
    train_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        n_per_class=args.n_per_class,
        out=args.out,
        log_mlflow=not args.no_mlflow,
    )


if __name__ == "__main__":
    main()
