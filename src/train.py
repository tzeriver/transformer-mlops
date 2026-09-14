"""Train the tiny transformer classifier.

Saves artifacts/model.pt (weights + vocab + classes + config) and
artifacts/metrics.json (validation / test accuracy, run config).
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
) -> dict:
    set_seed(seed)
    loaders, vocab, _ = make_datasets(
        n_per_class=n_per_class, seed=seed, batch_size=batch_size
    )
    model = TinyTransformerClassifier(vocab_size=len(vocab), num_classes=len(CLASSES))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

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
    if verbose:
        print(f"saved artifacts -> {out} | test accuracy {test_acc:.3f}")
    return metrics


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-per-class", type=int, default=1200)
    p.add_argument("--out", type=Path, default=Path("artifacts"))
    args = p.parse_args()
    train_model(
        epochs=args.epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        seed=args.seed,
        n_per_class=args.n_per_class,
        out=args.out,
    )


if __name__ == "__main__":
    main()
