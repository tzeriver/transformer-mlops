"""Evaluate the trained artifact; exits non-zero if the quality gate fails.

Used by CI as a deployment gate: the container image is only built when the
model clears the minimum accuracy bar.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_datasets
from src.model import TinyTransformerClassifier


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    p.add_argument("--min-accuracy", type=float, default=0.85)
    args = p.parse_args()

    metrics = json.loads((args.artifacts / "metrics.json").read_text())
    bundle = torch.load(args.artifacts / "model.pt", map_location="cpu", weights_only=False)
    loaders, _, _ = make_datasets(
        n_per_class=metrics["n_per_class"], seed=metrics["seed"], batch_size=256
    )
    model = TinyTransformerClassifier(
        vocab_size=len(bundle["vocab"]), num_classes=len(bundle["classes"])
    )
    model.load_state_dict(bundle["state_dict"])
    model.eval()

    correct, n = 0, 0
    with torch.no_grad():
        for x, y in loaders["test"]:
            correct += (model(x).argmax(dim=1) == y).sum().item()
            n += len(x)
    acc = correct / n
    print(f"test accuracy {acc:.4f} (gate: >= {args.min_accuracy})")
    if acc < args.min_accuracy:
        print("QUALITY GATE FAILED")
        raise SystemExit(1)
    print("quality gate passed")


if __name__ == "__main__":
    main()
