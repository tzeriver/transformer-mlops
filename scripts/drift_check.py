"""Weekly drift check: re-evaluate the model on freshly generated data.

Compares current test accuracy against the baseline recorded in
artifacts/metrics.json. Prints a JSON report; exits 0 when healthy,
2 when drift is detected (accuracy drop beyond tolerance).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import make_datasets
from src.model import TinyTransformerClassifier

TOLERANCE = 0.03  # allowed absolute accuracy drop before flagging drift


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    artifacts = root / "artifacts"
    metrics = json.loads((artifacts / "metrics.json").read_text())
    bundle = torch.load(artifacts / "model.pt", map_location="cpu", weights_only=False)

    # Fresh data, different seed: simulates newly arriving production data.
    loaders, _, _ = make_datasets(
        n_per_class=metrics["n_per_class"], seed=metrics["seed"] + 999, batch_size=256
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
    current = correct / n
    baseline = metrics["test_accuracy"]
    drift = (baseline - current) > TOLERANCE
    print(json.dumps({
        "baseline_test_accuracy": baseline,
        "current_test_accuracy": round(current, 4),
        "delta": round(current - baseline, 4),
        "drift_detected": drift,
    }, indent=2))
    raise SystemExit(2 if drift else 0)


if __name__ == "__main__":
    main()
