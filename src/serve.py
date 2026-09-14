"""FastAPI inference service for the tiny transformer classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data import MAX_LEN, encode
from src.model import TinyTransformerClassifier

ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "model.pt"

_model = None
_vocab = None
_classes = None


def _ensure_loaded() -> None:
    global _model, _vocab, _classes
    if _model is not None:
        return
    if not ARTIFACT.exists():
        raise RuntimeError(f"model artifact not found at {ARTIFACT}; run src/train.py first")
    bundle = torch.load(ARTIFACT, map_location="cpu", weights_only=False)
    _vocab, _classes = bundle["vocab"], bundle["classes"]
    _model = TinyTransformerClassifier(vocab_size=len(_vocab), num_classes=len(_classes))
    _model.load_state_dict(bundle["state_dict"])
    _model.eval()


class PredictRequest(BaseModel):
    text: str = Field(min_length=1, max_length=500)


class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict[str, float]


app = FastAPI(title="tiny-transformer-classifier")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    _ensure_loaded()
    ids = torch.tensor([encode(req.text, _vocab, MAX_LEN)], dtype=torch.long)
    with torch.no_grad():
        probs = torch.softmax(_model(ids)[0], dim=0)
    top = int(probs.argmax())
    return PredictResponse(
        label=_classes[top],
        confidence=round(float(probs[top]), 4),
        probabilities={c: round(float(p), 4) for c, p in zip(_classes, probs)},
    )
