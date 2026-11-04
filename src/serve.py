"""FastAPI inference service for the tiny transformer classifier."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from fastapi import FastAPI, HTTPException
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


# --- ESOL GNN track (added alongside, transformer /predict untouched) ---

GNN_ARTIFACT = Path(__file__).resolve().parent.parent / "artifacts" / "gnn" / "model.pt"

_gnn_model = None
_gnn_y_mean = None
_gnn_y_std = None


def _ensure_gnn_loaded() -> None:
    global _gnn_model, _gnn_y_mean, _gnn_y_std
    if _gnn_model is not None:
        return
    if not GNN_ARTIFACT.exists():
        raise RuntimeError(
            f"GNN artifact not found at {GNN_ARTIFACT}; run src/gnn/train.py first"
        )
    from src.gnn.model import GCNRegressor

    bundle = torch.load(GNN_ARTIFACT, map_location="cpu", weights_only=False)
    _gnn_model = GCNRegressor(
        atom_dim=bundle["atom_dim"],
        bond_dim=bundle["bond_dim"],
        hidden_dim=bundle["config"]["hidden_dim"],
        num_layers=bundle["config"]["num_layers"],
        dropout=bundle["config"]["dropout"],
    )
    _gnn_model.load_state_dict(bundle["state_dict"])
    _gnn_model.eval()
    _gnn_y_mean, _gnn_y_std = bundle["y_mean"], bundle["y_std"]


class SolubilityRequest(BaseModel):
    smiles: str = Field(min_length=1, max_length=500)


class SolubilityResponse(BaseModel):
    smiles: str
    predicted_log_solubility: float


@app.post("/predict-solubility", response_model=SolubilityResponse)
def predict_solubility(req: SolubilityRequest) -> SolubilityResponse:
    from src.gnn.featurize import collate_graphs, featurize

    _ensure_gnn_loaded()
    try:
        graph = featurize(req.smiles)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    x, edge_index, edge_attr, batch, _ = collate_graphs([graph])
    with torch.no_grad():
        pred = _gnn_model(x, edge_index, edge_attr, batch)[0].item()
    return SolubilityResponse(
        smiles=req.smiles,
        predicted_log_solubility=round(pred * _gnn_y_std + _gnn_y_mean, 4),
    )
