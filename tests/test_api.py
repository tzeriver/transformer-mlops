import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from src import serve
from src.data import CLASSES
from src.train import train_model


def _client_with_tiny_model(tmp_path):
    out = tmp_path / "artifacts"
    train_model(epochs=1, n_per_class=40, batch_size=16, seed=0, out=out, verbose=False)
    # reset the lazy singleton so it picks up the monkeypatched path
    serve._model = None
    return TestClient(serve.app), out


def test_health(tmp_path, monkeypatch):
    client, out = _client_with_tiny_model(tmp_path)
    monkeypatch.setattr(serve, "ARTIFACT", out / "model.pt")
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_predict(tmp_path, monkeypatch):
    client, out = _client_with_tiny_model(tmp_path)
    monkeypatch.setattr(serve, "ARTIFACT", out / "model.pt")
    r = client.post("/predict", json={"text": "we trained the transformer for ten epochs"})
    assert r.status_code == 200
    body = r.json()
    assert body["label"] in CLASSES
    assert 0.0 <= body["confidence"] <= 1.0
    assert abs(sum(body["probabilities"].values()) - 1.0) < 0.01


def test_predict_rejects_empty(tmp_path, monkeypatch):
    client, out = _client_with_tiny_model(tmp_path)
    monkeypatch.setattr(serve, "ARTIFACT", out / "model.pt")
    r = client.post("/predict", json={"text": ""})
    assert r.status_code == 422
