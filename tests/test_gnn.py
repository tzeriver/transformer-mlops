import math
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import serve
from src.gnn.data import scaffold_split
from src.gnn.featurize import ATOM_DIM, BOND_DIM, collate_graphs, featurize
from src.gnn.model import GCNRegressor
from src.gnn.pipeline import train_one

_SMILES = [
    ("CCO", -0.77),
    ("CC", -1.5),
    ("CCC", -2.0),
    ("c1ccccc1", -2.1),
    ("Cc1ccccc1", -2.6),
    ("CC(=O)O", -0.2),
    ("CC(=O)Oc1ccccc1C(=O)O", -1.5),
    ("CN1C=NC2=C1C(=O)N(C)C(=O)N2C", -0.1),
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", -3.5),
    ("O=C(O)c1ccccc1", -1.8),
    ("Cc1ccc(cc1)O", -1.2),
    ("C1CCCCC1", -3.0),
]


def test_featurize_ethanol():
    g = featurize("CCO")
    assert g.num_nodes == 3  # C, C, O (implicit Hs are features, not nodes)
    assert g.x.shape == (3, ATOM_DIM)
    assert g.edge_index.shape == (2, 4)  # 2 bonds x 2 directions
    assert g.edge_attr.shape == (4, BOND_DIM)
    assert torch.isfinite(g.x).all()


def test_featurize_invalid_smiles_fail_cleanly():
    for bad in ["not a smiles!!!", "", "C1CC"]:
        with pytest.raises(ValueError, match="invalid SMILES"):
            featurize(bad)


def test_scaffold_split_has_no_scaffold_leak():
    graphs = [featurize(s, y=y) for s, y in _SMILES]
    splits = scaffold_split(graphs, seed=0)
    assert sum(len(v) for v in splits.values()) == len(graphs)
    train_scaffolds = {g.scaffold for g in splits["train"]}
    test_scaffolds = {g.scaffold for g in splits["test"]}
    assert not train_scaffolds & test_scaffolds


def test_model_forward_shape():
    graphs = [featurize(s) for s, _ in _SMILES[:4]]
    x, edge_index, edge_attr, batch, _ = collate_graphs(graphs)
    model = GCNRegressor(hidden_dim=16, num_layers=2)
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr, batch)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_collate_offsets_second_graph():
    g1, g2 = featurize("CCO"), featurize("CC")
    _, edge_index, _, batch, _ = collate_graphs([g1, g2])
    # g2's node indices start at 3 (g1's node count)
    assert batch.tolist() == [0, 0, 0, 1, 1]
    assert int(edge_index.max()) == 4


def _tiny_gnn_artifact(tmp_path):
    graphs = [featurize(s, y=y) for s, y in _SMILES]
    splits = {"train": graphs[:8], "val": graphs[8:10], "test": graphs[10:]}
    out = tmp_path / "gnn"
    train_one(
        splits,
        {
            "hidden_dim": 16,
            "num_layers": 2,
            "dropout": 0.0,
            "lr": 1e-2,
            "batch_size": 8,
            "epochs": 2,
            "seed": 0,
        },
        out=out,
        log_mlflow=False,
        verbose=False,
    )
    return out


def test_predict_solubility_endpoint(tmp_path, monkeypatch):
    out = _tiny_gnn_artifact(tmp_path)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    client = TestClient(serve.app)
    r = client.post("/predict-solubility", json={"smiles": "CCO"})
    assert r.status_code == 200
    body = r.json()
    assert body["smiles"] == "CCO"
    assert math.isfinite(body["predicted_log_solubility"])


def test_predict_solubility_rejects_invalid(tmp_path, monkeypatch):
    out = _tiny_gnn_artifact(tmp_path)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    client = TestClient(serve.app)
    r = client.post("/predict-solubility", json={"smiles": "not a smiles!!!"})
    assert r.status_code == 422
