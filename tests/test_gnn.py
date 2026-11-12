import math
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import serve
from src.gnn.data import scaffold_split
from src.gnn.featurize import (
    ATOM_DIM,
    BOND_DIM,
    DESC_DIM,
    collate_graphs,
    featurize,
    stack_descriptors,
)
from src.gnn.model import GCNRegressor
from src.gnn.pipeline import build_model, train_one

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


def test_descriptors_shape_and_finite():
    graphs = [featurize(s) for s, _ in _SMILES[:4]]
    d = stack_descriptors(graphs)
    assert d.shape == (4, DESC_DIM)
    assert torch.isfinite(d).all()


def test_default_model_keeps_original_state_dict_keys():
    # Artifacts saved before pool/residual/descriptors existed must still load strictly.
    keys = set(GCNRegressor(hidden_dim=16).state_dict())
    assert not any(k.startswith("desc_") for k in keys)
    assert build_model({"hidden_dim": 16, "num_layers": 2, "dropout": 0.0}).state_dict().keys() == keys


def test_improved_model_forward_shape():
    graphs = [featurize(s) for s, _ in _SMILES[:4]]
    x, edge_index, edge_attr, batch, _ = collate_graphs(graphs)
    model = GCNRegressor(
        hidden_dim=16, num_layers=3, pool="mean_sum_max", residual=True, desc_dim=DESC_DIM
    )
    model.eval()
    with torch.no_grad():
        out = model(x, edge_index, edge_attr, batch, stack_descriptors(graphs))
    assert out.shape == (4,)
    assert torch.isfinite(out).all()
    with pytest.raises(ValueError, match="descriptors"):
        model(x, edge_index, edge_attr, batch)


def test_max_pooling_is_per_molecule():
    # A large molecule must not leak its max activations into a small one.
    small, big = featurize("C"), featurize("CCCCCCCCCCCCCCCCCCCC")
    x, edge_index, edge_attr, batch, _ = collate_graphs([small, big])
    model = GCNRegressor(hidden_dim=8, num_layers=1, pool="mean_sum_max")
    model.eval()
    with torch.no_grad():
        alone = model(*collate_graphs([small])[:4])
        together = model(x, edge_index, edge_attr, batch)
    assert torch.allclose(alone[0], together[0], atol=1e-6)


def test_improved_artifact_serves(tmp_path, monkeypatch):
    graphs = [featurize(s, y=y) for s, y in _SMILES]
    splits = {"train": graphs[:8], "val": graphs[8:10], "test": graphs[10:]}
    out = tmp_path / "gnn"
    cfg = {
        "hidden_dim": 16, "num_layers": 3, "dropout": 0.0, "lr": 1e-2,
        "batch_size": 8, "epochs": 2, "seed": 0,
        "pool": "mean_sum_max", "residual": True, "descriptors": True,
    }
    train_one(splits, cfg, out=out, log_mlflow=False, verbose=False)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    r = TestClient(serve.app).post("/predict-solubility", json={"smiles": "CCO"})
    assert r.status_code == 200
    assert math.isfinite(r.json()["predicted_log_solubility"])


def test_bond_free_molecules_serve(tmp_path, monkeypatch):
    # Single atoms and salts have no bonds; they used to crash with a 500.
    g = featurize("C")
    assert g.edge_index.shape == (2, 0)
    assert g.edge_attr.shape == (0, BOND_DIM)
    out = _tiny_gnn_artifact(tmp_path)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    client = TestClient(serve.app)
    for smiles in ["C", "O", "[Na+].[Cl-]"]:
        r = client.post("/predict-solubility", json={"smiles": smiles})
        assert r.status_code == 200, smiles
