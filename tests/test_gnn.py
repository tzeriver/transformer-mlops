import math
import sys
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import serve
from src.gnn.data import is_organic, load_aqsoldb_rows, replace_train, scaffold_split
from src.gnn.domain import DomainChecker, build_domain
from src.gnn.featurize import (
    ATOM_DIM,
    BOND_DIM,
    DESC_DIM,
    collate_graphs,
    featurize,
    parse_mol,
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


def test_domain_flags_out_of_domain_inputs():
    checker = DomainChecker(build_domain([s for s, _ in _SMILES]))
    warnings, sim = checker.check(parse_mol("CCO"))  # a training molecule
    assert warnings == [] and sim == 1.0
    warnings, _ = checker.check(parse_mol("[Na+].[Cl-]"))
    text = " ".join(warnings)
    assert "Na" in text and "fragments" in text and "no carbon" in text
    warnings, _ = checker.check(parse_mol("CC[Si](C)(C)C"))
    assert any("Si" in w for w in warnings)


def test_endpoint_returns_domain_fields(tmp_path, monkeypatch):
    out = _tiny_gnn_artifact(tmp_path)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    client = TestClient(serve.app)
    body = client.post("/predict-solubility", json={"smiles": "CCO"}).json()
    assert body["nearest_train_similarity"] == 1.0 and body["warnings"] == []
    body = client.post("/predict-solubility", json={"smiles": "O"}).json()
    assert any("no carbon" in w for w in body["warnings"])


def test_artifact_without_domain_still_serves(tmp_path, monkeypatch):
    # Artifacts saved before the domain check existed load without it.
    out = _tiny_gnn_artifact(tmp_path)
    bundle = torch.load(out / "model.pt", weights_only=False)
    del bundle["domain"]
    torch.save(bundle, out / "model.pt")
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    r = TestClient(serve.app).post("/predict-solubility", json={"smiles": "O"})
    assert r.status_code == 200
    assert r.json()["nearest_train_similarity"] is None and r.json()["warnings"] == []


def _write_aqsoldb(path, smiles_y):
    from rdkit.Chem.inchi import MolToInchiKey

    lines = ["ID,SMILES,Solubility,InChIKey"]
    for i, (smi, y) in enumerate(smiles_y):
        key = MolToInchiKey(parse_mol(smi)) if smi != "not a smiles" else "BAD"
        lines.append(f'X-{i},"{smi}",{y},{key}')
    path.write_text("\n".join(lines) + "\n")


def test_aqsoldb_loader_keeps_organic_subset(tmp_path):
    csv_path = tmp_path / "aq.csv"
    _write_aqsoldb(csv_path, [
        ("CCO", 1.1), ("[Na+].[Cl-]", 0.8), ("O", 1.7), ("CC[Si](C)(C)C", -3.0),
        ("not a smiles", 0.0), ("c1ccccc1O", 0.0),
    ])
    rows, counts = load_aqsoldb_rows(csv_path)
    assert [r[0] for r in rows] == ["CCO", "c1ccccc1O"]
    assert counts == {"rows": 6, "unparseable": 1, "non_organic": 3}
    assert is_organic(parse_mol("CCO")) and not is_organic(parse_mol("O"))


def test_replace_train_removes_val_test_leaks(tmp_path):
    graphs = [featurize(s, y=y) for s, y in _SMILES]
    by = {g.smiles: g for g in graphs}
    splits = {
        "train": [by["CCO"]],
        "val": [by["c1ccccc1"]],
        "test": [by["C1CCCCC1"]],
    }
    rows = [
        ("C1CCCCC1", -3.0, "XDTMQSROBMDMFD-UHFFFAOYSA-N"),  # cyclohexane: a test molecule
        ("CC1CCCCC1", -3.8, "UAEPNZWRGJTJPN-UHFFFAOYSA-N"),  # shares its scaffold
        ("Cc1ccccc1", -2.2, "YXFVVABEGXRONW-UHFFFAOYSA-N"),  # shares the val scaffold
        ("CCCC", -2.6, "IJDNQMDRQITEOD-UHFFFAOYSA-N"),       # acyclic: kept
    ]
    new, info = replace_train(splits, rows, exclude_scaffolds=True)
    assert [g.smiles for g in new["train"]] == ["CCCC"]
    assert new["val"] == splits["val"] and new["test"] == splits["test"]
    assert info["dropped_same_molecule"] == 1 and info["dropped_shared_scaffold"] == 2
    new, info = replace_train(splits, rows, exclude_scaffolds=False)
    assert sorted(g.smiles for g in new["train"]) == ["CC1CCCCC1", "CCCC", "Cc1ccccc1"]


_GAMMA_CYCLODEXTRIN = (
    "OCC1OC2OC3C(O)C(O)C(OC3CO)OC4C(O)C(O)C(OC4CO)OC5C(O)C(O)C(OC5CO)OC6C(O)C(O)"
    "C(OC6CO)OC7C(O)C(O)C(OC7CO)OC8C(O)C(O)C(OC8CO)OC9C(O)C(O)C(OC9CO)OC1C(O)C2O"
)


def test_extreme_descriptors_are_clipped_and_flagged(tmp_path, monkeypatch):
    # RDKit counts hundreds of rings for cyclodextrins; unclipped, that one
    # descriptor once pushed a prediction to log S +106.
    graphs = [featurize(s, y=y) for s, y in _SMILES]
    splits = {"train": graphs[:8], "val": graphs[8:10], "test": graphs[10:]}
    out = tmp_path / "gnn"
    cfg = {
        "hidden_dim": 16, "num_layers": 2, "dropout": 0.0, "lr": 1e-2,
        "batch_size": 8, "epochs": 2, "seed": 0, "pool": "mean", "descriptors": True,
    }
    train_one(splits, cfg, out=out, log_mlflow=False, verbose=False)
    monkeypatch.setattr(serve, "GNN_ARTIFACT", out / "model.pt")
    monkeypatch.setattr(serve, "_gnn_model", None)
    client = TestClient(serve.app)
    body = client.post("/predict-solubility", json={"smiles": _GAMMA_CYCLODEXTRIN}).json()
    assert any("descriptors far outside" in w and "ring count" in w for w in body["warnings"])
    # Clipped inputs keep the output on the same scale as ordinary molecules.
    ordinary = client.post("/predict-solubility", json={"smiles": "CCO"}).json()
    assert abs(body["predicted_log_solubility"] - ordinary["predicted_log_solubility"]) < 20
