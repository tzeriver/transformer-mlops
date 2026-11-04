"""SMILES -> molecular graph featurization (RDKit, plain PyTorch tensors).

Graph A: atoms are nodes, bonds are edges. Atom features describe the atom
itself; bond features describe the bond. Invalid SMILES fail loudly with a
ValueError so bad inputs never silently become empty graphs.
"""

from __future__ import annotations

import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

# RDKit prints a parse error to stderr for every bad SMILES; we raise our own
# ValueError instead, so silence the C++ logger for clean CLI/test output.
RDLogger.DisableLog("rdApp.*")

# Atom feature layout:
#   atomic number one-hot over 9 common elements + other bucket: 10
#   degree one-hot 0..5 + other bucket (degree is clamped, so it stays 0): 7
#   formal charge (scalar): 1
#   hybridization one-hot over 5 types + other bucket: 6
#   aromatic flag: 1
#   total num Hs (scalar): 1
#   in-ring flag: 1
_ATOM_NUMBERS = [6, 7, 8, 16, 9, 17, 35, 53, 15]  # C N O S F Cl Br I P
_HYBRIDIZATIONS = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2,
]
ATOM_DIM = (len(_ATOM_NUMBERS) + 1) + 7 + 1 + (len(_HYBRIDIZATIONS) + 1) + 1 + 1 + 1

# Bond feature layout:
#   bond type one-hot over 4 types + other bucket: 5
#   conjugated flag: 1
#   in-ring flag: 1
_BOND_TYPES = [
    Chem.rdchem.BondType.SINGLE,
    Chem.rdchem.BondType.DOUBLE,
    Chem.rdchem.BondType.TRIPLE,
    Chem.rdchem.BondType.AROMATIC,
]
BOND_DIM = (len(_BOND_TYPES) + 1) + 1 + 1


def _one_hot(value, choices) -> list[float]:
    vec = [0.0] * (len(choices) + 1)
    vec[choices.index(value) if value in choices else len(choices)] = 1.0
    return vec


def _atom_features(atom) -> list[float]:
    feats: list[float] = []
    feats += _one_hot(atom.GetAtomicNum(), _ATOM_NUMBERS)
    feats += _one_hot(min(atom.GetDegree(), 5), [0, 1, 2, 3, 4, 5])
    feats.append(float(atom.GetFormalCharge()))
    feats += _one_hot(atom.GetHybridization(), _HYBRIDIZATIONS)
    feats.append(1.0 if atom.GetIsAromatic() else 0.0)
    feats.append(float(atom.GetTotalNumHs()))
    feats.append(1.0 if atom.IsInRing() else 0.0)
    return feats


def _bond_features(bond) -> list[float]:
    return (
        _one_hot(bond.GetBondType(), _BOND_TYPES)
        + [1.0 if bond.GetIsConjugated() else 0.0]
        + [1.0 if bond.IsInRing() else 0.0]
    )


class MolGraph:
    """One molecule: node features, directed edge list, bond features, target."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        y: float | None = None,
        smiles: str = "",
        scaffold: str = "",
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.y = y
        self.smiles = smiles
        self.scaffold = scaffold

    @property
    def num_nodes(self) -> int:
        return self.x.size(0)


def parse_mol(smiles: str):
    """Parse a SMILES string; raise ValueError on anything RDKit rejects."""
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError(f"invalid SMILES: {smiles!r}")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError(f"invalid SMILES: {smiles!r}")
    return mol


def featurize(smiles: str, y: float | None = None) -> MolGraph:
    """SMILES -> MolGraph. Raises ValueError for invalid SMILES."""
    mol = parse_mol(smiles)
    x = torch.tensor([_atom_features(a) for a in mol.GetAtoms()], dtype=torch.float32)
    src, dst, attrs = [], [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = _bond_features(bond)
        src += [i, j]
        dst += [j, i]
        attrs += [feats, feats]
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    edge_attr = torch.tensor(attrs, dtype=torch.float32)
    return MolGraph(
        x, edge_index, edge_attr, y=y, smiles=smiles, scaffold=_scaffold_smiles(mol)
    )


def _scaffold_smiles(mol) -> str:
    """Bemis-Murcko scaffold SMILES ('' for acyclic molecules)."""
    scaffold = MurckoScaffold.GetScaffoldForMol(mol)
    if scaffold is None or scaffold.GetNumAtoms() == 0:
        return ""
    return Chem.MolToSmiles(scaffold)


def collate_graphs(graphs: list[MolGraph]):
    """Batch graphs by concatenation with node-index offsets (no torch-geometric).

    Returns (x, edge_index, edge_attr, batch, y) where batch[i] is the graph
    index each node belongs to, and y is None if no graph carries a target.
    """
    xs, srcs, dsts, attrs, batches, ys = [], [], [], [], [], []
    offset = 0
    for gi, g in enumerate(graphs):
        xs.append(g.x)
        srcs.append(g.edge_index[0] + offset)
        dsts.append(g.edge_index[1] + offset)
        attrs.append(g.edge_attr)
        batches.append(torch.full((g.num_nodes,), gi, dtype=torch.long))
        if g.y is not None:
            ys.append(g.y)
        offset += g.num_nodes
    y = torch.tensor(ys, dtype=torch.float32) if len(ys) == len(graphs) else None
    return (
        torch.cat(xs, dim=0),
        torch.stack([torch.cat(srcs), torch.cat(dsts)], dim=0),
        torch.cat(attrs, dim=0),
        torch.cat(batches, dim=0),
        y,
    )
