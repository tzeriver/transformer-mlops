"""Applicability domain: is a query molecule like the ones the model trained on?

Two kinds of signal, both computed against the training split and stored in
the model artifact (the container has no data/ directory):

- hard checks: elements never seen in training, several disconnected
  fragments (salts, mixtures), no carbon at all, or a heavy-atom count outside
  the training range. ESOL is single organic molecules, so these inputs are
  outside what the model can say anything about.
- nearest-neighbour similarity: max Tanimoto similarity (Morgan radius 2,
  2048 bits) to any training molecule. Across 5 scaffold splits, test
  molecules below LOW_SIMILARITY had clearly larger errors on average, but the
  signal is weak per molecule, so it is reported as a number plus a soft note,
  not as a confidence interval.
"""

from __future__ import annotations

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from src.gnn.featurize import parse_mol

LOW_SIMILARITY = 0.3

_FP_GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def fingerprint(mol):
    return _FP_GEN.GetFingerprint(mol)


def build_domain(train_smiles: list[str]) -> dict:
    """Summarize the training molecules into a picklable dict for the artifact."""
    mols = [parse_mol(s) for s in train_smiles]
    heavy = [m.GetNumHeavyAtoms() for m in mols]
    return {
        "elements": sorted({a.GetAtomicNum() for m in mols for a in m.GetAtoms()}),
        "max_fragments": max(len(Chem.GetMolFrags(m)) for m in mols),
        "heavy_atoms": [min(heavy), max(heavy)],
        "fingerprints": [fingerprint(m).ToBinary() for m in mols],
    }


class DomainChecker:
    """Loaded once per process; check() is cheap (one bulk Tanimoto pass)."""

    def __init__(self, domain: dict):
        self.elements = set(domain["elements"])
        self.max_fragments = domain["max_fragments"]
        self.min_heavy, self.max_heavy = domain["heavy_atoms"]
        self.fps = [DataStructs.ExplicitBitVect(b) for b in domain["fingerprints"]]

    def check(self, mol) -> tuple[list[str], float]:
        """Return (warnings, nearest training-set Tanimoto similarity)."""
        warnings = []
        unseen = sorted(
            {a.GetSymbol() for a in mol.GetAtoms() if a.GetAtomicNum() not in self.elements}
        )
        if unseen:
            warnings.append(f"contains elements not seen in training: {', '.join(unseen)}")
        if len(Chem.GetMolFrags(mol)) > self.max_fragments:
            warnings.append(
                "several disconnected fragments (salt or mixture); "
                "training data are single molecules"
            )
        if not any(a.GetAtomicNum() == 6 for a in mol.GetAtoms()):
            warnings.append("no carbon atoms; training data are organic molecules")
        heavy = mol.GetNumHeavyAtoms()
        if not self.min_heavy <= heavy <= self.max_heavy:
            warnings.append(
                f"{heavy} heavy atoms, outside the training range "
                f"{self.min_heavy}-{self.max_heavy}"
            )
        similarity = max(DataStructs.BulkTanimotoSimilarity(fingerprint(mol), self.fps))
        if similarity < LOW_SIMILARITY:
            warnings.append(
                f"low similarity to training molecules (nearest Tanimoto "
                f"{similarity:.2f} < {LOW_SIMILARITY}); errors are larger on "
                "average for such molecules"
            )
        return warnings, similarity
