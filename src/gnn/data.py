"""ESOL dataset loading and splitting.

ESOL (MoleculeNet / Delaney): ~1,128 compounds with measured log solubility
in mols per litre. Source CSV columns: 'smiles' and
'measured log solubility in mols per litre'.

Two splits, same fractions, different honesty:
- scaffold split: group by Bemis-Murcko scaffold so no scaffold appears in
  two splits. This is the honest number — it tests generalization to new
  chemistry, the way a real screen would.
- random split: same fractions, shuffled rows. Reported only for contrast;
  it flatters the model because near-duplicate scaffolds leak across splits.

Optional larger training set: AqSolDB (Sorkun et al., Sci. Data 2019; 9,982
compounds, CC0). It already contains every ESOL compound, so when it replaces
the training split, every val/test molecule (by InChIKey) and, for the
scaffold split, every val/test scaffold is removed from it first. Val and
test stay ESOL, so results remain comparable with the ESOL-only model.
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem
from rdkit.Chem.inchi import MolToInchiKey

from src.gnn.featurize import MolGraph, featurize, parse_mol

TARGET_COL = "measured log solubility in mols per litre"
SMILES_COL = "smiles"

AQSOLDB_SMILES_COL = "SMILES"
AQSOLDB_TARGET_COL = "Solubility"
AQSOLDB_KEY_COL = "InChIKey"
# ESOL's elements: C N O F P S Cl Br I. The organic subset keeps single,
# carbon-containing molecules made only of these.
ORGANIC_ELEMENTS = {6, 7, 8, 9, 15, 16, 17, 35, 53}


def load_esol_rows(csv_path: Path) -> list[tuple[str, float]]:
    """Read (smiles, log solubility) rows from the ESOL CSV."""
    rows = []
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            smiles = (row.get(SMILES_COL) or "").strip()
            raw = row.get(TARGET_COL)
            if not smiles or raw is None or raw == "":
                continue
            rows.append((smiles, float(raw)))
    return rows


def is_organic(mol) -> bool:
    """One fragment, at least one carbon, and only ESOL's elements."""
    atoms = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return (
        len(Chem.GetMolFrags(mol)) == 1
        and 6 in atoms
        and all(z in ORGANIC_ELEMENTS for z in atoms)
    )


def load_aqsoldb_rows(
    csv_path: Path, organic_only: bool = True
) -> tuple[list[tuple[str, float, str]], dict]:
    """Read (smiles, log S, InChIKey) rows from AqSolDB's curated CSV.

    Returns the rows and counts of what was dropped (unparseable SMILES, and
    non-organic compounds when organic_only), so nothing disappears silently.
    """
    rows, counts = [], {"rows": 0, "unparseable": 0, "non_organic": 0}
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            counts["rows"] += 1
            smiles = (row.get(AQSOLDB_SMILES_COL) or "").strip()
            try:
                mol = parse_mol(smiles)
            except ValueError:
                counts["unparseable"] += 1
                continue
            if organic_only and not is_organic(mol):
                counts["non_organic"] += 1
                continue
            rows.append((smiles, float(row[AQSOLDB_TARGET_COL]), row[AQSOLDB_KEY_COL]))
    return rows, counts


def replace_train(
    splits: dict[str, list[MolGraph]],
    rows: list[tuple[str, float, str]],
    exclude_scaffolds: bool,
) -> tuple[dict[str, list[MolGraph]], dict]:
    """Swap in an external training set, minus anything that would leak.

    Drops rows whose InChIKey matches a val/test molecule and, when
    exclude_scaffolds (scaffold split), rows sharing a val/test scaffold.
    """
    held = splits["val"] + splits["test"]
    held_keys = {MolToInchiKey(parse_mol(g.smiles)) for g in held}
    held_scaffolds = {g.scaffold for g in held}
    kept = [(s, y) for s, y, key in rows if key not in held_keys]
    graphs, skipped = featurize_rows(kept)
    train = [g for g in graphs if not (exclude_scaffolds and g.scaffold in held_scaffolds)]
    info = {
        "dropped_same_molecule": len(rows) - len(kept),
        "dropped_shared_scaffold": len(graphs) - len(train),
        "skipped": skipped,
        "n_train": len(train),
    }
    return {"train": train, "val": splits["val"], "test": splits["test"]}, info


def featurize_rows(rows: list[tuple[str, float]]) -> tuple[list[MolGraph], int]:
    """Featurize rows; invalid SMILES are skipped and counted, never hidden."""
    graphs, skipped = [], 0
    for smiles, y in rows:
        try:
            graphs.append(featurize(smiles, y=y))
        except ValueError:
            skipped += 1
    return graphs, skipped


def _assign_groups(
    groups: list[list[MolGraph]], fracs: tuple[float, float, float]
) -> dict[str, list[MolGraph]]:
    """Greedy group assignment: largest groups first, each to the split with
    the largest remaining absolute deficit. Keeps splits near target sizes
    even when one scaffold group is very large (e.g. acyclic molecules)."""
    names = ["train", "val", "test"]
    totals = sum(len(g) for g in groups)
    targets = {n: f * totals for n, f in zip(names, fracs)}
    splits: dict[str, list[MolGraph]] = {n: [] for n in names}
    for group in sorted(groups, key=len, reverse=True):
        # Absolute remaining deficit: a large group goes to the split that
        # still needs the most molecules, so small splits are filled by
        # small groups at the end instead of being swamped early.
        name = max(names, key=lambda n: targets[n] - len(splits[n]))
        splits[name].extend(group)
    return splits


def scaffold_split(
    graphs: list[MolGraph],
    seed: int = 42,
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[MolGraph]]:
    by_scaffold: dict[str, list[MolGraph]] = defaultdict(list)
    for g in graphs:
        by_scaffold[g.scaffold].append(g)
    rng = random.Random(seed)
    groups = list(by_scaffold.values())
    rng.shuffle(groups)  # break ties deterministically; sizes drive assignment
    return _assign_groups(groups, fracs)


def random_split(
    graphs: list[MolGraph],
    seed: int = 42,
    fracs: tuple[float, float, float] = (0.8, 0.1, 0.1),
) -> dict[str, list[MolGraph]]:
    rng = random.Random(seed)
    items = list(graphs)
    rng.shuffle(items)
    n = len(items)
    n_train, n_val = int(fracs[0] * n), int(fracs[1] * n)
    return {
        "train": items[:n_train],
        "val": items[n_train:n_train + n_val],
        "test": items[n_train + n_val:],
    }
