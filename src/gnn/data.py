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
"""

from __future__ import annotations

import csv
import random
from collections import defaultdict
from pathlib import Path

from src.gnn.featurize import MolGraph, featurize

TARGET_COL = "measured log solubility in mols per litre"
SMILES_COL = "smiles"


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
