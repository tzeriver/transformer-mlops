# Data

## esol.csv

MoleculeNet ESOL (Delaney, *J. Chem. Inf. Comput. Sci.* 2004), DeepChem
mirror. 1,128 compounds with measured log solubility (mol/L). Used for
validation and test in every run, and for training by default.

## aqsoldb.csv

AqSolDB curated solubility set: Sorkun, Khetan & Er, "AqSolDB, a curated
reference set of aqueous solubility and 2D descriptors for a diverse set of
compounds", *Scientific Data* 6, 143 (2019), doi:10.1038/s41597-019-0151-1.

- Source: Harvard Dataverse, doi:10.7910/DVN/OVHAW8, file
  `curated-solubility-dataset` (original CSV upload, file id 3407241),
  dataset version 1.0.
- License: CC0 1.0 (as stated on the Dataverse record).
- 9,982 compounds merged from 9 public sources; one of them is ESOL, so every
  ESOL compound is in here, 386 of them with a different selected value.
- Checksums: MD5 `02f732e52de921b8adaf4ef03cf747bf` (matches Dataverse),
  SHA-256 `7c47e021d03a5720f53cfddb7aef90c8911ea276a36306c839798733d8630332`.

Used only with `src/gnn/train.py --dataset aqsoldb-organic`, which keeps the
organic subset (one fragment, contains carbon, only C N O F P S Cl Br I) and
removes ESOL val/test molecules and scaffolds before training.
