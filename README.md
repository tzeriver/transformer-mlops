# tiny-transformer-mlops

A deployable MLOps portfolio project: a **small transformer classifier** trained on
synthetic data, served behind a FastAPI prediction endpoint, with tests, a quality-gated
CI pipeline, a container image, and a weekly drift check.

## What it does

- **Model** — 2-layer, 4-head transformer encoder (~65k params) classifying short
  research-note sentences into `methods` / `results` / `background`.
- **Data** — 100% synthetic, template-generated demo data (`src/data.py`). No real or
  proprietary data anywhere in this repo.
- **API** — FastAPI service (`src/serve.py`):
  - `GET /health` → `{"status": "ok"}`
  - `POST /predict` with `{"text": "..."}` → `{"label", "confidence", "probabilities"}`
  - `POST /predict-solubility` with `{"smiles": "CCO"}` → `{"smiles", "predicted_log_solubility",
    "nearest_train_similarity", "warnings"}` (ESOL GNN track; invalid SMILES → 422)

## ESOL GNN track (Gap 2 + Gap 4)

A second track in the same repo: a small GCN (plain PyTorch, no
torch-geometric) predicting water solubility (log mol/L) on MoleculeNet ESOL
(~1,128 compounds, `data/esol.csv`, DeepChem mirror). RDKit turns each SMILES
into a molecular graph (atoms = nodes, bonds = edges, atom + bond features);
2 message-passing layers with bond-gated messages, global mean pooling, MLP
head. The transformer track is untouched — separate module (`src/gnn/`),
separate artifacts (`artifacts/gnn/`).

Evaluation honesty: the headline number uses a **scaffold split** (Bemis-Murcko
scaffolds never shared across train/val/test), with a random-split model
trained for contrast — the gap between the two is the point.

```bash
pip install -r requirements.txt   # adds rdkit, mlflow, prefect (torch separate, as above)

python src/gnn/train.py --epochs 100
# trains scaffold model -> artifacts/gnn/model.pt + metrics.json,
# trains an identical random-split model for contrast, logs both runs to
# ./mlruns, and runs the champion/challenger registry gate (a new model only
# becomes champion when it beats the current champion's validation RMSE)

python src/gnn/evaluate.py --max-rmse 1.5   # recomputes scaffold test RMSE from the artifact

mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
# experiment "esol-gnn", model registry "esol-gnn" (champion / challenger aliases)

python flows/gnn_flow.py   # Prefect: data prep -> train -> evaluate -> register (retries=2 per task)
# trains the served msm-desc model by default; takes the same model options as
# src/gnn/train.py (e.g. --pool mean --no-descriptors --variant baseline)
# Sandbox note: this environment's proxy vars break Prefect's ephemeral
# server startup (httpx chokes parsing them); the flow only talks to
# localhost, so run it with proxies unset:
# env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY python flows/gnn_flow.py

uvicorn src.serve:app --port 8000
curl -X POST localhost:8000/predict-solubility \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO"}'
# {"smiles":"CCO","predicted_log_solubility":...,"nearest_train_similarity":1.0,"warnings":[]}
```

Model options (all off by default, so the default model is unchanged):

```bash
python src/gnn/train.py --pool mean_sum_max --descriptors --variant msm-desc
#   --pool mean_sum_max  concat mean + scaled sum + max pooling (molecule size, extreme atoms)
#   --descriptors        add 8 RDKit descriptors (logP, MW, TPSA, ...) to the head
#   --residual           skip connections after layer 1 (pair with --num-layers 4)
#   --variant NAME       MLflow run name prefix + "variant" param/tag
```

Every MLflow run also logs `git_commit`, `git_dirty`, the ESOL CSV's
`data_sha256`, the training `dataset`, and the training file's
`train_data_sha256`.

Larger training set (opt-in): `--dataset aqsoldb-organic` trains on AqSolDB
(`data/aqsoldb.csv`, 9,982 compounds, CC0; see `data/README.md`), keeping its
organic subset: one fragment, contains carbon, only ESOL's elements. AqSolDB
already contains every ESOL compound, so every ESOL val/test molecule and, for
the scaffold split, every val/test scaffold is removed from it first. Val and
test stay ESOL, so the numbers below compare like with like:

| ESOL scaffold split | ESOL only (902 train) | AqSolDB organic (~7,900 train) |
|---|---|---|
| seed 0 | 0.768 | 0.733 |
| seed 1 | 0.815 | 0.849 |
| seed 2 | 0.770 | 0.729 |
| seed 3 | 0.787 | 0.750 |
| seed 4 | 0.907 | 0.777 |
| mean | 0.809 | 0.768 |

On ESOL-type molecules: better on 4 of 5 splits, about 5% lower RMSE. Training
on all of AqSolDB (salts, metals, inorganics included) did not help there
(mean 0.828, measured before the descriptor clip below).

On broader chemistry the gain is larger. On scaffold-split test sets drawn
from AqSolDB itself (758 organic compounds not in ESOL, new scaffolds), RMSE
was 1.33 / 1.36 / 1.36 for the ESOL-only model vs 1.03 / 1.12 / 1.11 for the
AqSolDB model (split seeds 0-2), and about 1.45 vs 1.1 on the half of those
molecules that look least like ESOL (nearest Tanimoto < 0.3).

Retuning for the larger set did not help: hidden 64/128 x 2/3 layers, 100 vs
200 epochs and dropout 0.2 all landed within noise (validation RMSE
0.766-0.793), so the msm-desc defaults stay. Training takes ~10x longer (~5
min here vs ~30 s), so CI and the served model stay on ESOL for now.

Descriptor clipping: standardized descriptors are clipped to +-5 training SDs
(`DESC_CLIP` in `src/gnn/model.py`), and a hard domain warning names any
descriptor beyond that. RDKit counts 264 rings for gamma-cyclodextrin, and
before the clip that one value drove a prediction to log S +106. Clipping
also moved the served ESOL model's scaffold test RMSE from 0.939 to 0.929
without retraining.

AqSolDB also gives a measurement-noise reference: for the 2,236 compounds
reported by more than one source, the between-source SD has median 0.13 and
RMS 0.52 log units.

Ablation over 5 scaffold splits (split seeds 0-4, 100 epochs, best-val epoch):

| Variant | Scaffold test RMSE (mean ± sd) |
|---|---|
| default (2-layer GCN, mean pool) | 1.077 ± 0.086 |
| `--pool mean_sum_max --descriptors` | 0.819 ± 0.048 (better on 5/5 splits) |
| + `--residual --num-layers 4` | 0.814 ± 0.047 (depth adds ~nothing) |

The same 5 splits were used to choose the variants, so treat ~0.82 as
slightly optimistic. A single split's RMSE moves by ~±0.08, so compare
variants over several split seeds, not one run.

Status (GNN track, 100 epochs, seed 42, 2-layer GCN hidden 64,
`--pool mean_sum_max --descriptors`, variant `msm-desc`): scaffold-split test
RMSE **0.939** (val 0.745) vs random-split test RMSE **0.669** (val 0.559) —
the scaffold split stays the honest number; the random one flatters the model.
The seed-42 scaffold split is one of the harder ones, so this sits above the
5-split mean in the table above. The previous default model scored 1.174 on
the same split. Registry: this run was promoted to champion (val 0.745 beat the
previous champion's 1.043). See `artifacts/gnn/metrics.json`.

Applicability domain (`src/gnn/domain.py`): every solubility response says
how much the input resembles the training data. The artifact stores a summary
of the training split (elements, fragment count, heavy-atom range, Morgan
fingerprints, ~40 KB), and the endpoint returns:

- `warnings`: empty when the input looks like the training data. Hard
  warnings for elements never seen in training, salts/mixtures (several
  fragments), no carbon, or a heavy-atom count outside the training range —
  the prediction is still returned but is not meaningful (e.g. water, NaCl,
  siloxanes). A soft warning when the nearest training molecule has Tanimoto
  similarity < 0.3.
- `nearest_train_similarity`: that nearest-neighbour Tanimoto value.

The soft warning is a weak, average-level signal, not an error bar. Over 5
scaffold splits it flagged ~28% of test molecules, whose RMSE averaged 0.98 vs
0.74 for the rest (larger on every split), but per-molecule rank correlation
with error is only about −0.1 to −0.25. Hard warnings fire on ESOL itself only
for the largest glycosides (digoxin-size), which exceed the training size
range. Artifacts saved before this existed still serve, with
`nearest_train_similarity: null` and no warnings.

A 5-seed ensemble's spread was also tried as a per-prediction uncertainty and
dropped: it ranked errors poorly (Spearman +0.06 to +0.26) and only 28-37% of
errors fell within 2x the spread, for ~0.02 RMSE gain at 5x training cost.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python src/train.py --epochs 10          # trains, writes artifacts/model.pt + metrics.json
python src/evaluate.py --min-accuracy 0.85

uvicorn src.serve:app --port 8000
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"we trained the transformer for ten epochs with adam"}'
# {"label":"methods","confidence":0.98,"probabilities":{"methods":0.98,...}}
```

## CI/CD (`.github/workflows/ci.yml`)

Every push runs: **lint (ruff) → unit tests (pytest) → train → quality gate
(`evaluate.py`, fails below 85% test accuracy) → train GNN (the served
`msm-desc` variant) → GNN quality gate (`src/gnn/evaluate.py`, fails above 1.5
scaffold test RMSE) → build container image → smoke-test the running image**
(`/health` + a real `/predict` and `/predict-solubility` call). The last step,
which runs even when an earlier one fails, uploads `mlflow.db` + `mlruns/`
(both tracks' runs) as a downloadable run artifact.

## Container

```bash
docker build -t transformer-mlops .
docker run -p 8000:8000 transformer-mlops
```

`artifacts/` must exist at build time — CI trains the model before building.
No container daemon is available in this dev environment, so the image build is
verified in GitHub Actions (see CI).

## Monitoring

`scripts/drift_check.py` re-evaluates the model on freshly generated data and
compares against the baseline in `artifacts/metrics.json`, exiting 2 on drift
(accuracy drop > 3pp). `.github/workflows/drift.yml` runs it every Monday
06:00 UTC (or manually via Actions → drift → Run workflow); a drift result
fails the run, which notifies the repo owner. Nothing retrains automatically.

Because `artifacts/` is gitignored, the workflow trains a fresh baseline before
checking, and the data is synthetic, so this exercises the monitoring loop
rather than watching a deployed model.

## Layout

```
src/
  data.py      synthetic dataset generation + tokenization
  model.py     tiny transformer classifier
  train.py     training loop, saves artifacts/model.pt + metrics.json
  evaluate.py  test-set evaluation + CI quality gate
  serve.py     FastAPI inference service (/predict + /predict-solubility)
  gnn/
    featurize.py   RDKit SMILES -> molecular graph (atom/bond features)
    data.py        ESOL loading, scaffold + random splits
    model.py       plain-PyTorch GCN regressor (bond-gated messages)
    domain.py      applicability domain: out-of-domain warnings + similarity
    pipeline.py    train/evaluate/champion-challenger promotion (MLflow)
    train.py       CLI: scaffold model + random-split contrast + registry gate
    evaluate.py    artifact evaluation + RMSE quality gate
tests/         unit + API tests (incl. tests/test_gnn.py)
flows/
  gnn_flow.py  Prefect: data prep -> train -> evaluate -> register
data/
  esol.csv     MoleculeNet ESOL (DeepChem mirror, 1,128 compounds)
scripts/
  drift_check.py   weekly drift monitor
.github/workflows/ci.yml      lint, tests, train + gate both tracks, image smoke test
.github/workflows/drift.yml   weekly drift check
Dockerfile
requirements.txt   (pinned; install torch separately per instructions above)
```

## Status (verified 2026-09-13)

- [x] Model trains: **test accuracy 100%** on held-out synthetic data (10 epochs, best-val checkpoint)
- [x] Tests + lint pass: 7/7 pytest, ruff clean
- [x] Quality gate passes (`evaluate.py`, bar = 85%)
- [x] Drift check healthy (no drift on fresh data)
- [x] Live endpoint verified: `GET /health` → `{"status":"ok"}`; `POST /predict`
      correctly classified methods/results/background samples at ~1.0 confidence
- [x] Container image build verified in CI — green:
  https://github.com/tzeriver/transformer-mlops/actions/runs/34793760990
  (lint → tests → train → quality gate → docker build → image smoke test)
- [x] Public repo: https://github.com/tzeriver/transformer-mlops
- [ ] Live public endpoint via AWS App Runner — deferred until interview prep
      (workflow drafted at `.github/workflows/deploy.yml`, not yet added to repo;
      ~10 min to activate when needed)

## Deploy (AWS App Runner)

1. IAM user `mlops-deployer` with `AmazonEC2ContainerRegistryFullAccess`,
   `AWSAppRunnerFullAccess`, plus an inline policy allowing
   `iam:GetRole/CreateRole/AttachRolePolicy/PassRole` on
   `arn:aws:iam::<account-id>:role/AppRunnerECRAccessRole`. Create an access key.
2. Repo secrets: `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`
   (Settings → Secrets and variables → Actions).
3. Add `.github/workflows/deploy.yml`, then run it once via
   Actions → deploy → Run workflow.

Every green CI run on `main` then auto-deploys: retrain → rebuild → push to ECR →
create/update the App Runner service (health checks hit `/health`) → smoke-test
the live endpoint. ~$50/mo if left running 24/7 in us-east-1; delete the service
when not demoing.

**Pending access:** pushing to GitHub (`tzeriver`) and deploying to AWS both need
account access this environment doesn't have. The repo is ready to `git push`, and
the CI workflow deploys the smoke-tested image on every push once connected.
