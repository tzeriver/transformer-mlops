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
  - `POST /predict-solubility` with `{"smiles": "CCO"}` → `{"smiles", "predicted_log_solubility"}`
    (ESOL GNN track; invalid SMILES → 422)

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
# Sandbox note: this environment's proxy vars break Prefect's ephemeral
# server startup (httpx chokes parsing them); the flow only talks to
# localhost, so run it with proxies unset:
# env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY -u no_proxy -u NO_PROXY python flows/gnn_flow.py

uvicorn src.serve:app --port 8000
curl -X POST localhost:8000/predict-solubility \
  -H 'Content-Type: application/json' \
  -d '{"smiles":"CCO"}'
# {"smiles":"CCO","predicted_log_solubility":...}
```

Status (GNN track, 100 epochs, seed 42, 2-layer GCN hidden 64): scaffold-split
test RMSE **1.161** (val 1.043) vs random-split test RMSE **0.956** (val 0.915)
— the ~0.20 gap is the scaffold split doing its job; the random number flatters
the model. Current artifact is from the Prefect flow run; the CLI run matched
within 0.001 (scaffold test 1.160). Registry: v1 = champion (val 1.043), v2 =
challenger (val 1.0431, missed promotion by 0.0002 — the gate working as
designed). See `artifacts/gnn/metrics.json`.

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
(`evaluate.py`, fails below 85% test accuracy) → build container image →
smoke-test the running image** (`/health` + a real `/predict` call).

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
compares against the baseline in `artifacts/metrics.json`. A scheduled weekly
run flags drift (accuracy drop > 3pp); retraining is triggered only on drift —
the loop runs quietly otherwise.

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
.github/workflows/ci.yml
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
