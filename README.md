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
  serve.py     FastAPI inference service
tests/         unit + API tests
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
- [ ] Container image build verified in CI (pending first push — no container daemon here)
- [ ] Public repo + live public endpoint (pending GitHub/AWS access — see below)

**Pending access:** pushing to GitHub (`tzeriver`) and deploying to AWS both need
account access this environment doesn't have. The repo is ready to `git push`, and
the CI workflow deploys the smoke-tested image on every push once connected.
