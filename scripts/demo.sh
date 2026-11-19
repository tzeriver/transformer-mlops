#!/usr/bin/env bash
# One-command local demo: train both models if their artifacts are missing,
# build the container image, and serve the API on localhost:8000.
#   scripts/demo.sh            # uses ./.venv/bin/python, or python on PATH
#   PORT=8080 scripts/demo.sh
set -euo pipefail

cd "$(dirname "$0")/.."
PY=${PYTHON:-$( [ -x .venv/bin/python ] && echo .venv/bin/python || echo python )}
PORT=${PORT:-8000}

if [ ! -f artifacts/model.pt ]; then
  echo "== training the text classifier (~15 s)"
  "$PY" src/train.py --epochs 10
fi
if [ ! -f artifacts/gnn/model.pt ]; then
  echo "== training the solubility GNN on AqSolDB (~4 min)"
  "$PY" src/gnn/train.py --epochs 100 --pool mean_sum_max --descriptors \
    --dataset aqsoldb-organic --variant msm-desc-aqsoldb
fi

echo "== building the image"
docker build -t transformer-mlops .

echo "== serving on http://localhost:$PORT (Ctrl-C to stop)"
echo "   try: curl -X POST localhost:$PORT/predict-solubility -H 'Content-Type: application/json' -d '{\"smiles\":\"CCO\"}'"
docker run --rm -p "$PORT:8000" transformer-mlops
