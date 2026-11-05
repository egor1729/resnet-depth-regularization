#!/usr/bin/env bash
# Open-weight models and checkpoints used in the thesis. Downloads about 3 GB of weights
# from the Hugging Face Hub. The anatomy step (Jacobians) dominates the run time of the
# deep models; add --no-anatomy for a quicker run.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python}
"$PY" scripts/run_open_weights.py \
  PleIAs/Baguettotron PleIAs/Baguettotron@init \
  PleIAs/Monad PleIAs/Monad@init \
  HuggingFaceTB/SmolLM2-135M HuggingFaceTB/SmolLM2-135M@init \
  HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints@step-240000 \
  HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints@step-960000 \
  HuggingFaceTB/SmolLM2-135M-intermediate-checkpoints@step-1680000 \
  EleutherAI/pythia-160m EleutherAI/pythia-160m@step0 EleutherAI/pythia-160m@step1000 \
  EleutherAI/pythia-160m@step16000 EleutherAI/pythia-160m@step64000 \
  --out outputs/open_weights "$@"
"$PY" scripts/open_weights_table.py outputs/open_weights
