#!/bin/bash
# Step 0/1 chain for the validation campaign: manifests -> cost measurement -> X3 census.
# Safe to re-run; every step is idempotent (manifests are rewritten, JSONs overwritten).
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
IMAGES=$HOME/ai4life/phuongnh/vlm-truth/data/coco2014/val2014/val2014
CODE=$REPO/results/validation_2026-10-01/code
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
mkdir -p "$RUN/step0" "$RUN/step1"
cd "$REPO" || exit 1

echo "=== s0_corpus $(date -Is) ==="
"$P" "$CODE/s0_corpus.py" --images-dir "$IMAGES" --out "$RUN/step0" || { echo S0_FAILED; exit 1; }

echo "=== s1_corpus $(date -Is) ==="
"$P" "$CODE/s1_corpus.py" --out "$RUN/step0" || { echo S1_FAILED; exit 1; }

echo "=== measure_cost $(date -Is) ==="
"$P" "$CODE/measure_cost.py" \
    --image-manifest "$RUN/step0/manifest-fit.jsonl" \
    --text-manifest "$RUN/step0/manifest-text-fit.jsonl" \
    --n 1 --json "$RUN/step0/cost.json" || { echo COST_FAILED; exit 1; }

echo "=== x3_census $(date -Is) ==="
"$P" "$CODE/x3_census.py" --manifest "$RUN/step0/manifest-fit.jsonl" \
    --n-samples 50 --out "$RUN/step1/x3_norms.json" || { echo X3_FAILED; exit 1; }

echo "=== step0/1 chain done $(date -Is) ==="
