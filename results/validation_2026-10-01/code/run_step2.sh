#!/bin/bash
# Step 2 chain: S1 text-only control fit (WikiText train, skip_first=16) + held-out scoring.
# DTYPE/DIMBATCH come from the X6 verdict; N_PROMPTS is cut by the rule-3 budget check.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
DIMBATCH=${DIMBATCH:-8}
N_PROMPTS=${N_PROMPTS:-100}
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step2"
cd "$REPO" || exit 1

echo "=== S1 fit $(date -Is) dim_batch=$DIMBATCH n_prompts=$N_PROMPTS ==="
"$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step0/manifest-text-fit.jsonl" \
    --layers all --masks all --dim-batch "$DIMBATCH" --dtype bfloat16 --skip-first 16 \
    --checkpoint-every 5 --limit "$N_PROMPTS" --out "$RUN/s1-text" \
    --notes "S1 WikiText-103 train, skip_first=16, bf16" || { echo S1_FIT_FAILED; exit 1; }

echo "=== S1 score $(date -Is) ==="
"$P" "$CODE/s1_score.py" --lens-dir "$RUN/s1-text/artifacts" --mask all \
    --heldout-manifest "$RUN/step0/manifest-text-heldout.jsonl" \
    --fit-manifest "$RUN/step0/manifest-text-fit.jsonl" \
    --json "$RUN/step2/s1_score.json" || { echo S1_SCORE_FAILED; exit 1; }

echo "=== step2 chain done $(date -Is) ==="
