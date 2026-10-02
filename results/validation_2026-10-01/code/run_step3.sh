#!/bin/bash
# Step 3 chain: S2 caption fits (question halves A and B, 50 samples each), merge, held-out
# evaluation with tags/placeholder modes, A4/X8 half cross-scoring and A4/E6 cross-corpus.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
DIMBATCH=${DIMBATCH:-8}
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step3"
cd "$REPO" || exit 1

echo "=== split halves $(date -Is) ==="
"$P" "$CODE/split_halves.py" --manifest "$RUN/step0/manifest-fit.jsonl" \
    --split-json "$RUN/step0/corpus-split.json" --out-dir "$RUN/step3" || { echo SPLIT_FAILED; exit 1; }

# Dtype: fp32 weights + TF32 matmuls (D10/D18). The X6 verdict (step1/x6_dtype.json) is
# use_fp32: true - per-layer medians are 1.2-1.5 % but L0's max is 49 % - and the fp32+TF32
# leg cost 2486.5 s for 8 samples against 2205.8 s for bf16 (1.13x), so the pre-registered
# rule is affordable on the production caption lens.
for half in a b; do
    echo "=== S2 fit half $half $(date -Is) ==="
    "$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step3/manifest-half-$half.jsonl" \
        --layers all --masks text,image,all --dim-batch "$DIMBATCH" --dtype float32 --allow-tf32 \
        --checkpoint-every 5 --out "$RUN/s2-half-$half" \
        --notes "S2 caption pilot, question half $half, 50 samples, skip_first=1, fp32+TF32 (X6)" \
        || { echo S2_HALF_${half}_FAILED; exit 1; }
done

echo "=== merge halves $(date -Is) ==="
"$P" scripts/fit_llava.py --merge "$RUN/s2-half-a" "$RUN/s2-half-b" \
    --out "$RUN/s2-merged" --notes "S2 merged 100-sample caption lens (halves A+B)" \
    || { echo MERGE_FAILED; exit 1; }

echo "=== s2_eval $(date -Is) ==="
"$P" "$CODE/s2_eval.py" --main-lens-dir "$RUN/s2-merged/artifacts" \
    --heldout-manifest "$RUN/step0/manifest-heldout.jsonl" \
    --half-lens-a "$RUN/s2-half-a/artifacts" --half-lens-b "$RUN/s2-half-b/artifacts" \
    --split-json "$RUN/step0/corpus-split.json" \
    --text-lens-dir "$RUN/s1-text/artifacts" --text-heldout "$RUN/step0/manifest-text-heldout.jsonl" \
    --json "$RUN/step3/s2_eval.json" || { echo S2_EVAL_FAILED; exit 1; }

echo "=== step3 chain done $(date -Is) ==="
