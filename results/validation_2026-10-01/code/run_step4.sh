#!/bin/bash
# Step 4 chain (X1): fit the same 20-image shard with target_mask=all and target_mask=text,
# compare the text-block lens rows, and run the single-sample bit-level check.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
N_SHARD=${N_SHARD:-20}
LAYERS=0,8,16,24,30
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step4"
cd "$REPO" || exit 1

for variant in all text; do
    echo "=== X1 fit target_mask=$variant $(date -Is) ==="
    "$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step3/manifest-half-a.jsonl" \
        --limit "$N_SHARD" --layers "$LAYERS" --masks text,image,all --dim-batch 8 --dtype bfloat16 \
        --skip-first 1 --target-mask "$variant" --checkpoint-every 10 \
        --out "$RUN/x1-$variant" --notes "X1 shard $N_SHARD, target_mask=$variant" \
        || { echo X1_${variant}_FAILED; exit 1; }
done

echo "=== x1_compare $(date -Is) ==="
"$P" "$CODE/x1_compare.py" --dir-all "$RUN/x1-all/artifacts" --dir-text "$RUN/x1-text/artifacts" \
    --manifest "$RUN/step3/manifest-half-a.jsonl" --sample-index 0 --layers "$LAYERS" \
    --json "$RUN/step4/x1_targetmask.json" || { echo X1_COMPARE_FAILED; exit 1; }

rm -f "$RUN/x1-all/checkpoint.pt" "$RUN/x1-text/checkpoint.pt"
echo "=== step4 chain done $(date -Is) ==="
