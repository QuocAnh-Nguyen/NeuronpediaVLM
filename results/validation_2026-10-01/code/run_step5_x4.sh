#!/bin/bash
# Step 5 chain (X4, conditional): the same 10-image shard fitted under four source-mask
# boundaries (skip_first = 1/8/16/32) to measure the V3 boundary sensitivity empirically.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
N_SHARD=${N_SHARD:-10}
LAYERS=0,8,16,24,30
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step5"
cd "$REPO" || exit 1

VARIANTS=()
for sf in 1 8 16 32; do
    echo "=== X4 fit skip_first=$sf $(date -Is) ==="
    "$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step3/manifest-half-a.jsonl" \
        --limit "$N_SHARD" --layers "$LAYERS" --masks text,image,all --dim-batch 8 --dtype bfloat16 \
        --skip-first "$sf" --checkpoint-every 5 --out "$RUN/x4-sf$sf" \
        --notes "X4 shard $N_SHARD, skip_first=$sf" || { echo X4_${sf}_FAILED; exit 1; }
    VARIANTS+=("$sf=$RUN/x4-sf$sf/artifacts")
done

echo "=== x4_eval $(date -Is) ==="
"$P" "$CODE/x4_eval.py" --variants "${VARIANTS[@]}" \
    --heldout-manifest "$RUN/step0/manifest-heldout.jsonl" \
    --json "$RUN/step5/x4_skipfirst.json" || { echo X4_EVAL_FAILED; exit 1; }

rm -f "$RUN"/x4-sf*/checkpoint.pt
echo "=== step5 chain done $(date -Is) ==="
