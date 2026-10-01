#!/bin/bash
# Step 1 chain: real-checkpoint equivalence gate, then X6 (bf16 vs fp32, 8 samples each).
# The gate runs first and aborts the chain: no fit may run before it passes.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
IMAGES=$HOME/ai4life/phuongnh/vlm-truth/data/coco2014/val2014/val2014
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step1"
cd "$REPO" || exit 1
LAYERS=0,8,16,24,30

echo "=== gate $(date -Is) ==="
"$P" scripts/check_equivalence.py --backend hf-llava --model llava-hf/llava-1.5-7b-hf \
    --device cuda --dtype bfloat16 --image "$IMAGES/COCO_val2014_000000000139.jpg" \
    || { echo GATE_FAILED; exit 1; }

echo "=== X6 bf16 $(date -Is) ==="
"$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step0/manifest-fit.jsonl" \
    --limit 8 --layers "$LAYERS" --masks text,image,all --dim-batch 8 --dtype bfloat16 \
    --skip-first 1 --checkpoint-every 4 --out "$RUN/x6-bf16" \
    --notes "X6 bf16, 8 samples, layers $LAYERS" || { echo X6_BF16_FAILED; exit 1; }

echo "=== X6 fp32 $(date -Is) ==="
"$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step0/manifest-fit.jsonl" \
    --limit 8 --layers "$LAYERS" --masks text,image,all --dim-batch 4 --dtype float32 \
    --skip-first 1 --checkpoint-every 4 --out "$RUN/x6-fp32" \
    --notes "X6 fp32, 8 samples, layers $LAYERS, dim_batch 4" || { echo X6_FP32_FAILED; exit 1; }

echo "=== x6_compare $(date -Is) ==="
"$P" "$CODE/x6_compare.py" --dir-a "$RUN/x6-bf16/artifacts" --dir-b "$RUN/x6-fp32/artifacts" \
    --json "$RUN/step1/x6_dtype.json" || { echo COMPARE_FAILED; exit 1; }
rm -f "$RUN/x6-bf16/checkpoint.pt" "$RUN/x6-fp32/checkpoint.pt"
echo "=== step1 chain done $(date -Is) ==="
