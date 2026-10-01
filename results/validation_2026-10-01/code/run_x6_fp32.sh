#!/bin/bash
# X6 fp32 leg (TF32). Settings match x6-bf16's provenance exactly (manifest
# step0/manifest-fit.jsonl, 8 samples, layers 0,8,16,24,30, target 31, skip_first 1,
# dim_batch 8, max_seq_len 1536), except dtype.
#
# torch defaults allow_tf32=False and true-fp32 cuBLAS on H100 runs on CUDA cores (~15x
# slower than bf16 tensor cores; measured ~10 min/sample), which does not fit the campaign
# budget. TF32 keeps fp32 storage/accumulation with a 10-bit mantissa and is the
# register's intended "fp32/tf32" comparison (D20/X6).
#
# Memory: fp32 weights are ~29 GiB; the 06:48 attempt OOMed with 47.65 GiB allocated by
# PyTorch and 119 MiB free (222 MiB request), so the leg needs the GPU nearly to itself.
# Exit 3 (retryable) when free memory is below the guard; run_x6_retry.sh retries.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
LAYERS=0,8,16,24,30
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO" || exit 1

MIN_FREE_MIB=${MIN_FREE_MIB:-55000}
FREE_MIB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
FREE_MIB=${FREE_MIB:-0}
if [ "$FREE_MIB" -lt "$MIN_FREE_MIB" ]; then
    echo "X6 fp32 leg skipped: free GPU memory ${FREE_MIB} MiB < ${MIN_FREE_MIB} MiB"
    exit 3
fi

mkdir -p "$RUN/x6-fp32"
echo "=== X6 fp32+TF32 fit $(date -Is) free=${FREE_MIB}MiB ==="
"$P" scripts/fit_llava.py --backend hf-llava --manifest "$RUN/step0/manifest-fit.jsonl" \
    --layers "$LAYERS" --target-layer 31 --masks text,image,all --dim-batch 8 --skip-first 1 \
    --max-seq-len 1536 --dtype float32 --allow-tf32 --limit 8 --seed 0 \
    --checkpoint-every 4 --out "$RUN/x6-fp32" \
    --notes "X6 fp32+TF32, 8 samples, layers $LAYERS, dim_batch 8 (matches x6-bf16)" \
    || { echo X6_FP32_FAILED; exit 1; }

echo "=== x6_compare $(date -Is) ==="
"$P" "$CODE/x6_compare.py" --dir-a "$RUN/x6-bf16/artifacts" --dir-b "$RUN/x6-fp32/artifacts" \
    --json "$RUN/step1/x6_dtype.json" || { echo COMPARE_FAILED; exit 1; }
rm -f "$RUN/x6-fp32/checkpoint.pt"
echo "=== X6 (fp32+TF32) done $(date -Is) ==="
