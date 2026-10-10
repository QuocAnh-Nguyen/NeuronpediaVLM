#!/bin/bash
# P10 (x12): cross-image residual patching - localize WHERE visual content commits to the
# caption (zero fitting; the pre-emission workspace). Patches A's image-token rows with B's at
# each swept layer; measures caption flips + object-word swaps; the text-variant is the control.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }
OUT=$V/step10; mkdir -p $OUT
if [ ! -f "$OUT/x12_cross_image_patch.json" ]; then
    run_step x12 20000 $P $C/x12_cross_image_patch.py \
        --images-dir /data/baodq/coco2014/val2014 \
        --instances-json /data/baodq/coco2014/annotations/instances_val2014.json \
        --n-pairs 8 --layers 0,8,12,16,20,24,28,30 --variants image,image-last,text \
        --max-new-tokens 32 --backend hf-llava --dtype bfloat16 \
        --json $OUT/x12_cross_image_patch.json || echo "X12_FAILED"
else
    echo "[x12] present - skip"
fi
echo "=== phase10 launcher done $(date -Is) ==="
