#!/bin/bash
# P22 (x19): image-level hallucination-risk experiment - mean/max per-step lens entropy, the
# pre-generation true-category gap, and their standardized fusion; n=200. Home-redirected
# (the /data volume is 100% full; outputs+logs on /home/nvidia). The x19 script lives in
# /home/nvidia/vlm-run (C) because /data is full; its sibling imports (x15/x18) resolve via
# the PYTHONPATH entry for the repo code dir (S).
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
C=/home/nvidia/vlm-run
S=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
OUT=/home/nvidia/vlm-lens-out/validation/step11
LOGD=/home/nvidia/vlm-logs
mkdir -p $OUT $LOGD
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens:$S PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }

if [ ! -f "$OUT/x19_halluc_risk.json" ]; then
    run_step x19 20000 $P $C/x19_halluc_risk.py \
        --lens-dir $V/s2-merged/artifacts --bias-dir $V/step5e \
        --images-dir /data/baodq/coco2014/val2014 \
        --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
        --n-images 200 --layers 20,24,28,30 \
        --json $OUT/x19_halluc_risk.json || echo "X19_FAILED"
else
    echo "[x19] present - skip"
fi
echo "=== phase22 launcher done $(date -Is) ==="
