#!/bin/bash
# P21: scale-up on the HOME filesystem - /data is 100% full (shared 18T volume; other tenants),
# so outputs+logs go to /home/nvidia (31G free). Reads (code, repo, lens artifacts, coco, envs)
# stay on /data. PYTHONDONTWRITEBYTECODE guards .pyc writes into the read-only /data code dir.
# Runs x15 n200 then x17 n200 sequentially with the free-memory guard.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
OUT=/home/nvidia/vlm-lens-out/validation/step11
LOGD=/home/nvidia/vlm-logs
mkdir -p $OUT $LOGD
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }

if [ ! -f "$OUT/x15_hesitation_n200.json" ]; then
    run_step x15n200 20000 $P $C/x15_hesitation.py \
        --lens-dir $V/s2-merged/artifacts --bias-dir $V/step5e \
        --images-dir /data/baodq/coco2014/val2014 \
        --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
        --n-images 200 --max-new-tokens 40 \
        --json $OUT/x15_hesitation_n200.json || echo "X15N200_FAILED"
else
    echo "[x15n200] present - skip"
fi

if [ ! -f "$OUT/x17_attention_mirror_n200.json" ]; then
    run_step x17n200 20000 $P $C/x17_attention_mirror.py \
        --lens-dir $V/s2-merged/artifacts \
        --images-dir /data/baodq/coco2014/val2014 \
        --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
        --n-images 200 --max-new-tokens 40 \
        --json $OUT/x17_attention_mirror_n200.json || echo "X17N200_FAILED"
else
    echo "[x17n200] present - skip"
fi
echo "=== phase21 home scale-up done $(date -Is) ==="
