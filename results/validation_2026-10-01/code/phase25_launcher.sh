#!/bin/bash
# P25 (x22): internal-reasoning two-hop swaps (+ depth precedence), flexible-generalization
# swaps (vendored 192 trials), and the layer-band structure statistics.
# NOTE: x22 uses the RAW J-lens only (the paper's convention) - no --bias-dir flag.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
S=$R/results/validation_2026-10-01/code
C=/home/nvidia/vlm-run
P=/data/anhnq/envs/vlm_truth_py313/bin/python
OUT=/home/nvidia/vlm-lens-out/validation/step12
LOGD=/home/nvidia/vlm-logs
mkdir -p $OUT $LOGD
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens:$S PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }

for arm in multihop flexgen structure; do
    if [ ! -f "$OUT/x22_reasoning_structure_$arm.json" ]; then
        run_step x22-$arm 20000 $P $C/x22_reasoning_structure.py --arm $arm \
            --lens-dir $V/s2-merged/artifacts \
            --json $OUT/x22_reasoning_structure_$arm.json || echo "X22_${arm}_FAILED"
    else
        echo "[x22-$arm] present - skip"
    fi
done
echo "=== phase25 launcher done $(date -Is) ==="
