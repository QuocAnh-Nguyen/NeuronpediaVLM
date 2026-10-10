#!/bin/bash
# P16 (x18): POPE pre-generation handoff - can one forward at the last prompt position with the
# calibrated lens separate present/absent objects (and flag the model's own hallucinated yes)?
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }
OUT=$V/step11; mkdir -p $OUT
if [ ! -f "$OUT/x18_pope_handoff.json" ]; then
    run_step x18 20000 $P $C/x18_pope_handoff.py \
        --lens-dir $V/s2-merged/artifacts --bias-dir $V/step5e \
        --images-dir /data/baodq/coco2014/val2014 \
        --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
        --layers 20,24,28,30 --n-images 50 --json $OUT/x18_pope_handoff.json || echo "X18_FAILED"
else
    echo "[x18] present - skip"
fi
echo "=== phase16 launcher done $(date -Is) ==="
