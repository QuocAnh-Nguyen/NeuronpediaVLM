#!/bin/bash
# P12 (x13): the CAUSAL CURE TEST - mean-replacement ablation of hallucinated object directions
# (COCO instances ground truth), per layer 16/24/28/30, with a grounded-category control arm.
# Zero fitting; the J-lens vectors only supply edit directions.
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
if [ ! -f "$OUT/x13_cure_test.json" ]; then
    run_step x13 20000 $P $C/x13_cure_test.py \
        --json $OUT/x13_cure_test.json --n-images 40 --layers 16,24,28,30 \
        --max-new-tokens 60 --backend hf-llava || echo "X13_FAILED"
else
    echo "[x13] present - skip"
fi
echo "=== phase12 launcher done $(date -Is) ==="
