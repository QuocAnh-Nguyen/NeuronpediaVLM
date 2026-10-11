#!/bin/bash
# P28: real-checkpoint equivalence gates for the two NEW architectures before any fit.
# llava-13b reuses the verified LlavaLensModel (40-layer sanity); qwen2vl exercises the new
# Qwen2VLLensWrapper on the real 28-layer model. Exit markers: GATE_<KIND>_PASS/FAIL.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
S=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_gate() { local kind=$1 need=$2 g; g=$(wait_gpu "$need" "gate-$kind") || return 1; CUDA_VISIBLE_DEVICES=$g $P $R/scripts/check_equivalence.py --model $kind --backend hf-llava \
    --device cuda --dtype bfloat16 \
    --image /data/baodq/coco2014/val2014/COCO_val2014_000000036773.jpg; }
for kind in llava-13b qwen2vl; do
    if run_gate $kind 26000 | tee /tmp/gate_$kind.log | grep -q "EQUIVALENCE PASS"; then
        echo "GATE_${kind}_PASS"
    else
        echo "GATE_${kind}_FAIL"; tail -5 /tmp/gate_$kind.log
    fi
done
echo "=== phase28 gates done $(date -Is) ==="
