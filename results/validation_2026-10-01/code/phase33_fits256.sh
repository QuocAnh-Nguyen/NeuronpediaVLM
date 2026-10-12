#!/bin/bash
# P33: the 256-token caption-budget round - rebuild each model's corpus at
# --max-new-tokens 256 (SAME seeded image sample as the 64-token fits; only the budget
# changes), then re-fit via the OOM-resilient supervisors (staggered per model:
# shard 1 launches only after shard 0 commits), then merge. Models run sequentially.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
S=$R/results/validation_2026-10-01/code
C=/home/nvidia/vlm-run
P=/data/anhnq/envs/vlm_truth_py313/bin/python
LOGD=/home/nvidia/vlm-logs; mkdir -p $LOGD
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens:$S PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }

fit_model() {  # kind fits_subdir need
    local kind=$1 sub=$2 need=$3
    local F=$V/fits/$sub
    mkdir -p $F
    if [ ! -f "$F/manifest256.jsonl" ]; then
        g=$(wait_gpu $need "build-$kind") || return 1
        CUDA_VISIBLE_DEVICES=$g $P $C/build_manifest.py --model $kind --max-new-tokens 256 \
            --out $F/manifest256.jsonl || { echo "BUILD_${kind}_FAILED"; return 1; }
    else echo "[$kind manifest] present - skip"; fi
    if [ ! -f "$F/s0/artifacts/lens-text.pt" ]; then
        FIT_EXTRA="--manifest $F/manifest256.jsonl --dim-batch 3" \
            nohup setsid bash $C/fit_supervisor.sh $kind 0/2 $F/s0 $LOGD/fit256_${kind}_s0.log $need > $LOGD/sup256_${kind}_s0.log 2>&1 &
        echo "[$kind s0] supervisor launched - waiting for commit"
        for i in $(seq 1 240); do
            grep -q "survived the load window" $LOGD/sup256_${kind}_s0.log 2>/dev/null && break
            grep -qE "WAIT_TIMEOUT" $LOGD/sup256_${kind}_s0.log 2>/dev/null && { echo "S0_${kind}_STALLED"; return 1; }
            sleep 60
        done
        grep -q "survived the load window" $LOGD/sup256_${kind}_s0.log 2>/dev/null || { echo "S0_${kind}_NO_COMMIT"; return 1; }
    else echo "[$kind s0] present - skip"; fi
    if [ ! -f "$F/s1/artifacts/lens-text.pt" ]; then
        FIT_EXTRA="--manifest $F/manifest256.jsonl --dim-batch 3" \
            nohup setsid bash $C/fit_supervisor.sh $kind 1/2 $F/s1 $LOGD/fit256_${kind}_s1.log $need > $LOGD/sup256_${kind}_s1.log 2>&1 &
        echo "[$kind s1] supervisor launched"
    fi
    while :; do
        S0N=$(ls $F/s0/artifacts/lens-*.pt 2>/dev/null | wc -l)
        S1N=$(ls $F/s1/artifacts/lens-*.pt 2>/dev/null | wc -l)
        [ "$S0N" -ge 3 ] && [ "$S1N" -ge 3 ] && break
        sleep 120
    done
    if [ ! -f "$F/merged/artifacts/lens-text.pt" ]; then
        $P $R/scripts/fit_llava.py --model $kind --merge $F/s0 $F/s1 --out $F/merged > $LOGD/merge256_${kind}.log 2>&1 \
            && echo "FIT256_${kind}_MERGED" || echo "FIT256_${kind}_MERGE_FAILED"
    else echo "[$kind merged] present - skip"; fi
}

fit_model llava-7b llava-1.5-7b-hf 30000
fit_model qwen2vl qwen2-vl-7b-instruct 30000
fit_model llava-13b llava-1.5-13b-hf 45000
echo "=== phase33 fits256 done $(date -Is) ==="
