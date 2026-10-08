#!/bin/bash
# Phase-3b launcher: bf16 moment census (the fp32 attempt OOM'd on a 20-GiB window) and a
# merged-only WITH-BIAS zoo run (the bias A/B against the unbiased lens_zoo.json produced by
# the phase3 launcher). Each step window-gated; idempotent (skips when its output exists).
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pick_gpu() {  # $1 = MiB needed; echoes the freest GPU index with >= $1 free
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}

wait_gpu() {  # $1 = MiB, $2 = label; waits up to 4 h, echoes the GPU index
    local need=$1 label=$2 waited=0 g
    while :; do
        g=$(pick_gpu "$need")
        if [ -n "$g" ]; then
            echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2
            echo "$g"; return 0
        fi
        if [ "$waited" -ge 14400 ]; then
            echo "[$label] NO_GPU after 4h $(date -Is)" >&2; return 1
        fi
        sleep 60; waited=$((waited + 60))
    done
}

run_step() {  # $1 = label, $2 = MiB needed, rest = command
    local label=$1 need=$2 g; shift 2
    g=$(wait_gpu "$need" "$label") || return 1
    CUDA_VISIBLE_DEVICES=$g "$@"
}

if [ ! -f "$V/step4/bias-text.pt" ]; then
    run_step census 20000 $P $C/moment_census.py --lens-dir $V/s2-merged/artifacts --mask text \
        --manifest $V/step0/manifest-fit.jsonl --out $V/step4/bias-text.pt --json $V/step4/bias-text.json \
        --dtype bfloat16 \
        || echo "CENSUS_FAILED"
else
    echo "[census] bias-text.pt present - skip"
fi

if [ -f "$V/step4/bias-text.pt" ] && [ ! -f "$V/step4/lens_zoo_biased.json" ]; then
    run_step zoobias 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $V/step4/lens_zoo_biased.json --lens merged=$V/s2-merged/artifacts --bias-dir $V/step4 \
        || echo "ZOOBIA_FAILED"
else
    echo "[zoobias] lens_zoo_biased.json present or bias missing - skip"
fi

echo "=== phase3b launcher done $(date -Is) ==="
