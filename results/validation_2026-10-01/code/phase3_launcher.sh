#!/bin/bash
# Phase-3 launcher: moment census -> lens zoo eval -> x9b alpha sweep, each gated on a
# >=20 GiB GPU window (the Brev box is shared; windows are first-come-first-served).
# Skips a step whose output already exists (idempotent re-runs).
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
            echo "[$label] GPU $g >= ${need}MiB $(date -Is)"
            echo "$g"; return 0
        fi
        if [ "$waited" -ge 14400 ]; then
            echo "[$label] NO_GPU after 4h $(date -Is)"; return 1
        fi
        sleep 60; waited=$((waited + 60))
    done
}

run_step() {  # $1 = label, $2 = MiB needed, rest = command
    local label=$1 need=$2 g; shift 2
    g=$(wait_gpu "$need" "$label") || return 1
    CUDA_VISIBLE_DEVICES=$g "$@"
}

# 1) moment census -> bias/scale artifact for the affine correction
if [ ! -f "$V/step4/bias-text.pt" ]; then
    run_step census 20000 $P $C/moment_census.py --lens-dir $V/s2-merged/artifacts --mask text \
        --manifest $V/step0/manifest-fit.jsonl --out $V/step4/bias-text.pt --json $V/step4/bias-text.json \
        || echo "CENSUS_FAILED"
else
    echo "[census] bias-text.pt present - skip"
fi

# 2) lens zoo eval -> the data-scaling curve + LQS
if [ ! -f "$V/step4/lens_zoo.json" ]; then
    run_step zoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $V/step4/lens_zoo.json \
        --lens x1_all=$V/x1-all/artifacts --lens x1_text=$V/x1-text/artifacts \
        --lens half_a=$V/s2-half-a/artifacts --lens half_b=$V/s2-half-b/artifacts \
        --lens merged=$V/s2-merged/artifacts \
        --bias-dir $V/step4 \
        || echo "ZOO_FAILED"
else
    echo "[zoo] lens_zoo.json present - skip"
fi

# 3) alpha sweep -> the minimal effective edit strength
if [ ! -f "$V/step4/x9b_alpha_sweep.json" ]; then
    run_step x9b 22000 $P $C/x9b_alpha_sweep.py --lens-dir $V/s2-merged/artifacts \
        --manifest $V/step0/manifest-heldout.jsonl --n-samples 10 \
        --norms-json $V/step1/x3_norms.json --json $V/step4/x9b_alpha_sweep.json \
        || echo "X9B_FAILED"
else
    echo "[x9b] x9b_alpha_sweep.json present - skip"
fi

echo "=== phase3 launcher done $(date -Is) ==="
