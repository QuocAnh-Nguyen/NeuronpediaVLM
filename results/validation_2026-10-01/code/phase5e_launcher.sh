#!/bin/bash
# P5e: extended shift selection. The fit-split sweep was MONOTONE in the forward shift
# (P1 +0.158 < P2 +0.255 < P4 +0.465 < P8 +0.622 LQS on half_a) - test whether larger shifts
# keep helping: build shiftP16/shiftP30 (P30 = J_30 for every layer, the one-matrix lens - a
# 31x storage/compute reduction if it wins) and re-run the selection sweep with the full family
# on the fit split.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pick_gpu() {
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}

wait_gpu() {
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

run_step() {
    local label=$1 need=$2 g; shift 2
    g=$(wait_gpu "$need" "$label") || return 1
    CUDA_VISIBLE_DEVICES=$g "$@"
}

OUT=$V/step5; SHIFTS2=$OUT/synth-shifts2
mkdir -p $OUT

if [ ! -f "$SHIFTS2/variants.json" ]; then
    $P $C/synth_lenses.py --src-lens-dir $V/s2-merged/artifacts --out-root $SHIFTS2 \
        --variants "shiftP8,shiftP16,shiftP30,shiftP4,shiftM4" || echo "SHIFTS2_FAILED"
else
    echo "[shifts2] present - skip"
fi

if [ -f "$SHIFTS2/variants.json" ] && [ ! -f "$OUT/lens_zoo_shifts_fit2.json" ]; then
    run_step shiftfit2 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step3/manifest-half-a.jsonl \
        --json $OUT/lens_zoo_shifts_fit2.json --mask text --tags text \
        --lens shiftP4=$SHIFTS2/shiftP4/artifacts --lens shiftP8=$SHIFTS2/shiftP8/artifacts \
        --lens shiftP16=$SHIFTS2/shiftP16/artifacts --lens shiftP30=$SHIFTS2/shiftP30/artifacts \
        --lens shiftM4=$SHIFTS2/shiftM4/artifacts || echo "SHIFTFIT2_FAILED"
else
    echo "[shiftfit2] present or shifts missing - skip"
fi

echo "=== phase5e launcher done $(date -Is) ==="
