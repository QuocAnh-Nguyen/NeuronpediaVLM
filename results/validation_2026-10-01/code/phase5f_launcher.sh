#!/bin/bash
# P5f: the one-matrix lens on the HELD-OUT. The extended fit-split sweep is monotone in the
# forward shift through P30 (P4 +0.465 < P8 +0.622 < P16 +0.735 < P30 +1.072 LQS on half_a) -
# J_30 applied to EVERY layer (one [d,d] matrix, 31x smaller than the full set) beats everything
# uncorrected. This scores shiftP30/shiftP16 on the 300-sample held-out, bare and with the census
# bias dir (the bias was fitted for the correct pairing; does it transfer to the one-matrix lens?).
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

if [ ! -f "$OUT/lens_zoo_onematrix.json" ]; then
    run_step omzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_onematrix.json --mask text --tags text \
        --lens shiftP30=$SHIFTS2/shiftP30/artifacts --lens shiftP16=$SHIFTS2/shiftP16/artifacts \
        || echo "OMZOO_FAILED"
else
    echo "[omzoo] present - skip"
fi

if [ ! -f "$OUT/lens_zoo_onematrix_bias.json" ]; then
    run_step ombias 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_onematrix_bias.json --mask text --tags text --bias-dir $V/step4d \
        --lens shiftP30=$SHIFTS2/shiftP30/artifacts --lens shiftP16=$SHIFTS2/shiftP16/artifacts \
        || echo "OMBIAS_FAILED"
else
    echo "[ombias] present - skip"
fi

echo "=== phase5f launcher done $(date -Is) ==="
