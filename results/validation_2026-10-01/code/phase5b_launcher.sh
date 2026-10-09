#!/bin/bash
# P5b: synthetic-lens zoo. Scores six structural variants of the fitted 100-sample merged lens
# (alphaI = scaled identity, diag, rank64/rank256 truncated SVD, shiftP4/shiftM4 layer-shifted)
# on the 300-sample held-out with the standard zoo evaluator - zero new fits. Tests how much of
# the J-map carries the signal (alphaI vs full J), whether off-diagonal transport matters (diag),
# whether low-rank suffices (rank-r), and whether the mid-layer bump lives in the state or the
# transport (shift variants). The synth build is CPU-only; only the zoo needs a GPU (>=20 GiB).
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

OUT=$V/step5; SYNTH=$OUT/synth
mkdir -p $OUT

if [ ! -f "$SYNTH/variants.json" ]; then
    $P $C/synth_lenses.py --src-lens-dir $V/s2-merged/artifacts --out-root $SYNTH \
        || echo "SYNTH_FAILED"
else
    echo "[synth] present - skip"
fi

if [ -f "$SYNTH/variants.json" ] && [ ! -f "$OUT/lens_zoo_synth.json" ]; then
    run_step synthzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_synth.json --mask text --tags text \
        --lens alphaI=$SYNTH/alphaI/artifacts --lens diag=$SYNTH/diag/artifacts \
        --lens rank64=$SYNTH/rank64/artifacts --lens rank256=$SYNTH/rank256/artifacts \
        --lens shiftP4=$SYNTH/shiftP4/artifacts --lens shiftM4=$SYNTH/shiftM4/artifacts \
        || echo "SYNTHZOO_FAILED"
else
    echo "[synthzoo] present or synth missing - skip"
fi

echo "=== phase5b launcher done $(date -Is) ==="
