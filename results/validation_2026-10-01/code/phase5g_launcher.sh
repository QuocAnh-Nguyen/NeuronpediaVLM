#!/bin/bash
# P5g: repaired lens v2 - per-layer best shift from the EXTENDED selection (P4/P8/P16/P30/M4;
# the fit2 sweep's winners vary by band: early prefers P16/P30, L15-21 prefers P16, late prefers
# P4/P8). Composes J'_l = J_{l+delta_l} and scores it on the held-out bare (the census bias hurt
# the v1 re-paired lens mid-late; it is fitted for the correct pairing).
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

OUT=$V/step5; REP2=$OUT/repaired2
mkdir -p $REP2

if [ ! -f "$REP2/artifacts/provenance.json" ]; then
    $P $C/repaired_lens.py --src-lens-dir $V/s2-merged/artifacts \
        --selection $OUT/lens_zoo_shifts_fit2.json --out $REP2 || echo "REPAIR2_FAILED"
else
    echo "[repair2] present - skip"
fi

if [ -f "$REP2/artifacts/provenance.json" ] && [ ! -f "$OUT/lens_zoo_repaired2.json" ]; then
    run_step rep2zoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_repaired2.json --mask text --tags text \
        --lens repaired2=$REP2/artifacts || echo "REP2ZOO_FAILED"
else
    echo "[rep2zoo] present or repair missing - skip"
fi

echo "=== phase5g launcher done $(date -Is) ==="
