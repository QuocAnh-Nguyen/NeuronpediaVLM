#!/bin/bash
# P5d: the re-paired lens. Composes J'_l = J_{l+delta_l} with per-layer deltas selected on the
# FIT split (phase5c's shift sweep on half_a; no held-out leakage), then scores it on the 300-sample
# held-out - bare and with the census bias dir (the bias was fitted for the CORRECT pairing; the
# probe in phase5c tests whether it transfers to a shifted one).
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

OUT=$V/step5; REP=$OUT/repaired
mkdir -p $REP

if [ ! -f "$REP/artifacts/provenance.json" ]; then
    $P $C/repaired_lens.py --src-lens-dir $V/s2-merged/artifacts \
        --selection $OUT/lens_zoo_shifts_fit.json --out $REP || echo "REPAIR_FAILED"
else
    echo "[repair] present - skip"
fi

if [ -f "$REP/artifacts/provenance.json" ] && [ ! -f "$OUT/lens_zoo_repaired.json" ]; then
    run_step repzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_repaired.json --mask text --tags text \
        --lens repaired=$REP/artifacts || echo "REPZOO_FAILED"
else
    echo "[repzoo] present or repair missing - skip"
fi

if [ -f "$REP/artifacts/provenance.json" ] && [ ! -f "$OUT/lens_zoo_repaired_bias.json" ]; then
    run_step repbias 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_repaired_bias.json --mask text --tags text --bias-dir $V/step4d \
        --lens repaired=$REP/artifacts || echo "REPBIAS_FAILED"
else
    echo "[repbias] present or repair missing - skip"
fi

echo "=== phase5d launcher done $(date -Is) ==="
