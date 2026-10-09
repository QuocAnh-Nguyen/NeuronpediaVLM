#!/bin/bash
# P5d: tuned-lens-style KL calibration of the J-lens readout (zero Jacobian fits): fit per-layer
# output temperature + a refined scale multiplier by minimizing KL(lens || model) on the
# 40-sample fit split (half_a), then emit a calibrated bias dir and score the merged lens with
# it on the 300-sample held-out. The (m, T) grid collapses to one effective scalar c = s_l*m/T
# per layer (the split is a redundant reparameterization; c is the meaningful quantity).
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
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

OUT=$V/step5; CAL=$OUT/calib; CALDIR=$V/step5d
mkdir -p $OUT $CALDIR

if [ ! -f "$CAL/calib.json" ]; then
    run_step calib 20000 $P $C/calib_readout.py --lens-dir $V/s2-merged/artifacts \
        --bias-dir $V/step4d --manifest $V/step3/manifest-half-a.jsonl --mask text \
        --limit 40 --out $CAL/calib.json --dtype bfloat16 || echo "CALIB_FAILED"
else
    echo "[calib] present - skip"
fi

if [ -f "$CAL/calib.json" ] && [ ! -f "$CALDIR/bias-text.pt" ]; then
    $P $C/write_calibrated_bias.py --base-bias-dir $V/step4d --calib $CAL/calib.json \
        --out $CALDIR --mask text || echo "WRITER_FAILED"
fi

if [ -f "$CALDIR/bias-text.pt" ] && [ ! -f "$OUT/lens_zoo_cal.json" ]; then
    run_step calzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_cal.json --mask text --tags text --bias-dir $CALDIR \
        --lens merged=$V/s2-merged/artifacts || echo "CALZOO_FAILED"
fi

echo "=== phase5d launcher done $(date -Is) ==="
