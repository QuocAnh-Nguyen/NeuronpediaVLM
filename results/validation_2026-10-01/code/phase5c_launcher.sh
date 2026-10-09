#!/bin/bash
# P5c: re-paired transport experiment. The synth zoo found the FORWARD layer shift (+4) beats the
# correct pairing on the held-out (LQS +0.524 vs the full J's -0.102, both uncorrected) while the
# backward shift hurts - later maps transport earlier states better, so the per-layer (map, state)
# pairing is suboptimal. This launcher: (1) builds the shift family (+1/+2/+8; +4/-4 exist),
# (2) scores all five shifts on the FIT split (half_a, 50 samples) for per-layer selection with no
# held-out leakage, (3) probes shiftP4 + the census bias dir (does the bias correction transfer to
# a shifted pairing, or does it need a shifted census?).
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

OUT=$V/step5; SHIFTS=$OUT/synth-shifts
mkdir -p $OUT

if [ ! -f "$SHIFTS/variants.json" ]; then
    $P $C/synth_lenses.py --src-lens-dir $V/s2-merged/artifacts --out-root $SHIFTS \
        --variants "shiftP1,shiftP2,shiftP4,shiftP8,shiftM4" || echo "SHIFTS_FAILED"
else
    echo "[shifts] present - skip"
fi

if [ -f "$SHIFTS/variants.json" ] && [ ! -f "$OUT/lens_zoo_shifts_fit.json" ]; then
    run_step shiftfit 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step3/manifest-half-a.jsonl \
        --json $OUT/lens_zoo_shifts_fit.json --mask text --tags text \
        --lens shiftP1=$SHIFTS/shiftP1/artifacts --lens shiftP2=$SHIFTS/shiftP2/artifacts \
        --lens shiftP4=$SHIFTS/shiftP4/artifacts --lens shiftP8=$SHIFTS/shiftP8/artifacts \
        --lens shiftM4=$SHIFTS/shiftM4/artifacts || echo "SHIFTFIT_FAILED"
else
    echo "[shiftfit] present or shifts missing - skip"
fi

if [ -f "$SHIFTS/variants.json" ] && [ ! -f "$OUT/lens_zoo_shiftp4_bias.json" ]; then
    run_step shiftp4bias 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_shiftp4_bias.json --mask text --tags text --bias-dir $V/step4d \
        --lens shiftP4=$SHIFTS/shiftP4/artifacts || echo "SHIFTP4BIAS_FAILED"
else
    echo "[shiftp4bias] present or shifts missing - skip"
fi

echo "=== phase5c launcher done $(date -Is) ==="
