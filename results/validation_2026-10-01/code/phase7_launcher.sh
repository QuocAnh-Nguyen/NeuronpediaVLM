#!/bin/bash
# P7: the target_mask=text confirmation. The phase3c 23.7-h refit completed 20:44 but its zoo
# was terminated 8 s in by an unrelated SIGTERM. Score the tmtext50 lens BARE (the x1_text-
# comparable scaling-run comparison) and with the best calibration payload (step5e: census
# bias+scale + logit_bias) on the 300-sample held-out - the D40 x1_text lever composed with
# the P1/P5 levers.
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

OUT=$V/step4; TM=$V/tmtext-half-a/artifacts; BEST=$V/step5e

if [ -f "$TM/provenance.json" ] && [ ! -f "$OUT/lens_zoo_tmtext.json" ]; then
    run_step tmzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_tmtext.json --mask text --tags text --lens tmtext50=$TM \
        || echo "TMZOO_FAILED"
else
    echo "[tmzoo] present or fit missing - skip"
fi

if [ -f "$TM/provenance.json" ] && [ -f "$BEST/bias-text.pt" ] \
    && [ ! -f "$OUT/lens_zoo_tmtext_cal.json" ]; then
    run_step tmzoo2 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_tmtext_cal.json --mask text --tags text --bias-dir $BEST \
        --lens tmtext50cal=$TM \
        || echo "TMZOO2_FAILED"
else
    echo "[tmzoo2] present or inputs missing - skip"
fi

echo "=== phase7 launcher done $(date -Is) ==="
