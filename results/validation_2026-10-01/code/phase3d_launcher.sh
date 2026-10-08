#!/bin/bash
# Phase-3d: L0-scale-fixed bias variant. The A/B showed the affine correction is a big win
# (LQS -0.10 -> +0.34; L20 ratio 64.6 -> 19.0; bump ratio 2.11x -> 0.68x) but L0 regressed
# (rank +15%, KL 14.2 -> 21.9) because the census's ratio-of-means scale is noisy-negative at
# the embedding layer (-3.19). This variant clamps every scale into [0.5, 2.0] (a scale is a
# magnitude correction; the sign flip at L0 is an artifact) and re-scores the merged lens.
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

BDIR=$V/step4d
mkdir -p $BDIR

if [ ! -f "$BDIR/bias-text.pt" ]; then
    $P - <<'PY'
import sys
sys.path.insert(0, "/data/anhnq/NeuronpediaVLM/src")
from vlm_lens.artifacts import load_bias, save_bias
src = "/data/anhnq/vlm-lens-out/validation/step4/bias-text.pt"
dst = "/data/anhnq/vlm-lens-out/validation/step4d/bias-text.pt"
payload, meta = load_bias(src)
clamped = {}
for layer, entry in payload.items():
    s = float(entry["scale"])
    clamped[layer] = {"bias": entry["bias"], "scale": min(2.0, max(0.5, s))}
save_bias(dst, {**clamped, "meta": {**meta, "scale_clamp": [0.5, 2.0]}})
print("clamped scales:", {k: round(float(v["scale"]), 3) for k, v in sorted(clamped.items(), key=lambda kv: int(kv[0]))})
PY
else
    echo "[clamp] bias present - skip"
fi

if [ ! -f "$V/step4/lens_zoo_biased_clamped.json" ]; then
    run_step zooclamp 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $V/step4/lens_zoo_biased_clamped.json --lens merged=$V/s2-merged/artifacts --bias-dir $BDIR \
        || echo "ZOOCLAMP_FAILED"
else
    echo "[zooclamp] present - skip"
fi

echo "=== phase3d launcher done $(date -Is) ==="
