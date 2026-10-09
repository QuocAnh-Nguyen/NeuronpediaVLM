#!/bin/bash
# P5e: logit-space bias calibration (the R2/R3-endorsed lever): apply per-layer
# logit_bias = mean(z_lens - z_model) (the calib run's gap file, computed at the same base
# config) on top of the census bias+scale, and score the merged lens on the 300-sample
# held-out. Tests whether the additive vocab correction generalizes (the tuned-lens
# marginal-bias fix) or overfits (2482 fit positions vs 32k vocab dims).
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

OUT=$V/step5; LB=$V/step5e
mkdir -p $OUT $LB

if [ ! -f "$LB/bias-text.pt" ]; then
    $P - <<'PYEOF'
import sys
sys.path.insert(0, "/data/anhnq/NeuronpediaVLM/src")
import torch
from vlm_lens.artifacts import load_bias, save_bias
gap_payload = torch.load(
    "/data/anhnq/vlm-lens-out/validation/step5/calib/calib_logitgap.pt", weights_only=True
)
if isinstance(gap_payload, dict) and "meta" in gap_payload:
    gap_payload = {k: v for k, v in gap_payload.items() if k != "meta"}
base_payload, base_meta = load_bias("/data/anhnq/vlm-lens-out/validation/step4d/bias-text.pt")
out = {}
for layer, entry in base_payload.items():
    key = int(layer) if isinstance(layer, str) else layer
    g = gap_payload.get(str(key), gap_payload.get(key))
    if g is None:
        out[layer] = entry
        continue
    # subtract the lens-model gap: E[z_corrected] = E[z_model]
    out[layer] = {"bias": entry["bias"], "scale": entry["scale"], "logit_bias": (-g).float().cpu()}
save_bias(
    "/data/anhnq/vlm-lens-out/validation/step5e/bias-text.pt",
    {**out, "meta": {**base_meta, "logit_bias_source": "calib_logitgap (mean z_lens - z_model, base config)"}},
)
print("wrote logit-bias bias file for", len(out), "layers")
PYEOF
else
    echo "[lb] present - skip"
fi

if [ -f "$LB/bias-text.pt" ] && [ ! -f "$OUT/lens_zoo_logitbias.json" ]; then
    run_step lbzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_logitbias.json --mask text --tags text --bias-dir $LB \
        --lens merged=$V/s2-merged/artifacts || echo "LBZOO_FAILED"
fi

echo "=== phase5e launcher done $(date -Is) ==="
