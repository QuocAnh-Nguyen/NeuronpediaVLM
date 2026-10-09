#!/bin/bash
# P5f: readout-span overlap diagnostic (R2/R3 priority-5, zero fits): how much of each fitted
# J_l lies in W_U's dominant readout directions, plus cross-layer overlap of J_l's top singular
# directions — explains the alpha-sweep's rare concept-directed edits (concept subspaces
# ~orthogonal to the readout span) and quantifies mid-layer basis drift. One GPU dump of W_U,
# then CPU-only analysis.
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

OUT=$V/step5
mkdir -p $OUT

if [ ! -f "$OUT/W_U.pt" ]; then
    g=$(wait_gpu 20000 wudump) || exit 1
    CUDA_VISIBLE_DEVICES=$g $P - <<'PYEOF'
import sys
sys.path.insert(0, "/data/anhnq/NeuronpediaVLM/src")
import torch
from vlm_lens.models.llava import LlavaLensModel
model = LlavaLensModel.from_pretrained(dtype=torch.bfloat16, device="cuda", local_files_only=True)
w = model.unembed_weight().detach().float().cpu()
torch.save(w, "/data/anhnq/vlm-lens-out/validation/step5/W_U.pt")
print("saved W_U", tuple(w.shape))
PYEOF
else
    echo "[wudump] present - skip"
fi

if [ -f "$OUT/W_U.pt" ] && [ ! -f "$OUT/span_overlap.json" ]; then
    $P $C/readout_span_overlap.py --lens-dir $V/s2-merged/artifacts --mask text \
        --unembed $OUT/W_U.pt --out $OUT/span_overlap.json --digest $OUT/span_overlap_digest.txt \
        || echo "SPANOVERLAP_FAILED"
else
    echo "[spanoverlap] present or W_U missing - skip"
fi

echo "=== phase5f launcher done $(date -Is) ==="
