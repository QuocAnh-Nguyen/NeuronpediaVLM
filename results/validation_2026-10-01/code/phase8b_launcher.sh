#!/bin/bash
# P8b: build the translator-scaling corpus (n=1000, disjoint from every existing fit/held-out
# manifest) - the prerequisite for the phase8 translator-scaling fits. Captions are
# model-generated (the deployment path), bf16, greedy; ~40 min on a free GPU.
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

IMAGES=/data/baodq/coco2014/val2014
EXCL=$(ls $V/step0/*.jsonl $V/step3/*.jsonl 2>/dev/null | tr '\n' ',' | sed 's/,$//')
echo "images dir: $IMAGES ($(ls $IMAGES 2>/dev/null | wc -l) jpgs)"
echo "exclude manifests: $EXCL"

if [ ! -f "$V/manifest-scaling.jsonl" ]; then
    run_step build 20000 $P $C/build_caption_manifest_n.py --images-dir $IMAGES --n 1000 \
        --out $V/manifest-scaling.jsonl --exclude-manifests "$EXCL" \
        --backend hf-llava --dtype bfloat16 \
        || echo "BUILD_FAILED"
else
    echo "[build] present - skip"
fi

echo "=== phase8b launcher done $(date -Is) ==="
