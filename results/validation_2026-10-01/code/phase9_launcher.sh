#!/bin/bash
# P9: the J-class INTRA-FIT learning curve (the user's ask: record performance while fitting to
# see whether it is really going up). A bf16 n=40 fit (layers all, target_mask=text, the first 40
# rows of the disjoint scaling manifest) with --probe-every 5 --probe-manifest
# step0/manifest-heldout.jsonl --probe-n 16: every 5 samples the running mean-J is scored on 16
# held-out samples (kl/rank/top1 per layer) and one compact line lands in the fit log - the
# learning curve. bf16 because the curve, not the artifact, is the deliverable (~4 h).
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

OUT=$V/step9-jprobe
mkdir -p $OUT

if [ ! -f "$OUT/artifacts/provenance.json" ] && [ -f "$V/manifest-scaling.jsonl" ]; then
    run_step jprobe 20000 $P $R/scripts/fit_llava.py --backend hf-llava \
        --manifest $V/manifest-scaling.jsonl --limit 40 --layers all --masks text,image,all \
        --target-mask text --dim-batch 1 --dtype bfloat16 --checkpoint-every 5 \
        --probe-every 5 --probe-manifest $V/step0/manifest-heldout.jsonl --probe-n 16 \
        --out "$OUT" --notes "intra-fit probe curve demo (bf16 n=40, probe every 5)" \
        || echo "JPROBE_FAILED"
else
    echo "[jprobe] present or manifest missing - skip"
fi

echo "=== phase9 launcher done $(date -Is) ==="
