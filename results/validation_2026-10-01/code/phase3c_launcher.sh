#!/bin/bash
# Phase-3c: confirm the target_mask=text improvement. In the n-scaling run, the 20-sample
# target_mask=text fit (x1_text) beat every all-target lens: LQS +0.116 vs -0.03..-0.25, and
# the L30 rank ratio 2.61 vs 4.70-5.69 - i.e. excluding the image block's own next-image-token
# targets (V1's 24-255x cotangent mass, X1) from the FIT improves the text-disposition lens.
# This refits half_a's 50-sample manifest with target_mask=text at all layers and scores it on
# the shared held-out against the logit baseline. Window-gated (the fp32 fit needs ~31 GiB).
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

pick_gpu() {  # $1 = MiB needed; echoes the freest GPU index with >= $1 free
    nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | \
        awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'
}

wait_gpu() {  # $1 = MiB, $2 = label; waits up to 4 h, echoes the GPU index
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

run_step() {  # $1 = label, $2 = MiB needed, rest = command
    local label=$1 need=$2 g; shift 2
    g=$(wait_gpu "$need" "$label") || return 1
    CUDA_VISIBLE_DEVICES=$g "$@"
}

OUT=$V/tmtext-half-a
cd $R

if [ ! -f "$OUT/artifacts/provenance.json" ]; then
    run_step tmfit 36000 $P scripts/fit_llava.py --backend hf-llava \
        --manifest $V/step3/manifest-half-a.jsonl --layers all --masks text,image,all \
        --dim-batch 1 --dtype float32 --allow-tf32 --target-mask text --checkpoint-every 2 \
        --out "$OUT" \
        --notes "target_mask=text confirmation refit (P4; x1_text lever), half_a 50 samples, fp32+TF32" \
        || echo "TMFIT_FAILED"
else
    echo "[tmfit] present - skip"
fi

if [ -f "$OUT/artifacts/provenance.json" ] && [ ! -f "$V/step4/lens_zoo_tmtext.json" ]; then
    run_step tmzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $V/step4/lens_zoo_tmtext.json --lens tmtext50=$OUT/artifacts \
        || echo "TMZOO_FAILED"
else
    echo "[tmzoo] present or fit missing - skip"
fi

echo "=== phase3c launcher done $(date -Is) ==="
