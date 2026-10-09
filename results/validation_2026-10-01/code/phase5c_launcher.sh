#!/bin/bash
# P5c: bias-only identity decomposition (R3 P5, zero fitting): the identity as a synthetic lens
# carrying the SAME moment-census bias+scale payload as the J-lens (the calibrated logit lens,
# z = unembed(s_l*(h+b_l))) scored against the J-lens with the same payload on the 300-sample
# held-out — decomposes the mid-layer fix into (a) the affine correction and (b) the actual
# transport. One zoo run (two lenses + the untrained logit-lens baseline).
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

OUT=$V/step5; IDENT=$OUT/ident
mkdir -p $OUT

if [ ! -f "$IDENT/artifacts/provenance.json" ]; then
    $P $C/ident_lens.py --src-lens-dir $V/s2-merged/artifacts --mask text --out-dir $IDENT \
        || echo "IDENT_FAILED"
else
    echo "[ident] present - skip"
fi

if [ -f "$IDENT/artifacts/provenance.json" ] && [ ! -f "$OUT/lens_zoo_ident.json" ]; then
    run_step identzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_ident.json --mask text --tags text --bias-dir $V/step4d \
        --lens ident=$IDENT/artifacts --lens merged=$V/s2-merged/artifacts \
        || echo "IDENTZOO_FAILED"
else
    echo "[identzoo] present or ident missing - skip"
fi

echo "=== phase5c launcher done $(date -Is) ==="
