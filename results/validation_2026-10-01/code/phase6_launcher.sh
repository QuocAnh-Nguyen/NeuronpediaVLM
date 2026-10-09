#!/bin/bash
# P6: tuned-lens-style low-rank translators (the model-class ladder's next rung above the
# diagonal census bias+scale): fit per-layer A_l (rank 32) + bias by KL distillation to the
# model's own final logits on the 40-sample fit split, with translator input x_l = J_l h_l
# (our class extended) and x_l = h_l (the pure tuned lens, the literature's ceiling) — then
# score both composed lenses on the 300-sample held-out. J_l FIXED; no new Jacobian fits.
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

OUT=$V/step6; WU=$V/step5/W_U.pt
mkdir -p $OUT

if [ -f "$WU" ]; then
    if [ ! -f "$V/step6-lowrank/artifacts/provenance.json" ]; then
        run_step lrfit 20000 $P $C/lowrank_translator.py --lens-dir $V/s2-merged/artifacts \
            --manifest $V/step3/manifest-half-a.jsonl --mask text --limit 40 --rank 32 \
            --steps 800 --lr 1e-3 --seed 0 --unembed $WU --translator-input jacobian \
            --out-root $V/step6-lowrank --backend hf-llava --dtype bfloat16 || echo "LRFIT_FAILED"
    else
        echo "[lrfit] present - skip"
    fi
    if [ ! -f "$V/step6-raw/artifacts/provenance.json" ]; then
        run_step rawfit 20000 $P $C/lowrank_translator.py --lens-dir $V/s2-merged/artifacts \
            --manifest $V/step3/manifest-half-a.jsonl --mask text --limit 40 --rank 32 \
            --steps 800 --lr 1e-3 --seed 0 --unembed $WU --translator-input raw \
            --out-root $V/step6-raw --backend hf-llava --dtype bfloat16 || echo "RAWFIT_FAILED"
    else
        echo "[rawfit] present - skip"
    fi
    if [ -f "$V/step6-lowrank/artifacts/provenance.json" ] && [ -f "$V/step6-raw/artifacts/provenance.json" ] \
        && [ ! -f "$OUT/lens_zoo_lowrank.json" ]; then
        run_step lrzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
            --json $OUT/lens_zoo_lowrank.json --mask text --tags text \
            --lens lowrankJ=$V/step6-lowrank/artifacts --lens lowrankRaw=$V/step6-raw/artifacts \
            || echo "LRZOO_FAILED"
    else
        echo "[lrzoo] present or fits missing - skip"
    fi
else
    echo "W_U missing - run phase5f first"
fi

echo "=== phase6 launcher done $(date -Is) ==="
