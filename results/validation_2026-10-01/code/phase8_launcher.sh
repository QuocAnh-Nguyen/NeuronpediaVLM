#!/bin/bash
# P8: translator DATA-SCALING. The rank-32 KL translators measured only at n=40 (LQS 1.72); the
# user's ask: 1000 samples is the Sonnet-4.5 tuned-lens budget - test whether the translator
# class keeps improving with data. Fits at n=100/500/1000 on a fresh disjoint scaling manifest,
# then scores all rungs + the n=40 references on the 300-sample held-out. Translator fitting
# needs only cached activations (~2.4 s/sample forward), so ~3 h total, not 470 h of J-fitting.
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

MS=$V/manifest-scaling.jsonl; WU=$V/step5/W_U.pt
OUT=$V/step8
mkdir -p $OUT

if [ ! -f "$MS" ]; then
    echo "MISSING manifest-scaling.jsonl - run the corpus builder first"
fi

for N in 100 500 1000; do
    if [ ! -f "$OUT/n$N/artifacts/provenance.json" ] && [ -f "$MS" ] && [ -f "$WU" ]; then
        run_step lrfit$N 20000 $P $C/lowrank_translator.py --lens-dir $V/s2-merged/artifacts \
            --manifest $MS --mask text --limit $N --rank 32 --steps 800 --lr 1e-3 --seed 0 \
            --unembed $WU --translator-input jacobian --out-root $OUT/n$N \
            --backend hf-llava --dtype bfloat16 || echo "LRFIT${N}_FAILED"
    else
        echo "[lrfit$N] present or inputs missing - skip"
    fi
done

if [ -f "$OUT/n1000/artifacts/provenance.json" ] && [ ! -f "$OUT/lens_zoo_scaling.json" ]; then
    run_step lrzoo 20000 $P $C/lens_zoo_eval.py --heldout-manifest $V/step0/manifest-heldout.jsonl \
        --json $OUT/lens_zoo_scaling.json --mask text --tags text \
        --lens lrJ40=$V/step6-lowrank/artifacts --lens lrRaw40=$V/step6-raw/artifacts \
        --lens lrJ100=$OUT/n100/artifacts --lens lrJ500=$OUT/n500/artifacts \
        --lens lrJ1000=$OUT/n1000/artifacts \
        || echo "LRZOO_FAILED"
else
    echo "[lrzoo] present or fits missing - skip"
fi

echo "=== phase8 launcher done $(date -Is) ==="
