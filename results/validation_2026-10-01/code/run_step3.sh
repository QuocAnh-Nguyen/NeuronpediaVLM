#!/bin/bash
# Step 3 chain: S2 caption fits (question halves A and B, 50 samples each), merge, held-out
# evaluation with tags/placeholder modes, A4/X8 half cross-scoring and A4/E6 cross-corpus.
set -u
WORK=${VLM_WORK:-/data/anhnq}
REPO=${REPO:-$WORK/NeuronpediaVLM}
P=${P:-$WORK/envs/vlm_truth_py313/bin/python}
RUN=${RUN:-$WORK/vlm-lens-out/validation}
MOUNT=${MOUNT:-$RUN}
CODE=$REPO/results/validation_2026-10-01/code
LOGS=$REPO/results/validation_2026-10-01/logs
DIMBATCH=${DIMBATCH:-1}
export PYTHONPATH=$REPO/src PYTHONUNBUFFERED=1
export HF_HOME=${HF_HOME:-$WORK/hf_cache}
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN/step3" "$MOUNT" "$LOGS"
cd "$REPO" || exit 1

echo "=== split halves $(date -Is) ==="
"$P" "$CODE/split_halves.py" --manifest "$RUN/step0/manifest-fit.jsonl" \
    --split-json "$RUN/step0/corpus-split.json" --out-dir "$RUN/step3" || { echo SPLIT_FAILED; exit 1; }

# Dtype: fp32 weights + TF32 matmuls (D10/D18). The X6 verdict (step1/x6_dtype.json) is
# use_fp32: true - per-layer medians are 1.2-1.5 % but L0's max is 49 % - and the fp32+TF32
# leg cost 2486.5 s for 8 samples against 2205.8 s for bf16 (1.13x), so the pre-registered
# rule is affordable on the production caption lens.
# Two 50-sample halves. Each half picks its own GPU as soon as a card with >= 36 GiB free
# appears (lock-guarded so simultaneous picks land on distinct cards; with a single free
# card they run sequentially). The Brev box is shared with long-running co-tenant jobs and
# the fit's fp32 weights are ~29 GiB (~31 GiB peak at dim_batch 1), so waiting for a window
# is the norm; checkpoints make interruptions cheap. A watchdog kills the fits if no
# checkpoint lands for 190 min (a genuine hang, not slowness).
GPULOCKS=$LOGS/gpulocks
rm -rf "$GPULOCKS"; mkdir -p "$GPULOCKS"

pick_gpu() {  # $1 = MiB needed, $2 = half tag: waits (16 h cap) for an unlocked card and locks it
    local need=$1 tag=$2 waited=0 i free
    while :; do
        for i in $(nvidia-smi --query-gpu=index --format=csv,noheader); do
            [ -d "$GPULOCKS/g$i" ] && continue
            free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$i")
            if [ "${free:-0}" -ge "$need" ]; then
                mkdir "$GPULOCKS/g$i" 2>/dev/null || continue
                echo "$i"; return 0
            fi
        done
        if [ "$waited" -ge 57600 ]; then
            echo "[half $tag] no GPU with ${need}MiB free after 16 h" >&2
            return 1
        fi
        sleep 30; waited=$((waited + 30))
        if [ $((waited % 1800)) -eq 0 ]; then
            echo "[half $tag] waiting (30 min): $(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | tr '\n' ' ')" >&2
        fi
    done
}

run_half() {  # $1 = half: pick GPU, fit, release the lock (EXIT trap covers killed children)
    local half=$1 gpu rc
    gpu=$(pick_gpu "${GPU_NEED_MIB:-36000}" "$half") || return 1
    echo "[half $half] picked GPU $gpu $(date -Is)"
    trap "rmdir $GPULOCKS/g$gpu 2>/dev/null" EXIT
    CUDA_VISIBLE_DEVICES=$gpu "$P" scripts/fit_llava.py --backend hf-llava \
        --manifest "$RUN/step3/manifest-half-$half.jsonl" \
        --layers all --masks text,image,all --dim-batch "$DIMBATCH" --dtype float32 --allow-tf32 \
        --checkpoint-every 2 --out "$MOUNT/s2-half-$half" \
        --notes "S2 caption pilot, question half $half, 50 samples, skip_first=1, fp32+TF32 (X6)" \
        > "$LOGS/s2_half_${half}.log" 2>&1
    rc=$?
    rmdir "$GPULOCKS/g$gpu" 2>/dev/null
    return $rc
}

(
    sleep 11400
    while :; do
        if [ -z "$(find "$MOUNT" -name 'checkpoint.pt' -mmin -190 2>/dev/null | head -1)" ]; then
            echo "S2_WATCHDOG: no checkpoint.pt landed in 190 min - killing fits $(date -Is)"
            pkill -f 'fit_llava.py --backend hf-llava'
            exit 0
        fi
        sleep 600
    done
) & WD=$!

pids=()
run_half a & pids+=($!)
run_half b & pids+=($!)
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
kill "$WD" 2>/dev/null

if [ "$rc" -ne 0 ]; then
    echo "S2_HALF_FIT_FAILED (see $LOGS/s2_half_{a,b}.log)"
    exit 1
fi

echo "=== merge halves $(date -Is) ==="
"$P" scripts/fit_llava.py --merge "$MOUNT/s2-half-a" "$MOUNT/s2-half-b" \
    --out "$MOUNT/s2-merged" --notes "S2 merged 100-sample caption lens (halves A+B)" \
    || { echo MERGE_FAILED; exit 1; }

echo "=== s2_eval $(date -Is) ==="
"$P" "$CODE/s2_eval.py" --main-lens-dir "$MOUNT/s2-merged/artifacts" \
    --heldout-manifest "$RUN/step0/manifest-heldout.jsonl" \
    --half-lens-a "$MOUNT/s2-half-a/artifacts" --half-lens-b "$MOUNT/s2-half-b/artifacts" \
    --split-json "$RUN/step0/corpus-split.json" \
    --text-lens-dir "$RUN/s1-text/artifacts" --text-heldout "$RUN/step0/manifest-text-heldout.jsonl" \
    --json "$RUN/step3/s2_eval.json" || { echo S2_EVAL_FAILED; exit 1; }

echo "=== step3 chain done $(date -Is) ==="
