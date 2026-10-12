#!/bin/bash
# fit_supervisor.sh MODEL SHARD OUT LOG NEED_MIB
# Launch one fit shard with OOM-resilient GPU selection: poll until a GPU has NEED MiB
# free, then try candidates one at a time; a candidate is accepted only if the process
# survives its model-load window (EARLY_SECS) without an OOM marker; otherwise kill and
# try the next GPU; if all current candidates fail, re-poll (tenancy churns; MAXWAIT_HOURS
# bounds the total wait). Callers MUST stagger concurrent shards so two supervisors do not
# pick the same GPU in the same poll (see phase29 notes).
set -u
MODEL=$1; SHARD=$2; OUT=$3; LOG=$4; NEED=${5:-45000}
EARLY_SECS=${EARLY_SECS:-420}; MAXTRIES=${MAXTRIES:-6}; MAXWAIT_HOURS=${MAXWAIT_HOURS:-24}
R=/data/anhnq/NeuronpediaVLM
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

gpulist() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$NEED" '$2 >= N {print $1}' | tr '\n' ' '; }

deadline=$(( $(date +%s) + MAXWAIT_HOURS * 3600 ))
while :; do
    candidates=$(gpulist)
    if [ -z "$candidates" ]; then
        [ "$(date +%s)" -ge "$deadline" ] && { echo "[sup $SHARD] WAIT_TIMEOUT $(date -Is)"; exit 1; }
        echo "[sup $SHARD] no GPU with ${NEED}MiB free - waiting 300s $(date -Is)"
        sleep 300
        continue
    fi
    try=0
    for g in $candidates; do
        try=$((try + 1)); [ $try -gt $MAXTRIES ] && break
        echo "[sup $SHARD] try $try: GPU $g (need ${NEED}MiB) $(date -Is)"
        rm -f "$LOG"
        CORPUS_ARGS="--corpus coco --images-dir /data/baodq/coco2014/val2014 --n-images 100"
        case "$FIT_EXTRA" in *--manifest*) CORPUS_ARGS="";; esac
        CUDA_VISIBLE_DEVICES=$g setsid $P $R/scripts/fit_llava.py --model $MODEL --backend hf-llava \
            $CORPUS_ARGS --dtype bfloat16 \
            ${FIT_EXTRA:-} \
            --shard $SHARD --out "$OUT" > "$LOG" 2>&1 &
        pid=$!
        ok=1
        for i in $(seq 1 $EARLY_SECS); do
            sleep 1
            grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG" && { ok=0; break; }
            kill -0 $pid 2>/dev/null || { grep -qE "OutOfMemoryError|CUDA out of memory" "$LOG" && ok=0; break; }
        done
        if [ $ok -eq 1 ]; then
            echo "[sup $SHARD] GPU $g survived the load window - committing (pid $pid)"
            wait $pid; rc=$?
            echo "[sup $SHARD] finished rc=$rc $(date -Is)"
            exit $rc
        fi
        echo "[sup $SHARD] GPU $g OOM during load window - killing and trying the next GPU"
        pkill -P $pid 2>/dev/null; kill $pid 2>/dev/null; sleep 20
    done
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[sup $SHARD] WAIT_TIMEOUT $(date -Is)"; exit 1; }
    echo "[sup $SHARD] all candidates failed - re-polling in 300s $(date -Is)"
    sleep 300
done
