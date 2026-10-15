#!/bin/bash
# Waits for a >=22 GiB GPU window, then runs the corrected X1 compare with
# expandable_segments; up to 3 attempts, 4 h of waiting each.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for attempt in 1 2 3; do
    G=""
    for i in $(seq 1 240); do
        G=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' '$2>=22000 {print $2, $1}' | sort -rn | head -1 | awk '{print $2}')
        [ -n "$G" ] && break
        sleep 60
    done
    if [ -z "$G" ]; then echo "attempt $attempt: NO_GPU after 4h $(date -Is)"; continue; fi
    echo "attempt $attempt launching on GPU $G $(date -Is)"
    cd $R
    CUDA_VISIBLE_DEVICES=$G $P $C/x1_compare.py --dir-all $V/x1-all/artifacts --dir-text $V/x1-text/artifacts --manifest $V/step3/manifest-half-a.jsonl --json $V/step4/x1_targetmask_v2.json
    rc=$?
    echo "attempt $attempt rc=$rc $(date -Is)"
    [ "$rc" -eq 0 ] && exit 0
    sleep 120
done
echo "ALL_ATTEMPTS_FAILED $(date -Is)"
exit 1
