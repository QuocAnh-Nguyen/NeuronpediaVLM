#!/bin/bash
# P26 (x23): the VLM-only workspace arms - cross-modal verbal-report swap (TEXT/IMAGE/ALL
# scopes), modulation-during-captioning with concept-vs-visual co-occupancy, image-mixing
# ignition alpha sweeps, and modality-split workspace-band statistics.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
S=$R/results/validation_2026-10-01/code
C=/home/nvidia/vlm-run
P=/data/anhnq/envs/vlm_truth_py313/bin/python
OUT=/home/nvidia/vlm-lens-out/validation/step12
LOGD=/home/nvidia/vlm-logs
mkdir -p $OUT $LOGD
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens:$S PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }

for arm in report modulation ignition band; do
    if [ ! -f "$OUT/x23_vlm_workspace_$arm.json" ]; then
        run_step x23-$arm 20000 $P $C/x23_vlm_workspace.py --arm $arm \
            --lens-dir $V/s2-merged/artifacts --bias-dir $V/step5e \
            --images-dir /data/baodq/coco2014/val2014 \
            --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
            --json $OUT/x23_vlm_workspace_$arm.json || echo "X23_${arm}_FAILED"
    else
        echo "[x23-$arm] present - skip"
    fi
done
echo "=== phase26 launcher done $(date -Is) ==="
