#!/bin/bash
# P34: the 64-token-baseline eval arms for the budget comparison (D48), plus the 128-token
# generation budget on the x23 caption arms (the user's ask: 64 is too low to verbalize all
# image features). The 256-token lens evals are phase31's job; this phase covers the 64
# baselines: 7b -> the production s2-merged lens; 13b/qwen -> fits/<sub>/merged-64tok.
# Gates sized per d_model (7b/qwen ~26GB; 13b d_model 5120 needs ~40GB, so it waits with
# everything else). qwen/7b run first to use the scarcer big slots efficiently.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
S=$R/results/validation_2026-10-01/code
C=/home/nvidia/vlm-run
P=/data/anhnq/envs/vlm_truth_py313/bin/python
OUT=$V/step13; mkdir -p $OUT
LOGD=/home/nvidia/vlm-logs
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens:$S PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
run_step() { local label=$1 need=$2 g; shift 2; g=$(wait_gpu "$need" "$label") || return 1; CUDA_VISIBLE_DEVICES=$g "$@"; }

for kind in qwen2vl llava-7b llava-13b; do
    case $kind in
        llava-7b)  LENS=$V/s2-merged/artifacts; NEED=26000;;
        qwen2vl)   LENS=$V/fits/qwen2-vl-7b-instruct/merged-64tok/artifacts; NEED=26000;;
        llava-13b) LENS=$V/fits/llava-1.5-13b-hf/merged-64tok/artifacts; NEED=40000;;
    esac
    TAG="64tok"
    if [ ! -f "$OUT/x20_lens_evals_${kind}_${TAG}.json" ]; then
        run_step x20-$kind-$TAG $NEED $P $C/x20_lens_evals.py --model $kind --lens-dir $LENS \
            --variants j-lens,identity-uncal --json $OUT/x20_lens_evals_${kind}_${TAG}.json || echo "X20_${kind}_${TAG}_FAILED"
    else echo "[x20-$kind-$TAG] present - skip"; fi
    if [ ! -f "$OUT/x21_report_${kind}_${TAG}.json" ]; then
        run_step x21-$kind-$TAG $NEED $P $C/x21_verbal_report.py --model $kind --arm report --lens-dir $LENS \
            --json $OUT/x21_report_${kind}_${TAG}.json || echo "X21_${kind}_${TAG}_FAILED"
    else echo "[x21-$kind-$TAG] present - skip"; fi
    if [ ! -f "$OUT/x22_structure_${kind}_${TAG}.json" ]; then
        run_step x22-$kind-$TAG $NEED $P $C/x22_reasoning_structure.py --model $kind --arm structure --lens-dir $LENS \
            --json $OUT/x22_structure_${kind}_${TAG}.json || echo "X22_${kind}_${TAG}_FAILED"
    else echo "[x22-$kind-$TAG] present - skip"; fi
    if [ ! -f "$OUT/x23_ignition_${kind}_${TAG}.json" ]; then
        run_step x23i-$kind-$TAG $((NEED + 4000)) $P $C/x23_vlm_workspace.py --model $kind --arm ignition --lens-dir $LENS \
            --max-new-tokens 128 \
            --images-dir /data/baodq/coco2014/val2014 \
            --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
            --json $OUT/x23_ignition_${kind}_${TAG}.json || echo "X23I_${kind}_${TAG}_FAILED"
    else echo "[x23i-$kind-$TAG] present - skip"; fi
    if [ ! -f "$OUT/x23_band_${kind}_${TAG}.json" ]; then
        run_step x23b-$kind-$TAG $((NEED + 4000)) $P $C/x23_vlm_workspace.py --model $kind --arm band --lens-dir $LENS \
            --max-new-tokens 128 \
            --images-dir /data/baodq/coco2014/val2014 \
            --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
            --json $OUT/x23_band_${kind}_${TAG}.json || echo "X23B_${kind}_${TAG}_FAILED"
    else echo "[x23b-$kind-$TAG] present - skip"; fi
done
echo "=== phase34 evals64 done $(date -Is) ==="
