#!/bin/bash
# P31: the replication eval suite on the two new lenses (raw J-lens, the paper's convention):
# x20 lens evals (j-lens + vanilla logit lens), x21 verbal-report, x22 structure bands,
# x23 ignition + modality-split bands. Sequential, skip-if-present, failure-tolerant.
# Gates are per-model: qwen2vl (28 layers, d_model 3584) fits ~26GB slots now; llava-13b
# (d_model 5120) needs ~40GB. qwen2vl runs first so its steps are not blocked behind the
# scarcer 40GB slots.
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

for kind in qwen2vl llava-13b; do
    LENS=$V/fits/$( [ $kind = llava-13b ] && echo llava-1.5-13b-hf || echo qwen2-vl-7b-instruct )/merged/artifacts
    NEED=40000; [ $kind = qwen2vl ] && NEED=26000
    if [ ! -f "$OUT/x20_lens_evals_$kind.json" ]; then
        run_step x20-$kind $NEED $P $C/x20_lens_evals.py --model $kind --lens-dir $LENS \
            --variants j-lens,identity-uncal --json $OUT/x20_lens_evals_$kind.json || echo "X20_${kind}_FAILED"
    else echo "[x20-$kind] present - skip"; fi
    if [ ! -f "$OUT/x21_report_$kind.json" ]; then
        run_step x21-$kind $NEED $P $C/x21_verbal_report.py --model $kind --arm report --lens-dir $LENS \
            --json $OUT/x21_report_$kind.json || echo "X21_${kind}_FAILED"
    else echo "[x21-$kind] present - skip"; fi
    if [ ! -f "$OUT/x22_structure_$kind.json" ]; then
        run_step x22-$kind $NEED $P $C/x22_reasoning_structure.py --model $kind --arm structure --lens-dir $LENS \
            --json $OUT/x22_structure_$kind.json || echo "X22_${kind}_FAILED"
    else echo "[x22-$kind] present - skip"; fi
    if [ ! -f "$OUT/x23_ignition_$kind.json" ]; then
        run_step x23i-$kind $((NEED + 4000)) $P $C/x23_vlm_workspace.py --model $kind --arm ignition --lens-dir $LENS \
            --max-new-tokens 128 \
            --images-dir /data/baodq/coco2014/val2014 \
            --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
            --json $OUT/x23_ignition_$kind.json || echo "X23I_${kind}_FAILED"
    else echo "[x23i-$kind] present - skip"; fi
    if [ ! -f "$OUT/x23_band_$kind.json" ]; then
        run_step x23b-$kind $((NEED + 4000)) $P $C/x23_vlm_workspace.py --model $kind --arm band --lens-dir $LENS \
            --max-new-tokens 128 \
            --images-dir /data/baodq/coco2014/val2014 \
            --annotations /data/baodq/coco2014/annotations/instances_val2014.json \
            --json $OUT/x23_band_$kind.json || echo "X23B_${kind}_FAILED"
    else echo "[x23b-$kind] present - skip"; fi
done
echo "=== phase31 evals done $(date -Is) ==="
