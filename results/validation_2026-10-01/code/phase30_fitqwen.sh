#!/bin/bash
# P30: fit the Qwen2-VL-7B-Instruct J-lens through the new Qwen2VLLensModel wrapper (same
# corpus/masks/shard pattern as P29; variable-resolution image tokens are handled by the
# wrapper's encode_mm). Skip-if-present per step.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
P=/data/anhnq/envs/vlm_truth_py313/bin/python
F=$V/fits/qwen2-vl-7b-instruct
LOGD=/home/nvidia/vlm-logs; mkdir -p $LOGD $F
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
FIT_ARGS="--model qwen2vl --backend hf-llava --corpus coco --images-dir /data/baodq/coco2014/val2014 --n-images 100 --dtype bfloat16"

if [ ! -f "$F/s0/artifacts/lens-text.pt" ]; then
    g=$(wait_gpu 45000 shard0) && CUDA_VISIBLE_DEVICES=$g nohup $P $R/scripts/fit_llava.py $FIT_ARGS --shard 0/2 --out $F/s0 > $LOGD/fitqwen_s0.log 2>&1 &
else echo "[qwen s0] present - skip"; fi
sleep 240
if [ ! -f "$F/s1/artifacts/lens-text.pt" ]; then
    g=$(wait_gpu 45000 shard1) && CUDA_VISIBLE_DEVICES=$g nohup $P $R/scripts/fit_llava.py $FIT_ARGS --shard 1/2 --out $F/s1 > $LOGD/fitqwen_s1.log 2>&1 &
else echo "[qwen s1] present - skip"; fi
wait
if [ ! -f "$F/merged/artifacts/lens-text.pt" ]; then
    $P $R/scripts/fit_llava.py --model qwen2vl --merge $F/s0 $F/s1 --out $F/merged && echo "FITQWEN_MERGED" || echo "FITQWEN_MERGE_FAILED"
else echo "[qwen merged] present - skip"; fi
echo "=== phase30 fitqwen done $(date -Is) ==="
