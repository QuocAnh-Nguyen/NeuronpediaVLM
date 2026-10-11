#!/bin/bash
# P29: fit the llava-1.5-13b J-lens (100 COCO captions, masks text/image/all) in two shards
# run on DIFFERENT GPUs (shard 1 launches after shard 0's model load, so the free-memory
# guard naturally picks another GPU), then merge. Skip-if-present per step.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
P=/data/anhnq/envs/vlm_truth_py313/bin/python
F=$V/fits/llava-1.5-13b-hf
LOGD=/home/nvidia/vlm-logs; mkdir -p $LOGD $F
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH=$R/src:$R/third_party/jacobian-lens PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
pick_gpu() { nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F', ' -v N="$1" '$2 >= N {print $2, $1}' | sort -rn | head -1 | awk '{print $2}'; }
wait_gpu() { local need=$1 label=$2 waited=0 g; while :; do g=$(pick_gpu "$need"); if [ -n "$g" ]; then echo "[$label] GPU $g >= ${need}MiB $(date -Is)" >&2; echo "$g"; return 0; fi; if [ "$waited" -ge 86400 ]; then echo "[$label] NO_GPU after 24h $(date -Is)" >&2; return 1; fi; sleep 60; waited=$((waited + 60)); done; }
FIT_ARGS="--model llava-13b --backend hf-llava --corpus coco --images-dir /data/baodq/coco2014/val2014 --n-images 100 --dtype bfloat16"

if [ ! -f "$F/s0/lens-text.pt" ]; then
    g=$(wait_gpu 55000 shard0) && CUDA_VISIBLE_DEVICES=$g nohup $P $R/scripts/fit_llava.py $FIT_ARGS --shard 0/2 --out $F/s0 > $LOGD/fit13b_s0.log 2>&1 &
else echo "[13b s0] present - skip"; fi
sleep 240   # let shard 0 finish its model load + allocation before shard 1 picks a GPU
if [ ! -f "$F/s1/lens-text.pt" ]; then
    g=$(wait_gpu 55000 shard1) && CUDA_VISIBLE_DEVICES=$g nohup $P $R/scripts/fit_llava.py $FIT_ARGS --shard 1/2 --out $F/s1 > $LOGD/fit13b_s1.log 2>&1 &
else echo "[13b s1] present - skip"; fi
wait
if [ ! -f "$F/merged/lens-text.pt" ]; then
    $P $R/scripts/fit_llava.py --model llava-13b --merge $F/s0 $F/s1 --out $F/merged && echo "FIT13B_MERGED" || echo "FIT13B_MERGE_FAILED"
else echo "[13b merged] present - skip"; fi
echo "=== phase29 fit13b done $(date -Is) ==="
