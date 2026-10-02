#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# run_fd.sh - S1's finite-difference row (ii), in-chain and GPU-only (D22).
#
# Replaces the decoupled /tmp/s1_fd_watch.sh (D20), whose 3 h CPU fallback violates the
# 2026-10-02 directive that results are computed on GPU only. The chain runs this as the
# optional `fd` step under a 36 GiB guard (the FD model is the fp32 estimator); the
# campaign tolerates its failure. Re-runs the s1 scoring (the `--skip-fd` invocation of
# run_step2.sh plus the FD), publishes step2/s1_score.json by rename only when the
# check_ii_finite_difference row is present, so no reader ever sees a partial file.
set -u
R=$HOME/ai4life/phuongnh/vlm-lens
OUT=/data/vlm-lens/validation/step2/s1_score
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$R/src HF_HUB_OFFLINE=1
rm -f "$OUT.new.json"
"$P" "$R/results/validation_2026-10-01/code/s1_score.py" \
    --lens-dir /data/vlm-lens/validation/s1-text/artifacts --mask all \
    --heldout-manifest /data/vlm-lens/validation/step0/manifest-text-heldout.jsonl \
    --fit-manifest /data/vlm-lens/validation/step0/manifest-text-fit.jsonl \
    --json "$OUT.new.json" || { echo "FD_RUN_FAILED rc=$?"; exit 1; }
if grep -q check_ii_finite_difference "$OUT.new.json"; then
    mv -f "$OUT.new.json" "$OUT.json"
    echo "fd published $(date -Is)"
    exit 0
fi
echo "FD_UNPUBLISHED (no check_ii_finite_difference row)"
exit 1
