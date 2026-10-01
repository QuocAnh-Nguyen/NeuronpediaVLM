#!/bin/bash
# Supervisor for the remaining campaign steps, run only after run_step1.sh (gate + X6)
# has finished: S1 control fit + scoring -> S2 caption pilot + evaluation -> X1 -> X7/X9.
# Stops at the first failing step so the logs stay diagnosable; every step is resumable.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO" || exit 1

# The step1 chain (equivalence gate + X6 bf16/fp32) owns the GPU until it exits; wait for
# it here so this supervisor can be launched immediately without contending for memory.
echo "=== run_rest waiting for step1 chain $(date -Is) ==="
while pgrep -f "run_step1[.]sh" > /dev/null; do
    sleep 60
done
echo "=== step1 chain finished $(date -Is) ==="

# X6's fp32 leg first: torch defaults allow_tf32=False and a true-fp32 fit on H100 is ~15x
# slower than bf16 (measured ~10 min/sample, ~9 h for 8 samples). Re-run it with TF32 so the
# comparison fit fits the budget; a failure here does not block the rest of the campaign.
echo "=== X6 fp32+TF32 leg $(date -Is) ==="
bash "$CODE/run_x6_fp32.sh" || echo X6_FP32_LEG_FAILED_CONTINUING

echo "=== step2 (S1 text control) $(date -Is) ==="
bash "$CODE/run_step2.sh" || { echo STEP2_FAILED; exit 1; }

echo "=== step3 (S2 caption pilot) $(date -Is) ==="
bash "$CODE/run_step3.sh" || { echo STEP3_FAILED; exit 1; }

echo "=== step4 (X1 target-mask) $(date -Is) ==="
bash "$CODE/run_step4.sh" || { echo STEP4_FAILED; exit 1; }

echo "=== X7/X9 interventions $(date -Is) ==="
"$P" "$CODE/x7_x9_interventions.py" --lens-dir "$RUN/s2-merged/artifacts" \
    --manifest "$RUN/step0/manifest-heldout.jsonl" --n-samples 10 \
    --norms-json "$RUN/step1/x3_norms.json" --json "$RUN/step4/x7_x9.json" \
    || { echo X7X9_FAILED; exit 1; }

echo "=== campaign done $(date -Is) ==="
