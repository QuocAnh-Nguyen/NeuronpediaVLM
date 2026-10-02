#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Campaign supervisor with GPU free-memory guards (replaces run_rest.sh, whose chain died
# at 2026-10-01T~04:50 when S1's fit OOMed under a growing co-tenant - D12).
#
# Every step waits for a free-memory window (polled every 30 s, since free VRAM on this
# shared box is first-come-first-served). Failures we do not control (CUDA OOM, a SIGKILL
# from a co-tenant reclaiming the GPU - observed at 07:16 S1 and 07:28 X6 - or a stray
# CUDA error) are retried with no fixed attempt cap (MAX_ROUNDS); only 3 consecutive
# non-transient failures stop the chain, so a real bug still fails fast and identically.
# Fits are resumable via fit_masked's
# checkpoint; note the checkpoint's settings fingerprint includes dim_batch, so a step must
# always use the same dim_batch across retries:
#   * s1: dim_batch 8 to resume the existing 20-sample checkpoint from the old chain's fit
#     (run_step2.sh's own default is 8; it resumed nothing at 4 and hard-failed, D13)
#   * s2: dim_batch 4, fresh fits (halves activation memory; math is identical).
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
LOGS=$REPO/results/validation_2026-10-01/logs
export PYTHONPATH=$REPO/src
export HF_HUB_OFFLINE=1
export DIMBATCH=${DIMBATCH:-4}
mkdir -p "$LOGS" "$RUN/step2" "$RUN/step3" "$RUN/step4"
cd "$REPO" || exit 1

free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }

need_free() {  # $1 = MiB threshold, $2 = label; waits up to 16 h (the 2026-10-02 co-tenant
              #           held 72 GiB for >2 h at a stretch; a 6 h cap let the chain exit)
    local need=$1 label=$2 waited=0 free
    while :; do
        free=$(free_mib)
        if [ "${free:-0}" -ge "$need" ]; then
            echo "[$label] free=${free}MiB >= ${need}MiB - starting $(date -Is)"
            return 0
        fi
        if [ "$waited" -ge 57600 ]; then
            echo "[$label] GAVE UP after 16 h: free=${free}MiB < ${need}MiB"
            return 1
        fi
        sleep 30
        waited=$((waited + 30))
    done
}

run_guarded() {  # $1 = MiB threshold, $2 = label, rest = command
    local need=$1 label=$2
    shift 2
    local attempt=0 rc reason other_failures=0
    local max_rounds=${MAX_ROUNDS:-60}
    while [ "$attempt" -lt "$max_rounds" ]; do
        attempt=$((attempt + 1))
        need_free "$need" "$label" || return 1
        "$@" >"$LOGS/${label}_attempt${attempt}.log" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            echo "[$label] ok (attempt $attempt) $(date -Is)"
            return 0
        fi
        if grep -qa 'OutOfMemory' "$LOGS/${label}_attempt${attempt}.log"; then
            reason=OOM
        elif grep -qa 'CUDA error' "$LOGS/${label}_attempt${attempt}.log"; then
            reason=cuda-error
        elif grep -qa 'Killed' "$LOGS/${label}_attempt${attempt}.log"; then
            reason=sigkill
        else
            reason=other
        fi
        if [ "$reason" = other ]; then
            other_failures=$((other_failures + 1))
        else
            other_failures=0
        fi
        echo "[$label] attempt $attempt failed rc=$rc reason=$reason other=$other_failures/3 (next in 60 s)"
        if [ "$other_failures" -ge 3 ]; then
            echo "[$label] giving up: 3 consecutive non-transient failures"
            return 1
        fi
        sleep 60
    done
    echo "[$label] giving up after $max_rounds rounds"
    return 1
}

echo "=== run_campaign started $(date -Is) free=$(free_mib)MiB dim_batch_default=$DIMBATCH ==="

# S1: WikiText control fit + held-out scoring/gate. dim_batch 8 (resumes the 20-sample
# checkpoint left by the old chain; bf16 weights ~16 GiB, ~20 GiB peak) - the guard is 24 GiB
# so a modest window suffices (the 07:44Z attempt ran at 32.9 GiB free).
run_guarded 24000 s1 env DIMBATCH=8 bash "$CODE/run_step2.sh" || { echo CAMPAIGN_S1_FAILED; exit 1; }

# S2: two 50-sample caption half-fits + merge + held-out evaluation. fp32+TF32 weights are
# ~29 GiB (X6 verdict use_fp32) and dim_batch only scales activations (identical math, D13),
# so dim_batch 1 drops the peak to ~31 GiB: the guard is 36 GiB, not 55, which is what the
# 2026-10-02 co-tenant (57-73 GiB held for hours) makes decisive.
run_guarded 36000 s2 env DIMBATCH=1 bash "$CODE/run_step3.sh" || { echo CAMPAIGN_S2_FAILED; exit 1; }

# X1: 20-image shard pair, 6 recorded layers each, fp32+TF32 like S2, dim_batch 1 (~31 GiB).
run_guarded 36000 x1 env DIMBATCH=1 bash "$CODE/run_step4.sh" || { echo CAMPAIGN_X1_FAILED; exit 1; }

# X3 re-census with L24 (X7/X9's upper edit layer): forwards only, so it fits a narrower window
# than any fit. The L0/L16/L31 rows are recomputed identically (same manifest, same 50
# samples), so earlier citations stay valid; the new L24 row gives the X9 alpha grid its
# measured units instead of the auto-fallback.
run_guarded 22000 x3 "$P" "$CODE/x3_census.py" --manifest "$RUN/step0/manifest-fit.jsonl" \
    --n-samples 50 --out "$RUN/step1/x3_norms.json" \
    || { echo CAMPAIGN_X3_RECENSUS_FAILED; exit 1; }

# X7 conditioning + X9 edit sweep against the merged caption lens.
run_guarded 22000 x7x9 "$P" "$CODE/x7_x9_interventions.py" --lens-dir "$RUN/s2-merged/artifacts" \
    --manifest "$RUN/step0/manifest-heldout.jsonl" --n-samples 10 \
    --norms-json "$RUN/step1/x3_norms.json" --json "$RUN/step4/x7_x9.json" \
    || { echo CAMPAIGN_X7X9_FAILED; exit 1; }

echo "=== campaign done $(date -Is) ==="
