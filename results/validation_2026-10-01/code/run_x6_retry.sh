#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Opportunistic X6 fp32 leg.
#
# The first TF32 attempt (03:14) died with CUDA OOM: the co-tenant on this box held
# ~57-61 GiB at the time, so the leg could not even load fp32 weights (~30 GiB). The
# campaign's own fits (S1/S2/...) also leave no room for a co-running fp32 model.
#
# So: wait for every campaign launcher to exit, then retry the leg - memory-guarded - in
# a bounded loop, and run the comparator as soon as it produces artifacts. If the
# co-tenant never frees enough memory, log the skip reason; the report records the leg as
# environmentally infeasible with this evidence.
set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
STAMP=$(date -Is)
RESULTS="${RESULTS:-$HOME/ai4life/phuongnh/vlm-lens/results/validation_2026-10-01}"
LOGS=$RESULTS/logs
CODE=$RESULTS/code
PY=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
RUN=/data/vlm-lens/validation
BF16=$RUN/x6-bf16/artifacts
FP32=$RUN/x6-fp32/artifacts
echo "=== x6_retry started $STAMP (waiting for campaign launchers to finish) ==="
for _ in $(seq 1 720); do
    if ! pgrep -f 'run_res[t].sh|run_ste[p]2.sh|run_ste[p]3.sh|run_ste[p]4.sh|run_x7x[9].sh|run_campaign[.]sh|fit_llav[a].py' >/dev/null; then
        break
    fi
    sleep 60
done
echo "=== campaign idle, starting memory-guarded retries $(date -Is) ==="

# The box is shared and neighbours have SIGKILLed our fits (07:16 S1, 07:28 this leg), so a
# failed attempt is only fatal when it looks deterministic: three consecutive failures that
# are none of memory-window skip (rc 3), OOM, CUDA error, or SIGKILL. Real attempts are
# bounded so the script cannot spin forever; waits are short because windows appear and
# vanish within minutes on this box.
other_failures=0
for i in $(seq 1 60); do
    bash "$CODE/run_x6_fp32.sh" >"$LOGS/x6_fp32_retry_$i.log" 2>&1
    rc=$?
    log="$LOGS/x6_fp32_retry_$i.log"
    if [ "$rc" -eq 0 ]; then
        echo "=== x6 fp32 leg OK on attempt $i $(date -Is) ==="
        if [ -d "$BF16" ]; then
            echo "bf16 artifacts present at $BF16 (run_x6_fp32.sh already ran x6_compare)"
        else
            echo "WARNING: bf16 artifacts missing at $BF16 - comparator launched with stale dir"
        fi
        exit 0
    fi
    if [ "$rc" -eq 3 ]; then
        echo "retry $i: no window ($(grep -a -o 'free GPU memory [0-9]* MiB' "$log" | head -1)); next in 5 min"
        sleep 300
        continue
    fi
    if grep -qa 'OutOfMemory' "$log"; then
        reason=OOM
    elif grep -qa 'CUDA error' "$log"; then
        reason=cuda-error
    elif grep -qa 'Killed' "$log"; then
        reason=sigkill
    else
        reason=other
    fi
    echo "retry $i: leg failed rc=$rc reason=$reason (new window in 5 min)"
    if [ "$reason" = other ]; then
        other_failures=$((other_failures + 1))
    else
        other_failures=0
    fi
    if [ "$other_failures" -ge 3 ]; then
        echo "=== x6 fp32 leg giving up: 3 consecutive non-transient failures ==="
        exit 1
    fi
    sleep 300
done
echo "=== x6 fp32 leg abandoned after 60 rounds $(date -Is) ==="
exit 2
