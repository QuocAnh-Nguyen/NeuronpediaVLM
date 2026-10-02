#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# gpu_guard.sh - speed-aware campaign guard (D22).
#
# The box is shared, and the co-tenant has twice saturated the SMs *while leaving memory
# free*, so run_campaign's free-memory gates (D12/D21) are necessary but not sufficient.
# Measured 2026-10-02T17:05Z (three ~10 GiB co-tenant jobs, util pinned at 100%): TF32
# 20x4096^3 matmuls ran at 41 TFLOPS (~8% of the H100 PCIe's peak), and S2 produced <5
# samples in 2.05 h (>5x the X6-era 311 s/sample). At that derate every wall-minute of a
# fit burns ~10 minutes of the GPU-work budget.
#
# Policy (user directive 2026-10-02: results run on GPU only; CPU is for quick checks):
#   * No campaign running: probe the box. Start the campaign only when it probes fast
#     (>= TF_MIN TFLOPS TF32) and has >= FREE_MIN MiB free; run_campaign's per-step
#     need_free guards take it from there for each step's real appetite.
#   * Campaign running with a work step: watch the newest *.pt under $RUN. If nothing new
#     lands within TRIP_S seconds, the window derated underneath the step: kill the whole
#     chain (checkpoints make restarts cheap) and go back to probing.
#   * Probe only in the no-step state: a running step legitimately loads the SMs, so a
#     mid-run probe cannot distinguish "our work" from "co-tenant".
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
P=$HOME/miniconda3/envs/vlm_truth_py313/bin/python
CODE=$REPO/results/validation_2026-10-01/code
LOGS=$REPO/results/validation_2026-10-01/logs
RUN=/data/vlm-lens/validation
TF_MIN=150          # TFLOPS TF32; ~3.7x the 2026-10-02T17:05Z derated reading of 41
FREE_MIN=24000      # MiB; the campaign's per-step guards (24-36 GiB) take it from there
TRIP_S=5400         # 90 min without a new checkpoint while a step runs -> derated window
SLEEP=180

CAMPAIGN_RE='^bash .*/run_campaign\.sh$'
WORK_RE='^/home/.*/python .*(fit_llava|s1_score|s2_eval|x3_census|x7_x9_interventions)\.py'
STEP_RE='^bash .*/run_step[34]\.sh$'

log() { echo "[guard] $(date -Is) $*"; }

probe_tf() {
    $P - <<'PY' 2>/dev/null
import time, torch
torch.set_float32_matmul_precision('high')
c = torch.randn(4096, 4096, device='cuda'); d = torch.randn(4096, 4096, device='cuda')
for _ in range(3): c @ d
torch.cuda.synchronize(); t = time.time()
for _ in range(20): c @ d
torch.cuda.synchronize(); dt = time.time() - t
print(int(20 * 2 * 4096 ** 3 / dt / 1e12))
PY
}

log "started pid=$$ tf_min=${TF_MIN} free_min=${FREE_MIN} trip_s=${TRIP_S}"
work_seen_at=0; last_seen=0
while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    free=${free:-0}
    if pgrep -f "$CAMPAIGN_RE" >/dev/null 2>&1; then
        if pgrep -f "$WORK_RE" >/dev/null 2>&1; then
            newest=$(find "$RUN" -name '*.pt' -printf '%T@\n' 2>/dev/null | sort -n | tail -1 | cut -d. -f1)
            case "${newest:-}" in ''|*[!0-9]*) newest=0 ;; esac
            now=$(date +%s)
            if [ "$work_seen_at" -eq 0 ]; then
                work_seen_at=$now; last_seen=$newest
                log "work step running; output clock starts (newest_pt_age=$((now - newest))s free=${free}MiB)"
            elif [ "$newest" -gt "$last_seen" ]; then
                last_seen=$newest; work_seen_at=$now
                log "new output file seen; clock reset"
            elif [ $((now - work_seen_at)) -ge "$TRIP_S" ]; then
                log "TRIP: no new output in $((now - work_seen_at))s (free=${free}MiB) - killing chain for a fast window"
                pkill -f "$CAMPAIGN_RE" 2>/dev/null; pkill -f "$STEP_RE" 2>/dev/null; pkill -f "$WORK_RE" 2>/dev/null
                sleep 15
                if pgrep -f "$WORK_RE" >/dev/null 2>&1 || pgrep -f "$CAMPAIGN_RE" >/dev/null 2>&1; then
                    pkill -9 -f "$CAMPAIGN_RE" 2>/dev/null; pkill -9 -f "$WORK_RE" 2>/dev/null; sleep 5
                fi
                work_seen_at=0; last_seen=0
            fi
        else
            if [ "$work_seen_at" -ne 0 ]; then log "no work step running; clock cleared"; fi
            work_seen_at=0; last_seen=0
            log "campaign running, waiting on its own free-memory gate (free=${free}MiB)"
        fi
    else
        work_seen_at=0; last_seen=0
        if [ "$free" -ge "$FREE_MIN" ]; then
            tf=$(probe_tf); case "${tf:-}" in ''|*[!0-9]*) tf=0 ;; esac
            if [ "$tf" -ge "$TF_MIN" ]; then
                log "window: free=${free}MiB tf=${tf}TFLOPS - starting campaign"
                cd "$REPO" || exit 1
                nohup setsid bash "$CODE/run_campaign.sh" >> "$LOGS/run_campaign.log" 2>&1 &
                log "campaign started pid=$!"
                sleep 30
                continue
            fi
            log "free=${free}MiB but slow: tf=${tf}TFLOPS < ${TF_MIN} - waiting"
        else
            log "no window: free=${free}MiB < ${FREE_MIN}MiB - waiting"
        fi
    fi
    sleep "$SLEEP"
done
