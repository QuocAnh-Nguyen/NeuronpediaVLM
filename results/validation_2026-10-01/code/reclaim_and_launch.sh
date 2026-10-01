#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Reclaim the GPU from the dead chain and immediately park the new campaign on it.
#
# Why this exists: the box is shared with other users. At 04:1x the S1 fit of
# run_rest.sh's chain OOMed (`S1_FIT_FAILED`/`STEP2_FAILED` in rest_chain.log) and the
# OOM'd fit process stayed resident holding ~28 GiB while its parent chain had already
# exited. Free VRAM is first-come-first-served here, so the reclaim and the relaunch must
# happen in one shot: kill campaign leftovers, wait for nvidia-smi to confirm the release,
# launch run_campaign.sh (guarded steps, dim_batch 4). Never touches the X6 retry runner
# or other users' processes.
set -u
REPO=$HOME/ai4life/phuongnh/vlm-lens
RUN=/data/vlm-lens/validation
CODE=$REPO/results/validation_2026-10-01/code
LOGS=$REPO/results/validation_2026-10-01/logs
mkdir -p "$LOGS"
SELF=$$
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
# Campaign-specific patterns only (no co-tenant processes, no run_x6_retry.sh).
PAT='fit_llav[a].py|run_res[t].sh|run_ste[p][0-9].sh|run_campaign[.]sh|s1_scor[e].py|s2_ev[a]l.py|split_halve[s].py|x1_comp[a]re.py|x6_comp[a]re.py|x7_x9_interve[n]tions.py'

# Live-campaign guard: this script exists to clean up a *dead* chain. Killing a live
# supervisor (and its resumable fit) would throw away GPU progress, so it refuses unless
# FORCE=1 is set explicitly.
FORCE=${FORCE:-0}
live=$(pgrep -f 'run_campaign[.]sh|run_res[t].sh' | grep -v "^${SELF}$" || true)
if [ -n "$live" ] && [ "$FORCE" != "1" ]; then
    echo "refusing to reclaim: campaign supervisor(s) alive: $(echo $live | tr '\n' ' ')"
    echo "re-run with FORCE=1 to kill them and relaunch, or wait for the chain to exit."
    exit 1
fi
echo "=== reclaim_and_launch $(date -Is) free=$(free_mib)MiB ==="
victims=$(pgrep -f "$PAT" | grep -v "^${SELF}$" || true)
if [ -n "$victims" ]; then
    echo "leftovers to kill:"
    ps -o pid,ppid,etime,stat,rss,cmd -p $victims | cut -c1-180
    kill -9 $victims 2>/dev/null
else
    echo "no campaign leftovers"
fi

for _ in $(seq 1 30); do
    pgrep -f "$PAT" | grep -v "^${SELF}$" >/dev/null || break
    sleep 2
done
echo "after kill: free=$(free_mib)MiB"
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader | tr '\n' ' '
echo

# Park the campaign on the freed VRAM immediately (guards inside run_campaign.sh).
cd "$REPO" || exit 1
nohup bash "$CODE/run_campaign.sh" >"$LOGS/run_campaign.log" 2>&1 < /dev/null &
echo "run_campaign pid $! launched $(date -Is) free=$(free_mib)MiB"
