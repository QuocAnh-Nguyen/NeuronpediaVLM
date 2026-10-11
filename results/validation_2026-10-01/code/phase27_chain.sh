#!/bin/bash
# P27: workspace-replication chain - x20 evals -> x21 report arms -> x22 reasoning arms ->
# x23 VLM-only arms, sequentially (one GPU slot at a time; shared node). Skip-if-present,
# failure-tolerant. Home-redirected per D46.
set -u
C=/home/nvidia/vlm-run
for launcher in phase23_launcher.sh phase24_launcher.sh phase25_launcher.sh phase26_launcher.sh; do
    if [ -f "$C/$launcher" ]; then
        echo ">>> chain: $launcher $(date -Is)"
        bash "$C/$launcher" || echo "CHAIN_STEP_FAILED $launcher"
    else
        echo ">>> chain: $launcher MISSING - skip $(date -Is)"
    fi
done
echo "=== phase27 chain done $(date -Is) ==="
