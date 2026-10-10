#!/bin/bash
# P20: scale-up chain - x15 n200 then x17 n200 (sequential; shared node, one GPU slot at a
# time, same rationale as phase17). Each phase skips on existing JSON, failure-tolerant.
set -u
C=/data/anhnq/NeuronpediaVLM/results/validation_2026-10-01/code
for launcher in phase18_launcher.sh phase19_launcher.sh; do
    if [ -f "$C/$launcher" ]; then
        echo ">>> chain: $launcher $(date -Is)"
        bash "$C/$launcher" || echo "CHAIN_STEP_FAILED $launcher"
    else
        echo ">>> chain: $launcher MISSING - skip $(date -Is)"
    fi
done
echo "=== phase20 chain done $(date -Is) ==="
