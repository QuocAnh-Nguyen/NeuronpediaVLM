#!/bin/bash
# P32: multi-architecture chain - gates -> llava-13b fit -> qwen2vl fit -> eval suite.
# Sequential (skip-if-present, failure-tolerant); fits use the two-GPU stagger pattern.
set -u
C=/home/nvidia/vlm-run
for launcher in phase28_gates.sh phase29_fit13b.sh phase30_fitqwen.sh phase31_evals.sh; do
    if [ -f "$C/$launcher" ]; then
        echo ">>> chain: $launcher $(date -Is)"
        bash "$C/$launcher" || echo "CHAIN_STEP_FAILED $launcher"
    else
        echo ">>> chain: $launcher MISSING - skip $(date -Is)"
    fi
done
echo "=== phase32 chain done $(date -Is) ==="
