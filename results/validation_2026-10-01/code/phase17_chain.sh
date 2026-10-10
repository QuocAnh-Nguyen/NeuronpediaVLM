#!/bin/bash
# P17: round-2 chain - run the held-out-prediction experiments sequentially on one GPU slot
# at a time (the node is shared; parallel launchers would race the free-memory guard since it
# always sorts to the same GPU). Order: x18 (fastest) -> x16 -> x17 -> x15 (built last).
# Each phase script skips when its JSON exists and echoes LABEL_FAILED on error; a failure
# does not stop the chain.
set -u
C=/data/anhnq/NeuronpediaVLM/results/validation_2026-10-01/code
L=/data/anhnq/NeuronpediaVLM/results/validation_2026-10-01/logs
for launcher in phase16_launcher.sh phase14_launcher.sh phase15_launcher.sh phase13_launcher.sh; do
    if [ -f "$C/$launcher" ]; then
        echo ">>> chain: $launcher $(date -Is)"
        bash "$C/$launcher" || echo "CHAIN_STEP_FAILED $launcher"
    else
        echo ">>> chain: $launcher MISSING - skip $(date -Is)"
    fi
done
echo "=== phase17 chain done $(date -Is) ==="
