#!/bin/bash
# P5a: CPU structural dissection of the fitted 100-sample merged J-lens: per-layer
# identity/diagonal/low-rank/symmetry structure. Zero GPU; answers how much of the lens map is
# a scaled identity (the alphaI control) and how compressible the Jacobians are, next to the
# synthetic-lens zoo (P5b) that scores those hypotheses on the held-out.
set -u
R=/data/anhnq/NeuronpediaVLM; V=/data/anhnq/vlm-lens-out/validation
L=$R/results/validation_2026-10-01/logs; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
export HF_HOME=/data/anhnq/hf_cache HF_HUB_OFFLINE=1 PYTHONPATH=$R/src PYTHONUNBUFFERED=1
OUT=$V/step5
mkdir -p $OUT

if [ ! -f "$OUT/jacobian_structure.json" ]; then
    $P $C/jacobian_structure.py --lens-dir $V/s2-merged/artifacts --mask text \
        --json $OUT/jacobian_structure.json --digest $OUT/jacobian_structure_digest.txt \
        || echo "JSTRUCT_FAILED"
else
    echo "[jstruct] present - skip"
fi

echo "=== phase5a launcher done $(date -Is) ==="
