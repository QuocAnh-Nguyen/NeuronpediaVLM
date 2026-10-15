#!/bin/bash
# Phase-2 smoke: syntax gate + moment census + affine A/B + zoo eval on the tiny CPU
# fixture (336px images, per SliceC's finding that the tiny vision tower is fixed at 336).
set -u
R=/data/anhnq/NeuronpediaVLM; C=$R/results/validation_2026-10-01/code
P=/data/anhnq/envs/vlm_truth_py313/bin/python
T=/data/anhnq/tmp/smoke2; rm -rf $T; mkdir -p $T
export PYTHONPATH=$R/src HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

for f in $C/moment_census.py $C/lens_zoo_eval.py $C/x9b_alpha_sweep.py $C/s2_eval.py \
         $R/src/vlm_lens/readout.py $R/src/vlm_lens/artifacts.py; do
    $P -c "import ast; ast.parse(open('$f').read())" || { echo "SYNTAX_FAIL $f"; exit 1; }
done
echo "SYNTAX_OK"

$P - <<'PY' || exit 1
import sys; sys.path.insert(0, "/data/anhnq/NeuronpediaVLM/src")
import numpy as np, torch
from PIL import Image
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava
from vlm_lens.models.llava import LlavaLensModel, IMAGE_PLACEHOLDER
from vlm_lens.fitting import jacobian_for_sample
from vlm_lens.artifacts import save_lens_set
from vlm_lens.data.manifest import FitSample, write_manifest
from jlens.lens import JacobianLens

T = "/data/anhnq/tmp/smoke2"
torch.manual_seed(0)
hf, proc = build_tiny_llava(TinyLlavaConfig())
model = LlavaLensModel(hf, proc)
S = 336
paths = []
for i in range(2):
    rng = np.random.default_rng(i)
    arr = rng.integers(0, 255, (S, S, 3), dtype=np.uint8)
    p = f"{T}/img{i}.png"; Image.fromarray(arr).save(p); paths.append(p)
samples = [
    FitSample(
        sample_id=f"tiny-{i}",
        text=f"USER: {IMAGE_PLACEHOLDER}\nDescribe this image.\nASSISTANT: A small test scene.",
        images=(paths[i],),
        meta={"question": "Describe this image."},
    )
    for i in range(2)
]
write_manifest(f"{T}/manifest.jsonl", samples, meta={"corpus": "smoke", "n_images": 2})

jac, info = jacobian_for_sample(model, samples[0], [0, 1], masks=("text", "all"), dim_batch=1,
                                target_mask="all", skip_first=1, max_seq_len=1536)
d_model = next(iter(jac["text"].values())).shape[0]
lenses = {m: JacobianLens(jacobians={l: jac[m][l] for l in jac[m]}, n_prompts=1, d_model=d_model)
          for m in jac}
save_lens_set(f"{T}/tiny-lens", lenses, provenance={"provenance_version": 1, "masks": list(lenses), "notes": "smoke"})
print("TINY_LENS_OK", {m: sorted(lenses[m].jacobians) for m in lenses})
PY

$P $C/moment_census.py --lens-dir $T/tiny-lens --mask text --manifest $T/manifest.jsonl \
    --out $T/bias-text.pt --json $T/bias-text.json --backend tiny --skip-first 1 --max-seq-len 1536 \
    || { echo CENSUS_FAIL; exit 1; }
echo "CENSUS_OK"

$P - <<'PY' || exit 1
import sys; sys.path.insert(0, "/data/anhnq/NeuronpediaVLM/src")
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava
from vlm_lens.models.llava import LlavaLensModel
from vlm_lens.artifacts import load_lens_set, load_bias
from vlm_lens.data.manifest import read_manifest
from vlm_lens.readout import lens_readout

T = "/data/anhnq/tmp/smoke2"
hf, proc = build_tiny_llava(TinyLlavaConfig())
model = LlavaLensModel(hf, proc)
lenses, _ = load_lens_set(f"{T}/tiny-lens")
lens = lenses["text"]
payload, meta = load_bias(f"{T}/bias-text.pt")
bias = {int(k): v["bias"] for k, v in payload.items()}
scale = {int(k): v["scale"] for k, v in payload.items()}
sample = read_manifest(f"{T}/manifest.jsonl")[0]
out_plain = lens_readout(model, lens, sample, max_seq_len=1536)
out_aff = lens_readout(model, lens, sample, max_seq_len=1536, bias=bias, scale=scale)
layer = sorted(lens.jacobians)[0]
diff = float((out_plain.lens_logits[layer] - out_aff.lens_logits[layer]).abs().max())
print("AFFINE_DIFF", diff)
assert diff > 0, "affine correction had no effect"
print("AFFINE_AB_OK")
PY

$P $C/lens_zoo_eval.py --heldout-manifest $T/manifest.jsonl --json $T/zoo.json \
    --lens tiny=$T/tiny-lens --backend tiny --limit 1 || { echo ZOO_FAIL; exit 1; }
echo "ZOO_OK"
echo "=== smoke done ==="
