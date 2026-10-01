# NeuronpediaVLM - `vlm-lens`

Jacobian (J) lenses for `llava-hf/llava-1.5-7b-hf`: estimate the average input->output
Jacobian of the residual stream, `J_l = E[dh_final / dh_l]`, with the vendored Anthropic
reference estimator, then read out `unembed(J_l @ h)`. Primary analysis target is captioning
hallucination; the corpus layer also covers POPE yes/no (VQA). Every extension lives under
`src/vlm_lens/`; upstream `jlens` is vendored unmodified (see `third_party/NOTICE.md`).

## Setup

```bash
git clone https://github.com/anthropics/jacobian-lens third_party/jacobian-lens
git -C third_party/jacobian-lens checkout 581d398613e5602a5af361e1c34d3a92ea82ba8e
pip install -e ".[dev]"
```

Importing `vlm_lens` prepends `third_party/jacobian-lens` to `sys.path`; an installed `jlens`
package is only the fallback when the checkout is absent (`src/vlm_lens/_vendor.py`).

## Development

```bash
python -m pytest                                     # full suite, CPU-only, ~15 s
python scripts/check_equivalence.py --backend tiny   # exact-equality gate (atol=0)
python scripts/dry_run.py                            # end-to-end pipeline on CPU (~1 s)
ruff check .
```

Real-checkpoint gate before any H100 fit:

```bash
python scripts/check_equivalence.py --backend hf-llava --model llava-hf/llava-1.5-7b-hf \
    --device cuda --dtype bfloat16 --image /path/to.jpg
```

Fitting runs through `scripts/fit_llava.py` (corpus build/load -> `fit_masked` -> artifacts;
`--backend tiny|hf-llava`, `--shard I/N`, `--merge`).

Architecture, invariants and conventions: `AGENTS.md`. Assumption register (VLM-vs-LLM
differences, failure modes): `docs/jlens-vlm-assumptions.md`. Validation campaign results and
recommended production settings: `results/validation_2026-10-01/REPORT.md`.
