# Repository Guidelines

## Project Overview

`vlm-lens` builds **Jacobian (J) lenses** for `llava-hf/llava-1.5-7b-hf`: `lens_l(h) = unembed(J_l @ h)` where `J_l = E[∂h_final/∂h_l]` is the average input→output Jacobian of the LLM residual stream, estimated by the vendored Anthropic reference implementation. Primary analysis target is captioning hallucination; the data layer is corpus-agnostic so POPE yes/no (VQA) fits work through the same loop.

The upstream package (`jlens`) is vendored **unmodified** at `third_party/jacobian-lens` (Apache-2.0, commit `581d39…`, see `third_party/NOTICE.md`). Never edit it; every extension lives under `src/vlm_lens/` so artifacts stay loadable by upstream `jlens`/TransformerLens.

## Architecture & Data Flow

```
corpus builders                         fit                       analysis
data/text.py | captions.py | pope.py     fitting.fit_masked        readout.lens_readout
        | build_*_manifest                    |                   interventions.*
        v                                     v                   (add/ablate/swap,
data/manifest.py  JSONL: header+FitSample -> artifacts.py          generate_with_edits)
                                            lens-{mask}.pt + provenance.json
```

- Vendor wiring: importing `vlm_lens` runs `_vendor.ensure_jlens()`, which prepends `third_party/jacobian-lens` to `sys.path`; a pip-installed `jlens` is only a fallback when the vendor dir is absent.
- `LlavaLensModel` (`models/llava.py`) wraps HF `LlavaForConditionalGeneration` and delegates image fusion verbatim to HF (`get_image_features` → `masked_scatter` → LM stack); it only adds placeholder-safe encoding, modality masks, and a residual-only forward. `scripts/check_equivalence.py` asserts bit-level agreement with the HF forward.
- Fitting averages gradients over *source* positions selected by masks `{"text", "image", "all"}` (`positions.build_position_masks`); target positions are always the `all` mask. One backward pass set produces all masks.

## Key Directories

| Path | Purpose |
| --- | --- |
| `src/vlm_lens/fitting.py` | Core: `jacobian_for_sample`, `fit_masked` (resumable checkpoints, per-sample diagnostics) |
| `src/vlm_lens/models/llava.py` | HF LLaVA adapter (`LlavaLensModel`, `MultimodalBatch`); `tiny_llava.py` = CPU random-weight fixture model |
| `src/vlm_lens/data/` | Corpus builders → `manifest.py` JSONL; no model imports except optional caption generation |
| `src/vlm_lens/artifacts.py` | `lens-{mask}.pt` I/O + `provenance.json`, shard merging |
| `src/vlm_lens/readout.py`, `interventions.py` | Lens logits/`trace_generation`; J-lens-vector residual edits and edited generation |
| `src/vlm_lens/positions.py` | Source-position masks and summaries |
| `src/vlm_lens/evaluate.py` | Hold-out lens fidelity (`score_lens`: true-token rank, top-1 agreement, KL vs the model) per layer and modality tag |
| `src/vlm_lens/_batch.py` | Shared `as_batch` sample resolver (`FitSample`/`MultimodalBatch`/`str`) used by fit, readout, edits, scoring |
| `tests/` | pytest suite on tiny CPU fixtures (no downloads, no GPU) |
| `scripts/check_equivalence.py` | Equivalence gate: tiny CPU fixture or real checkpoint |
| `scripts/fit_llava.py` | Fit CLI: corpus build/load → `fit_masked` → artifacts; supports `--backend tiny|hf-llava`, `--shard I/N`, `--merge` |
| `scripts/dry_run.py` | CPU-only end-to-end pipeline check (tiny model, synthetic images, shape assertions); `--shape-check` verifies the real config offline |
| `third_party/jacobian-lens/` | Vendored upstream (`jlens`); read-only |

## Development Commands

```bash
pip install -e ".[dev]"                              # package + pytest/matplotlib/datasets
python -m pytest                                     # full suite, ~15 s CPU
python scripts/check_equivalence.py --backend tiny   # CPU equivalence gate (exact match required)
python scripts/dry_run.py                            # end-to-end pipeline dry run on CPU (~1 s, 0.7 GB)
ruff check .                                         # lint config [tool.ruff], line-length 100
```

Real-checkpoint gate before any H100 fit:

```bash
python scripts/check_equivalence.py --backend hf-llava --model llava-hf/llava-1.5-7b-hf \
    --device cuda --dtype bfloat16 --image /path/to.jpg
```

Local dev environment: conda env `vlm-lens` (Python 3.12, `torch 2.14.0+cpu`, `transformers 5.17.0`, pytest 9.1.1); there is no `python` shim on PATH, call e.g. `~/miniforge3/envs/vlm-lens/bin/python -m pytest`. Fitting and corpus building are driven by `scripts/fit_llava.py` (or the library calls it wraps); there are no console scripts.

## Code Conventions & Common Patterns

- Every file: `# SPDX-License-Identifier: Apache-2.0`, `from __future__ import annotations`, module docstring explaining *why*, explicit `__all__`; private helpers prefixed `_`.
- Value objects are frozen dataclasses (`FitSample`, `MultimodalBatch`, `FitInfo`, `ResidualEdit`); accumulators (`FitResult`, `LensReadout`) are mutable.
- Validation errors are `ValueError`; missing paths are `FileNotFoundError`; the per-sample loop catches `ValueError`, logs, records in `FitResult.skipped` and continues.
- All on-disk writes (manifests, checkpoints) are atomic (tmp + `os.replace`). Logging via `logging.getLogger(__name__)`; no prints.
- Heavy imports (`transformers`, `datasets`, `PIL`) are lazy inside functions; HF model/processor are typed `Any`.
- Fitted Jacobians always accumulate in float32 on CPU; returned readouts are float32 CPU. Generation is always greedy (`do_sample=False`); fixtures seed everything.
- `pyproject.toml` lists packages explicitly (`vlm_lens`, `vlm_lens.models`, `vlm_lens.data`) — add new subpackages there by hand.
- Version constants: `MANIFEST_VERSION`, `CHECKPOINT_VERSION`, `PROVENANCE_VERSION` (all `1`). Resume hard-fails on version/fingerprint mismatch; bumping a version invalidates existing files instead of shimming.

### Invariants that silently corrupt lenses if broken

- The lens lives on the **pre-norm** residual (block output, what `ActivationRecorder` hooks). `forward_mm` returns HF's post-final-norm `last_hidden_state` and is *not* the lens tensor; use `forward_residual` — applying the final norm twice is wrong.
- Masks select source positions only; `skip_first=1` is correct for multimodal fits, `skip_first=16` (`TEXT_SKIP_FIRST`) for text-only control fits matching the paper; `exclude_last=True` everywhere (final token has no next-token target).
- `encode_mm` rejects over-length prompts rather than truncating (truncation would drop image placeholders); it verifies tokenized image-token count == `n_images * image_seq_length`.
- `MultimodalBatch.expand(n)` (dim_batch) requires ≤1 image per sample when `n>1`; use `dim_batch=1` for multi-image samples. `dim_batch` is purely a memory knob.
- Artifact `.pt` files must keep exactly the upstream keys `J`/`n_prompts`/`source_layers`/`d_model`; embedded provenance must survive `torch.load(..., weights_only=True)` (plainify non-primitive types).
- `HFLensModel.__init__` mutates the HF model in place (`eval()`, `requires_grad_(False)`, optional per-block `torch.compile`). Hooks must stay on blocks, never a compiled whole module.
- All readout/intervention paths run under `torch.no_grad()`; fitting enables grad and retains only the fitted span (`start_graph_at=min(source_layers)`).
- J-lens-vector directions are rows of `W_U J_l` (`unembed_weight()[t] @ J`); the transpose is a different vector of the same scale and silently misdirects every add/ablate/swap (`test_lens_vectors_are_readout_gradients` guards it).
- Images reach the HF processor as **PIL objects**, never paths: `pathlib.Path` is rejected by transformers 5.x, and str/bytes go through `torchvision.io.decode_image`, which dies on a torchvision built without libjpeg (the cluster env's `torchvision` is an editable install of another project without it). `LlavaLensModel.load_image` and `data/captions._open_image` both open with PIL for this reason; `test_data_captions.py` pins it.

## Important Files

- `scripts/check_equivalence.py` — the pre-fit gate; `--backend tiny` demands exact (atol=0) equality, `hf-llava` a tolerance (`--atol`, default 1e-5). Exit code 0 only when it prints `EQUIVALENCE PASS`.
- `tests/conftest.py` — defines `TINY_CONFIG` (d_model=16, 3 blocks, 16 image tokens, vocab 64), the shared `tiny_model`/`tiny_lenses` fixtures, and the canonical `PROMPT`/`TEXT_ONLY` strings.
- `docs/jlens-vlm-assumptions.md` — assumption & decision register (VLM-vs-LLM differences, failure modes, priority experiments); read before a real fit.
- `src/vlm_lens/models/tiny_llava.py` — real HF classes with mini dims; keeps 576-token image block for 336/14 patches so placeholder logic matches production.
- `src/vlm_lens/_vendor.py` — `VENDOR_DIR`, `ensure_jlens()` (vendored `jlens` wins when the checkout exists, installed package is the fallback).
- Known gaps: root `README.md` carries the workflow but no API docs; `positions.py` still points at `vlm_lens.evaluate` for image-mask validation (now implemented); `src/vlm_lens.egg-info/` is stale build output; `--attn-implementation`/`--compile` paths are untested on real hardware.

## Runtime/Tooling Preferences

- Python ≥3.10 (`target-version = py310`); runtime deps `torch`, `transformers>=5.5` (only pinned dep), `huggingface_hub`, `numpy`, `pillow`; extras `dev = pytest, matplotlib, datasets`.
- Tests and `--backend tiny` run CPU-only with random weights — no network, no GPU, ~8 GB RAM is enough.
- `datasets` is needed only by `data/text.py` (WikiText streamed from HF Hub); `data/captions.py` and `data/pope.py` read local files only.
- Real fits need CUDA (H100 target) with bfloat16; `torch.compile` must not be combined with `device_map="auto"`.

## Testing & QA

- pytest, configured in `pyproject.toml` (`testpaths = ["tests"]`, `addopts = "-q"`); the vendored upstream `tests/` is intentionally not collected.
- Coverage intent: estimator correctness vs a brute-force reference in `test_fitting_masks.py`; mask semantics in `test_positions.py`; readout/intervention math in `test_readout_interventions.py`; artifact round-trip and upstream-key interop in `test_artifacts_interop.py`; hold-out scorer metrics vs a brute-force KL/rank reference in `test_evaluate.py`; synthetic corpus determinism in `test_data_dummy.py`; caption-builder manifest + processor argument contract in `test_data_captions.py`.
- Fixtures are session-scoped and tiny so a full fit + readout runs in seconds; keep new tests on that path — do not require downloads, GPU, or real checkpoints.
- Untested today: the `pope.py`/`text.py` builders, `_vendor.py` internals, and `models/llava.py` error paths (exercised only indirectly via fixtures). Manifest and caption builders are covered by `test_data_dummy.py`/`test_data_captions.py`. Add tests where you touch untested code.
- Before any real fit, run the equivalence gate on the actual checkpoint; after changes to layers/hooks/unembed, re-run `--backend tiny` (it must stay bit-exact).
