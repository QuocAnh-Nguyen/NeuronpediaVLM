# J-lens on LLaVA-1.5: assumption & decision register

Every load-bearing assumption behind `vlm_lens`, why it exists, what breaks if it is wrong, and
the cheapest experiment that falsifies it. Read §2 and §7 before spending GPU time; §2 ranks the
VLM-vs-LLM differences that can make the lens behave unlike its text-LLM original.

Status tags: `[verified]` = checked against this repo / this environment today; `[paper]` =
inherited from the J-lens reference implementation (`third_party/jacobian-lens`, upstream paper);
`[inference]` = reasoned, not yet measured.

---

## 0. Facts this register rests on

| # | Fact | Evidence |
| --- | --- | --- |
| F1 | Upstream estimator: **one** position mask (`skip_first .. seq-2`) is used as *both* source set and target set; there is no target-mask parameter. | `jlens/fitting.py:45-72,100-133` `[verified]` |
| F2 | Our extension keeps that: source masks `text/image/all` are averaged from one gradient pass set, but `target_positions` is always the `all` mask. | `fitting.py:117-128` `[verified]` |
| F3 | The scorer *excludes* positions whose next token is an image placeholder. | `evaluate.py:22-23,144` `[verified]` |
| F4 | Prompt `USER: <image>\nDescribe this image.\nASSISTANT:` = **18** text tokens (incl. BOS) + **576** placeholders = 594; 575 of 592 targets predict a placeholder (97.1 %). With a 40-token caption: 93.6 %. | real tokenizer + `[verified]` |
| F5 | The checkpoint config is thin: `config.json` has **no** `image_seq_length` (576 comes from the installed `transformers` `LlavaConfig` class default), and `text_config` carries only 5 keys (`_name_or_path: lmsys/vicuna-7b-v1.5`) — 32 layers / 4096 width are `LlamaConfig` class defaults. | `config.json` + resolved `LlavaConfig` `[verified]` |
| F6 | Placeholder-count guard is **fail-open**: it is skipped when `image_seq_length` resolves to 0. | `models/llava.py:129,233` `[verified]` |
| F7 | HF path = `get_image_features` (layer −2, `default` select ⇒ 576 patch tokens, no CLS) → `masked_scatter` at id 32000 → `language_model`, causal (`is_causal=True`, `create_causal_mask`). | installed `transformers` 5.x sources `[verified]` |
| F8 | Causal masking spans the image block: patch *i* attends only to patches ≤ *i*. | F7 `[verified]` |
| F9 | Preprocessing: shortest-edge 336 → center crop 336, CLIP mean/std (`do_center_crop=True`). | `preprocessor_config.json` `[verified]` |
| F10 | Fit mechanics: one forward on a `dim_batch`-fold expanded batch (views, no copy); `ceil(d_model/dim_batch)` backward passes with `retain_graph` on all but the last; graph retained from `min(source_layers)`; per-mask rows averaged from the *same* gradients (layers are free). | `fitting.py:144-186`, `models/llava.py:70-95` `[verified]` |
| F11 | Cost model: per sample ≈ `(2·d_model + dim_batch)` forward-equivalents. `dim_batch` is a memory knob, not a compute knob. | upstream docstring + algebra `[inference]` |
| F12 | Intervention directions: `v_t = W_U[t] @ Jᵀ`; `apply_edit` adds/ablates/swaps; `swap` builds a 2-column basis `[v_s, v_t]` and uses its pseudo-inverse without orthogonalization or conditioning checks. | `interventions.py:50-105` `[verified]` |
| F13 | Equivalence gate: bit-exact (atol 0) on the tiny fixture; tolerance 1e-5 for the real checkpoint. | `scripts/check_equivalence.py` `[verified]` |
| F14 | Upstream artifact interop (`JacobianLens.load`) and `weights_only=True` provenance round-trip. | `test_artifacts_interop.py` `[verified]` |
| F15 | The J-lens direction for token `t` is `v_t = W_U[t] @ J_l` — a *row* of `W_U J_l`, the gradient of the linear readout w.r.t. the residual. `interventions.lens_vectors` used `W_U[t] @ Jᵀ` (transposed: same scale, wrong direction); the §7 identity check exposed it and it is fixed, with a guard test. | `interventions.py:58-78`, `test_readout_interventions.py::test_lens_vectors_are_readout_gradients` `[verified]` |

---

## 1. Foundational (LLM-side) assumptions

- **A1 — the residual stream is the right workspace.** *Assumed:* interpretable computation is
  carried by block-output (pre-final-norm) residuals, and `unembed(h) = lm_head(final_norm(h))`.
  *Why:* matches the paper and the LLaMA pre-norm layout; verified bit-level on the tiny fixture.
  *If wrong:* hooking a post-norm tensor double-applies the final RMSNorm — readouts silently
  degrade rather than error. *Check:* `forward_residual` + equivalence gate (already enforced).

- **A2 — a single *average* linear map is informative.** *Assumed:* `J_l = E[∂h_final/∂h_l]`
  summarizes the network's disposition despite full nonlinearity. *Why:* paper's evidence on
  text LLMs (verbal reports, generalization). *If wrong:* the mean cancels bimodal structure —
  row *v* of `J_l` averages contexts where `∂logit_v/∂h` points in opposite directions, so the
  lens looks like a blur and fidelity plateaus far below the model. *Check:* per-layer
  rank/KL/agreement curves (must improve monotonically with depth) plus a per-sample dispersion
  diagnostic, not just `mean_rel_change`.

- **A3 — sum-over-later-targets semantics.** *Assumed:* seeding the cotangent at every valid
  target position `p'` and summing means each source row measures the source's influence on all
  later outputs. *Why:* the estimator's definition. *If wrong:* a source that influences many
  targets is over-weighted relative to one that influences few (positions near the end of the
  span get systematically less cotangent mass). *Check:* compare the narrower-target variant (X1)
  and inspect whether late-position sources behave anomalously.

- **A4 — corpus-averaged lens is transferable within the corpus.** *Assumed:* averaging over
  many images/questions yields a lens that is meaningful per sample. *Why:* the estimator
  variance falls as `1/√n`. *If wrong:* low variance still permits high bias (an artifact of one
  question template that is perfectly stable). *Check:* diversity is enforced in the caption
  corpus; verify by cross-corpus fit/score (X8).

- **A5 — `mean_rel_change` is a sufficient convergence signal.** *Assumed:* diminishing change of
  the running mean implies a converged estimator. *If wrong:* you stop at a stable-but-biased
  lens. *Check:* X1/X8 — always pair with held-out fidelity rows.

- **A6 — target = final block output (layer 31).** *Assumed:* the lens predicts what the model
  emits. *Why:* `target_layer` defaults to `n_layers-1`; the paper notes the penultimate layer can
  be better conditioned. *If wrong:* readouts are noisier than necessary. *Check:* X5.

---

## 2. VLM-vs-LLM differences that can make the lens behave unlike the paper

Ranked by expected impact on results.

- **V1 — for image sources the cotangent mass lands almost entirely on *other image positions*, not
  on the text the model actually predicts.** For a source patch at position `p` (image positions
  1..576) the estimator's targets are `p..seq-2`, and positions 1..575 are image positions whose
  next token is another placeholder. Weighted by the estimator's cotangent mass, ~94 % of the
  `image` (and likewise `all`) rows measure "how does this patch's residual influence the final
  residuals at later *patch* positions"; only ~6 % measure influence on *text* positions — the
  ones where the model's next-token distribution is the thing we want to interpret. (Per-position
  ratios average 89 %; the first patch alone is 97 %.) The `text` rows are **unaffected**: text
  sources sit after the block and can only causally reach text targets — a clean property worth
  remembering. *Why it exists:* the estimator inherits a single source/target mask (F1) designed
  where every token's next token carries meaning. *If wrong:* the `image`/`all` lenses mostly
  encode intra-block visual continuation; interpreting their readouts as verbal dispositions is
  unjustified, and the "image" lens the hallucination analysis needs (patch → caption influence)
  is not what we fit. *Check:* X1 (target-mask variant, same cost) — predict the image rows'
  influence-on-text improves while nothing else changes.

- **V2 — image tokens are not exchangeable.** Causal masking inside the block (F8) means the second
  patch has seen one patch and the last patch (position 576) has seen 575; their residuals have
  different information content and different norms. *Why it matters:* the `image` mask averages
  over the whole block, blurring a strong positional gradient. *If wrong:* the image lens is an
  average of 576 distinct estimands. *Check:* X2 (first vs last quarter as separate tags), X3.

- **V3 — `all` ≈ `image`.** 576 of ~594 positions are patches. *Assumed:* `all` is a neutral
  default. *If wrong:* users treat an image-continuation lens as a general-purpose lens. *Check:*
  report mask composition next to every score row (already in provenance: `mask_positions`).

- **V4 — image-position readouts have no natural ground truth.** The model never *emits* a patch
  embedding; its next-token distribution at a patch position is an artifact of the LM head applied
  to a patch residual. *Assumed:* lens-vs-model agreement at image positions still measures
  fidelity. *If wrong:* image-tag fidelity numbers are uninterpretable. *Check:* report
  fidelity both with and without placeholder targets (see V5) and treat image-tag values as
  descriptive only.

- **V5 — the scorer's placeholder exclusion silently narrows the `image` tag to one position per
  sample.** With a single contiguous image block, positions whose next token is *not* a placeholder
  within the image mask reduce to exactly `p = last patch`. So the `image` row of a score table has
  `n = #samples`, not `576·#samples` — high variance, and it measures "the last patch's text
  prediction", not the block. *Why it exists:* predicting a placeholder is not a verbalization
  target (F3) — correct for *interpretation*, wrong for *model-matching*. *Check:* add a
  `include_placeholders` scoring mode reporting both, and split the image mask by position
  (V2) so that the text-prediction question is asked of a meaningful subset.

- **V6 — `skip_first=1` vs the paper's 16.** The paper drops 16 leading positions because early
  text tokens are attention sinks with atypical statistics. In a VLM those 16 are patches, so we
  keep them and drop only BOS. *Why:* dropping 16 real patches would discard real content.
  *If wrong:* early patches are sinks too (their residuals may be dominated by a few massive
  activations) and they skew `image`/`all` rows. *Check:* X4 (`skip_first ∈ {1,8,16,32}` on the
  same shard) + the X3 norm census — if layer-0 norms at positions 1-16 look like BOS, raise it.

- **V7 — the vision tower and projector are outside the lens.** *Assumed:* hallucination-relevant
  computation worth localizing lives in the LM. *If wrong:* a wrong object attribute may already
  be baked into the patch embeddings, and no LM-side lens can see it; the lens will faithfully
  show the model "reading out" a feature that was never in the image. *Check:* image-swap
  interventions (swap the image, keep the prompt) and compare residual deltas at patch positions
  against caption changes; flag the vision-side as out of scope in any report.

- **V8 — prompt-format dependence.** The lens is a distribution average; changing the template
  (`USER: <image>...ASSISTANT:`, BOS, spacer tokens) shifts it. *If wrong:* fitting on one format
  and analyzing generations from another silently degrades fidelity. *Check:* hash the prompt
  template into provenance (not yet done) and always analyze with the same template.

- **V9 — multi-image samples.** `expand(n>1)` is only legal with ≤1 image per sample, so
  multi-image work forces `dim_batch=1` → `ceil(d_model/1) = 4096` backward passes per sample
  (F11). *If wrong:* a multi-image POPE variant quietly becomes 8× slower. *Check:* none needed;
  keep the constraint documented (it already raises).

- **V10 — config drift is only detected fail-open.** F5/F6: `image_seq_length=0` would disable
  the placeholder guard, and a different `vision_feature_select_strategy` (e.g. `full` → 577
  tokens) changes the block length. *If wrong:* silent misalignment between placeholder count and
  feature count (HF `masked_scatter` would raise on a count mismatch, so the failure mode is a
  hard error, but only *if* the guard's precondition holds). *Check:* make the guard fail-closed
  (derive 576 from `(image_size/patch_size)²` when the config is silent) and record the vision
  config in `model_fingerprint`.

- **V11 — `llava-hf` ≡ original LLaVA-1.5.** We fit *this* implementation. Differences
  (preprocessing, projector precision, patch selection, template) mean lenses do not transfer to
  `liuhaotian/llava-v1.5-7b` checkpoints. *Check:* not planned; state the implementation
  explicitly in every report.

- **V12 — caption-length truncation bias.** `captions.py` drops over-length samples instead of
  truncating (correct: truncation would amputate the caption we analyze). *If wrong:* the corpus
  skews short → less text cotangent mass per sample → noisier text lens. *Check:* log the drop
  rate in provenance (partially present: `skipped`).

---

## 3. Fitting-protocol decisions (register)

| # | Decision | Rationale | Alternative rejected / open |
| --- | --- | --- | --- |
| D1 | Vendor `jlens` unmodified; extensions live in `vlm_lens` | Estimator comparability with the paper; artifact interop | Patching upstream breaks provenance (`third_party/NOTICE.md`) |
| D2 | Use HF `LlavaModel` fusion verbatim; equivalence gate | Bit-level agreement (F13) removes a whole bug class | Custom fusion (re-implements `masked_scatter`, easy to get wrong) |
| D3 | Hook block outputs; `unembed` applies the final norm | A1 | Using HF `hidden_states[-1]` (post-norm) |
| D4 | Three masks per fit, shared gradient passes | Modality reductions are free (F10) | Separate runs (3× cost, no benefit) |
| D5 | `skip_first=1` multimodal, 16 for the text control | V6 | Fixing 16 everywhere (destroys 15 patches) |
| D6 | `exclude_last=True` | Last position has no next-token target | — |
| D7 | Target set = all valid positions (F2) | Matches upstream exactly | **Open:** target set = text-predicting positions only (V1, X1) |
| D8 | Fit on the model's *own* greedy captions (on-policy) | Analysis happens on the deployment distribution | Neutral text (WikiText control) for a distribution-free baseline |
| D9 | Diversity of questions/objects in the caption corpus | A4 bias mitigation | Single template = perfectly stable, possibly wrong |
| D10 | Reject over-length prompts, never truncate | Truncation can drop image placeholders (silent corruption) | Truncation (much cheaper, breaks the invariant) |
| D11 | Accumulate fp32 on CPU; artifacts fp16 by default | Memory; upstream-compatible files | fp32 artifacts (2× size; `--save-dtype` available) |
| D12 | Checkpoints store sums + counts; resume exact; fingerprint excludes `n_samples` | Resumable and order-independent means | Fingerprinting `n_samples` (blocks legitimate resume) |
| D13 | Shard merge weights each mask by its own `n_prompts` | Masks can have different counts (e.g. no image tokens) | Equal-weight merging (wrong for text-only samples) |
| D14 | Fitting all layers below the target is free | Same backward passes fill every source layer | Fitting a subset only to save compute (saves memory, not FLOPs) |
| D15 | `target_layer = 31` (final) | Directly predicts emitted tokens | Penultimate (paper notes better conditioning) — X5 |
| D16 | `dim_batch = 8` default | Memory knob only (F11) | Larger (OOM risk on 4096-dim rows: forward activations ∝ dim_batch) |
| D17 | One image per sample when `dim_batch > 1` | HF flattens images across the batch | Enforced by a `ValueError` |
| D18 | Greedy generation everywhere | Determinism; hallucination analysis is single-trajectory | Sampling (not reproducible) |
| D19 | No gradient checkpointing | Retained-graph `autograd.grad` against intermediate activations is incompatible; a recorder would also double-count recomputed forwards | `use_reentrant=False` + recompute-aware recorder (large change) |
| D20 | bf16 fit on CUDA | Memory/speed on H100; grads cast to fp32 | fp32/tf32 fit (X6 checks whether bf16 gradients bias J) |
| D21 | `torch.compile` off by default | Hook/recompile interactions unverified (listed as a gap) | Per-block compile (keep off for the first real fit) |
| D22 | All readout/intervention paths under `no_grad` | Prevents graph retention during analysis | — |

---

## 4. Analysis-side assumptions

- **E1 — lens logits approximate the model's next-token distribution.** *Why:* the whole point of
  the lens; validated in the paper for text LLMs. *If wrong:* every "the model was about to say X"
  claim fails. *Check:* `evaluate.score_lens` on held-out samples must show text-tag `rank_true`
  far below `vocab/2` and `top1_agree` well above chance; report the model ceiling row alongside
  (it is the upper bound).
- **E2 — the final-layer row is an anchor, not evidence.** It is the model by construction
  (E2 in code: `use_jacobian and layer == final_layer` → uses model logits).
- **E3 — image-tag fidelity is descriptive.** See V3/V4/V5; do not report image-tag rank as
  "the model's belief".
- **E4 — linearization vs intervention.** A lens readout computed *after* an edit is the linear
  model's prediction, not the edited network's behavior. *If wrong:* a "reasonable-looking"
  post-edit readout can be pure artifact. *Check:* causal claims always come from
  `generate_with_edits` (actual generation), with the lens used for hypothesis formation only.
- **E5 — α has different units per mode.** `add` adds `α·v` (norm scales with ‖v‖); `ablate`
  removes `α·(h·v̂)v̂` (α is a projection fraction); `swap` translates between two basis
  coordinates, and if `[v_s, v_t]` is nearly collinear the pseudo-inverse amplifies the edit
  (F12). *Check:* X7 — log `‖v‖`, layer residual norms, and the basis condition number; prefer
  residual-relative α (`α = k·‖h‖/‖v‖`).
- **E6 — holding out.** Score on a manifest disjoint from the fit (same distribution) and use
  those rows as the stopping rule (A5); cross-corpus scoring is an extrapolation test (X8).
- **E7 — intervention directions ignore the final RMSNorm's Jacobian.** `lens_vectors` uses the
  raw `W_U` row while the readout applies `lm_head ∘ final_norm`, whose Jacobian depends on the
  residual. So `v_t` is the exact gradient of the *lens's linear* readout, not of the model's
  logits, and `add`/`swap` only approximate "raise token t" in the real network (the paper's
  recipe has the same caveat). *Check:* X9 (actual generation) plus the residual-norm census (X3).

---

## 5. Engine / deployment assumptions

- **F1 — bf16 gradients are good enough.** The forward is bf16 (HF default), so autograd yields
  the bf16 graph's Jacobian; accumulation is fp32 (D11) but per-element precision is bf16.
  *If wrong:* J has ~1e-2 relative noise per sample; averaged over thousands of samples it may
  still be fine, but small-nods structure can vanish. *Check:* X6 (bf16 vs fp32 on 8 samples:
  relative Frobenius difference of the merged J).
- **F2 — memory is dominated by the retained graph times `dim_batch` from `min(source_layers)`.**
  *If wrong:* OOM at 594 tokens × 32 layers × bf16 attention intermediates × dim_batch. *Levers:*
  smaller `dim_batch`, a higher `min(source_layers)` (fit only deep layers), shorter sequences.
  *Check:* measured — M3 (47.6 GB peak at `dim_batch 8`, 67.5 GB at 16, OOM at 32 on a shared 80 GB card).
- **F3 — per-sample serial loop.** One sample = one forward + `ceil(d_model/dim_batch)` backward
  passes; shards are the parallelism unit. *Check:* measured — M2/M3 (266-345 s per sample; all-layer fits have the same FLOPs, D14).
- **F4 — artifacts stay upstream-loadable** (F14). Any new key must survive `weights_only=True`.
- **F5 — single GPU.** No cross-device reduction implemented; merge is CPU-side.
- **F6 — FLOP budget.** `[inference]` ≈ 8192 forward-equivalents per sample at `d=4096`
  (F11); for a 594-token, 7B LLaVA that is ~7e16 FLOPs ⇒ **minutes per sample** on one H100,
  independent of `dim_batch` and of how many source layers or targets you request. Plan a
  100-image pilot (order hours), not 5 000 images. *Check:* measured — M2/M3: 266 s per sample at
  ~660 tokens, **independent of `dim_batch`**; extrapolate from measurement, not from this estimate.
- **F7 — CPU RAM for merged artifacts.** 32 layers × 4096² × 2 bytes ≈ 1.07 GiB per mask (fp16)
  ⇒ ~3.2 GiB for three masks, doubling if loaded fp32. Fine locally; analyses should load one
  mask at a time.

---

## 5.1 Pilot measurements (2026-10-01, single H100 80 GB, bf16, co-tenant holding ~10 GB)

All numbers below are from the real checkpoint (`llava-hf/llava-1.5-7b-hf`) via
`scripts/fit_llava.py`; they replace the estimates §5 asked for.

| # | Measurement | Value |
| --- | --- | --- |
| M1 | Equivalence gate (1 COCO image, cuda/bf16) | `EQUIVALENCE PASS`; `max\|diff\| = 0.000e+00` on all three checks (1.9e7 lens-vs-HF logits), 25 s wall. F13 now verified for the real checkpoint, not just the fixture |
| M2 | Fit cost, 3-image pilot (masks text/image/all, layers 0/16/30, `dim_batch 8`, ~660-token samples) | 1034.5 s for 3 samples ≈ 345 s/sample |
| M3 | `dim_batch` sweep, 1 sample, text mask | 8 → 265.8 s / 47.6 GB peak; 16 → 266.9 s / 67.5 GB peak; 32 and 64 → CUDA OOM. **Zero speedup from a larger `dim_batch`** — F11's "memory knob, not a compute knob" is now measured |
| M4 | Artifact size | 100.7 MB per mask with 3 layers; 32 layers ≈ 1.07 GB per mask (F7) |
| M5 | Real-artifact interop | `torch.load(..., weights_only=True)` keys exactly `J`/`d_model`/`n_prompts`/`source_layers`, `J` a dict keyed by layer; `jlens.JacobianLens.load` loads it; `transport(h, 16)` equals raw `J[16] @ h`; CLI `--merge` of two 1-sample runs gives `n_prompts=2`, `max\|dJ\| = 2e-3` (fp16 rounding), provenance `merged_from` recorded |
| M6 | Fidelity, 3 training images, text lens (`score_lens`) | rank_true 19710 → 1198 → 236 → 37.6 (layer 0 → 16 → 30 → final); top1_agree 0.19 → 0.78; KL 12.0 → 0.38 — depth-monotone, as A2's check demands (train-set values; hold-out rows are the analysis-side gate, E6) |
| M7 | Image-token readout, descriptive (n=1 position) | last-layer (model) top-5 at a mid-block patch of a tennis-court image: `court, courts, Court, Centre`; layer 16: `Stadium, Tennis, Singles, Championships, Basketball` — coherent "disposition to verbalize" upstream of the model's own readout, but see V4/V5 |
| M8 | Environment (deployment landmines found) | Weights live in `~/.cache/huggingface/hub` (repo dirs are symlink shells into the cache-root `blobs/`, 35 GB); shard headers verified (686 tensors = index). `HF_HOME` must **not** point at `/data_mount` — exFAT cannot hold symlinks, so HF falls back to copying (30 GB > 17 GB free). The env's `torchvision` is an editable install of another project's source tree (`vlm-truth/…/CausalLens/vision/torchvision`, no libjpeg) ⇒ never hand the HF processor jpeg paths/bytes; use PIL (`captions._open_image`, `load_image`) |
| M9 | Bugs the real checkpoint exposed that the CPU fixture cannot (all fixed) | (a) `captions.generate_captions` handed `pathlib.Path`/`str` images to the HF processor → transformers 5.x `TypeError`, and paths/bytes route through `torchvision.io.decode_image`, which the cluster env's jpeg-less torchvision rejects; now passes PIL (`_open_image`), pinned by `test_data_captions.py`. (b) `lens_readout` rejected the final layer although its docstring promises it is always available; now allowed and equals `model_logits` (`test_lens_readout_final_layer_is_the_model_logits`). (c) `apply_edit` mixed CPU lens vectors with CUDA residuals → `RuntimeError: Expected all tensors to be on the same device`; `_aligned` now casts device+dtype (`test_apply_edit_aligns_vectors_to_the_residual`) |
| M10 | Real-weight analysis paths | `trace_generation` (24 tokens, layers 0/16/30) runs at ~2 s; step alignment holds (hook records == new tokens); layer 30 tracks the model step-by-step (emitted `dog` 0.166 vs lens 0.166 at step 4) while layer 16 is semantically adjacent but noisy. `generate_with_edits`: `add dog @L16` at α = 3‖h‖/‖v‖ = 80 collapses the caption to "dog dog dog…"; `swap cat→dog @L16` (α=1) rewrites it ("…a shelf with a variety of shoes"); `ablate ▁The @L16` leaves it unchanged. 0.6-1.0 s per 12-token edited generation |
| M11 | Layer-0/16 residual norms (X3 first cut, 1 image) | L0: BOS 8.3, patch1 3.0, patch16 20.2, patch575 33.4, last patch 79.9, first text 25.0; argmax = position 117 inside the block (639). L16: BOS **1568.8** (the massive-activation/attention-sink position by mid-depth), patches 26-48, text ~30. Early patches are *not* sinks at L0; the block has a strong positional norm gradient (V2/V6 evidence; `skip_first=1` already drops BOS) |
| M12 | Held-out fidelity, lens-text fit on 10 images, scored on 8 disjoint images (n=448 text positions) | rank_true 24732 → 1901 (L10) → **3095 (L20)** → 357 (L30) → 62.9 (model); top1_agree 0.000/0.085/0.241/0.717; KL 13.55/6.32/6.90/0.433. Layer 30 generalizes (0.72 top-1 for a 10-image average), but **mid-depth is non-monotone** (L10 beats L20), so A2's monotonicity check does not pass at n=10 — the average is still variance-dominated above ~L10. Image tag (n=8): rank 2206 vs the 47.8 ceiling, and also non-monotone (consistent with V4/V5). First datapoint for E1/E6; the production fit must be scored the same way |

Run outputs on the cluster go to `/data/vlm-lens/runs` (ext4, but the volume is shared and
was at ~31 GB free): 8.8 GB per all-layer run directory (6.2 GB checkpoint + 3 × 992 MiB artifacts).

## 5.2 Campaign measurements (2026-10-01, real checkpoint, 50-sample census + frozen corpora)

The validation campaign that produced `results/validation_2026-10-01/REPORT.md` added the
following measurements; they supersede the 1-sample pre-cuts above where they overlap.

| # | Measurement | Value |
| --- | --- | --- |
| M13 | Cost model, all 31 layers below the target, 3 masks, `dim_batch 8` | image sample (660 tok = 82 text + 576 image): **288.2 s**, 36.4 GiB peak; text sample (190 tok): **85.9 s**, 19.9 GiB peak → 8.0 h / 100 images, 2.4 h / 100 text prompts |
| M14 | **X3 census, 50 COCO samples, layers 0/16/31** (groups: BOS, positions 1–16, text before the block, first 16 patches, patches 17…end, text after the block) | L0: BOS 8.3, pos 1–16 29.8, text_pre 3.1, patch 1–16 38.6, patch 17–end 39.4, text_post 2.2 · L16: BOS **1568.8**, pos 1–16 34.7, text_pre 30.0, patch 1–16 35.8, patch 17–end 38.2, text_post 37.4 · L31: BOS **633.7**, pos 1–16 139.7, text_pre 151.8, patch 1–16 136.0, patch 17–end 155.0, text_post 205.0. BOS is a massive-activation outlier by mid-depth while positions 1–16 track the ordinary text/patch groups (dominant-dim Jaccard vs BOS = 0.111) ⇒ **`pos_1_16_sink_like=false`, `recommended_skip_first=1`**; the conditional X4 sweep is not triggered (D5 stands, with V6 now measured rather than assumed) |
| M15 | Frozen corpora for the campaign | COCO captions (on-policy, `prompt+caption`): 100 fit images / 30 held-out images, `seed=0`, 10-question bank cycling (10 samples per question; halves A/B are also image-disjoint); WikiText-103: 100 train / 30 validation prompts, 0 dropped, token lengths 139–568 (fit) and 124–400 (held-out) |

The census is `step1/x3_norms.json` (50 samples × 6 groups × 3 layers, `per_sample` rows
retained); the cost model is `step0/cost.json`. Both are regenerated by committed scripts
(`code/x3_census.py`, `code/measure_cost.py`).

## 6. Hallucination-task assumptions (why we are doing this at all)

- **G1 — the target phenomenon is LM-side.** See V7. If attribute errors are baked in upstream,
  the honest deliverable is "the lens shows the LM faithfully reading out a feature that does not
  match the image" — still useful, but a different claim.
- **G2 — "disposition to say" transfers to captions.** The paper's workspace evidence is
  text-LLM, prose tasks. Captioning is template-heavy and short; the lens may be sharper than in
  free-form prose (fewer plausible continuations) — or the task may simply be too shallow.
- **G3 — hallucinated tokens should be *high* in the lens.** Working hypothesis: a hallmark of
  hallucination is the LM's disposition favoring the emitted token despite the image; the honest
  contrast is against *ground-truth-correct* tokens for the same object. Test at token level:
  rank of the emitted noun under lens vs model, split by correctness. This is the first real
  analysis after the pilot fit, not a fit-time decision.
- **G4 — POPE later fits the same loop.** Yes/no answers are single tokens, so per-token lens
  scores suffice — which enables the token-restricted estimator of §7 (no full J needed).

---

## 7. Proposed alternate estimator: token-restricted lens vectors

For intervention work and yes/no analyses, the full `J_l` is overkill: what is needed is
`v_t = W_U[t] J_l` for a small token set `T`. That vector is a gradient:

```
grad_outputs[p', :] = W_U[t]        (identical at every target position p')
⇒ source-averaged grad = v_t = W_U[t] @ J_l      (tiny fixture: matches to 2.2e-8)
```

That is exactly what `interventions.lens_vectors` returns — the residual-space direction that
raises token `t`'s *linear* lens readout, in the same convention as the stored `J` (one-hot
cotangent at the pre-norm residual; the final RMSNorm enters only at readout). Running this check
*before* writing the estimator is what exposed the transposed direction in `lens_vectors` (F15):
the object it returned had the right scale and pointed elsewhere.

So one backward pass per token (or per `dim_batch` tokens, seeding a different token per batch
element) yields the source-averaged vectors for all layers at once. Cost drops from
`ceil(d_model/dim_batch)` to `ceil(|T|/dim_batch)` passes — 40× for 100 content words, 512× for
2 tokens — with no loss for "score token t at this layer" questions. It cannot produce
full-vocab readouts (no rank/KL across the vocabulary), so the full-`J` fit and this estimator
are complementary, not substitutes.

---

## 8. Priority experiments (cheap → expensive)

| # | Experiment | Tests | Decision rule |
| --- | --- | --- | --- |
| X1 | Add a target-mask variant (drop placeholder-position targets) and fit the same shard both ways | V1, D7 | Adopt if the `image`/`all` rows' held-out rank/KL/top-1 improve; the `text` rows *must* be unchanged (V1), so a change there means the implementation is wrong. Same cost (F11) |
| X2 | Split the image mask into per-quarter tags; score held-out fidelity per quarter | V2, V5 | If quarters differ sharply, report block-position-resolved results, not block averages |
| X3 | Residual-norm census by modality and position at layers 0/16/31 on ~50 images | V6, E5 | If positions 1-16 look sink-like, raise `skip_first`; use measured norms for α scaling |
| X4 | `skip_first ∈ {1,8,16,32}` on one shard | V6 | Pick the value that maximizes held-out text fidelity, not the one that "looks like" the paper |
| X5 | `target_layer` 31 vs 30 | A6 | Keep the better-conditioned one |
| X6 | bf16 vs fp32 fit on 8 samples; compare merged J | F1 | If relative difference > a few %, switch the production fit to fp32/tf32 |
| X7 | Condition number of `[v_s, v_t]` for planned swap pairs; sweep α | E5, F12 | Orthogonalize or alert above ~1e3; adopt residual-relative α |
| X8 | Cross-corpus transfer (fit captions → score POPE text positions, and the reverse) | A4, E6 | Quantifies how task-specific the lens is; sets the claim's scope |
| X9 | Edit sanity sweep (`add`/`swap` on ~10 samples, α grid, actual generation) | E4, E5 | Directions that never change generation are mis-scaled, not necessarily meaningless |
| X10 | Implement the §7 token-restricted estimator; cross-check `v_t` against the full-`J` path | §7, F15 | Identity verified by hand on the tiny fixture; implement when a token-restricted analysis (POPE yes/no, content words) is on the critical path |

## 9. Residual uncertainty that cannot be resolved locally (CPU, no weights)

Gradient precision on bf16 attention kernels (X6) and where attention sinks / massive activations
actually sit in a multimodal sequence (X3) remain open. Answered by the 2026-10-01 pilot (§5.1):
activation memory and wall-clock per sample
(M2/M3), depth-monotone fidelity in practice (M6, train-set), real-checkpoint equivalence and
artifact interop (M1/M5). Still unknown until a fit with held-out rows: whether lens fidelity
generalizes beyond the fitting samples, and whether the hallucination signal is strong enough to
detect at all (G3).
