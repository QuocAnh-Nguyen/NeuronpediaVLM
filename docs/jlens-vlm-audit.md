# Audit — Is the Jacobian lens (J-lens) applicable to LLaVA-1.5-7B, and is `vlm_lens` a faithful adaptation?

Date: 2026-10-02. Scope: theoretical feasibility + implementation correctness, read-only.
Paper audited: "Verbalizable Representations Form a Global Workspace in Language Models"
(Anthropic, 2026-07-06, https://transformer-circuits.pub/2026/workspace/), together with its
companion reference implementation vendored at `third_party/jacobian-lens/` (`jlens`, Apache-2.0).
Code audited: `src/vlm_lens/` (fit / readout / interventions / evaluate), `scripts/`, `tests/`,
and the campaign scripts under `results/validation_2026-10-01/code/`.

Every claim below is labeled **[fact]** (verifiable in a cited source), **[inference]** (my
deduction from cited facts; could be wrong), or **[unknown]**. No patches are proposed here —
the brief is diagnostic only.

How this audit was produced: eight parallel read-only units (paper extraction; upstream `jlens`
estimator; `vlm_lens` core; model/HF semantics; interventions/evaluate/campaign scorers; repo
assumptions register; external LLaVA facts + lens precedents; test-coverage evidence), followed
by independent spot-verification of every load-bearing claim by direct re-read of the cited
lines (listed inline as "[verified]"). No unit found a fundamental blocker.

---

## 1. Feasibility verdict

**No fundamental theoretical or mathematical blocker was found.** [inference, from the facts
below] The Jacobian lens is definable and well-posed on LLaVA-1.5-7B's post-fusion LLM residual
stream, and this repo's core math matches the paper's definitions. Three *semantic* caveats and
one *engineering* caveat qualify that verdict (Sections 3–5); none invalidates the approach.

**Facts the verdict rests on**

1. **The method is architecture-generic.** The paper defines
   `J_l = E_{t, t' >= t, prompt}[ d h_final,t' / d h_l,t ]` over the residual stream and reads
   out `lens(h_l) = softmax(W_U · norm(J_l h_l))`; nothing in the definition requires a
   text-only model, and the paper's own related work notes the *logit lens* family "extends to
   non-text tokens, decoding image patches in vision-language models" (workspace §Related work,
   fetch ~L639). [fact]
2. **LLaVA-1.5-7B's LLM is a stock Llama decoder.** The vision tower (CLIP ViT-L/14-336) and
   the 2-layer GELU MLP projector produce 576 embeddings per image, which are
   `masked_scatter`-ed into `inputs_embeds` before block 0; the LM then runs entirely on
   `inputs_embeds` (HF `modeling_llava.py`, `get_image_features` / `forward`; [fact], verified
   against the installed `transformers` in the local env). There is no cross-attention into the
   LM and no per-block vision path: **every image position is an ordinary residual-stream
   position from block 0 onward**, with its block-0 input being the projected patch feature
   (not a token embedding). [fact + inference]
3. **The repo's lens sits at exactly the right tensor.** The fitted/captured activation is the
   Llama decoder-layer output (pre-final-norm residual); the readout callable is the model's
   own head, `unembed = lm_head(final_norm(·))`, inherited from `jlens.hf.HFLensModel`
   (`jlens/hf.py:166-174`; docstring `src/vlm_lens/models/llava.py:126-128`); [verified]. This
   matches the paper's `softmax(W_U norm(J_l h_l))` exactly (final norm *after* transport, once).
4. **No VLM step breaks differentiability or the Jacobian's definition.** J is taken with
   respect to *residual activations*, not pixels; the vision tower/projector are upstream of all
   lens positions and their parameters are frozen in the fit; the gradient path through
   `masked_scatter` is irrelevant because the fit differentiates activations, not inputs.
   [fact + inference]

**Caveats (details in later sections)**

- **Extrapolation, not validation.** The paper contains **zero multimodal experiments**: term
  census over the full text — `multimodal`/`modality`/`modalities`/`video`/`audio` = 0
  occurrences; every `image`/`vision`/`visual` hit is an analogy, related-work citation, UI
  text, or speculation (workspace passim; [fact], independently re-verified). Applying the
  J-lens to a VLM "neither validates nor invalidates" it as far as the paper goes. [inference]
  Positive external precedents exist but are for *other* lens types on VLMs: logit-lens maps on
  LLaVA-v1.5-7B (Esmaeilkhani & Latecki, arXiv:2602.01530), visual-token vocabulary-space
  reading + causal ablation (Neo et al., arXiv:2410.07149), tuned-lens on MLLMs (VisLens,
  arXiv:2608.30705; also Phukan et al. arXiv:2411.19187; Wang et al. arXiv:2608.07302). [fact]
- **Upstream engineering limitation (not fundamental).** `jlens` as shipped cannot fit or read
  out *any* image-conditioned activation: its HF adapter's `forward` calls the inner text
  decoder directly (`jlens/hf.py:163-164`), so `pixel_values`/`inputs_embeds` are
  unrepresentable and image placeholders degrade to token embeddings. [fact] The repo resolves
  this with its own multimodal forward fork (`src/vlm_lens/models/llava.py:307-320`), verified
  bit-exact vs HF on the fixture and within tolerance on the real checkpoint by
  `scripts/check_equivalence.py`. [fact, campaign]
- **What the lens *means* at image positions is an interpretation, not a prediction.** The
  paper's readout is a next-token distribution; at an image position there is no "next token"
  in the intended sense (the model continues the patch block or starts the answer), so
  image-position readouts have no natural ground truth. [inference; registered in the repo as
  V4] This motivates the repo's modality masks — and its `text` mask is the paper-shaped lens;
  `image`/`all` are exploratory (Section 3).

**STOP-condition check (per the brief):** no theoretical or mathematical reason was found that
makes J-lens on this VLM fundamentally impossible or invalid. The obstacles are: (a) unvalidated
scope extension (fixable by validation, not by math), (b) interpretation limits of image/all
masks (registered, has a pending experiment X1), (c) paper-default target-layer mismatch
(unmeasured, X5 open). All are theoretically addressable.

---

## 2. Definition-faithfulness: paper ↔ upstream `jlens` ↔ `vlm_lens`

| # | Item | Paper | Upstream `jlens` | `vlm_lens` | Verdict |
|---|------|-------|------------------|------------|---------|
| 1 | J_l definition | `E_{t,t'>=t,prompt}[∂h_final,t'/∂h_l,t]` (workspace §methods-jlens) | Same; `t'>=t` automatic (cotangent at all valid targets; causality zeroes `t'<t`) | Same estimator, re-implemented (see #9) | **match** [fact; estimator pseudocode re-verified by me: `grad_z = e_i ⊗ 1_T`, `J[i,:] = mean_t G[t,:]`, workspace ~L952-966] |
| 2 | Estimator mechanics | one backward per output dim (batched), cotangents summed over targets; per-element mean over prompts | one forward per prompt, `ceil(d/dim_batch)` backwards; row block per pass; fp32 accumulation | identical mechanics; one gradient pass-set serves all source masks | **match** (`src/vlm_lens/fitting.py:156-200`, verified) |
| 3 | Sum-over-targets vs uniform mean over `t'` | Pseudocode sums (`e_i ⊗ 1_T`); the text gloss writes an expectation over `t'>=t` | Sums | Sums | **nuance, not deviation**: sources near the start carry more summed terms; the paper's own pseudocode has this weighing. [fact + inference] |
| 4 | Readout | `softmax(W_U norm(J_l h_l))` — norm *after* transport | `unembed = lm_head(final_norm(·))`; `transport = J@h` bare (`lens.py:135-143`) | inherited `unembed`; readout `model.unembed(J@h)` (`readout.py:139-142`) | **match** [verified] |
| 5 | J-lens vectors | "rows of `W_U J_l`" | no constructor shipped | `lens_vectors = W_U[t] @ J` (`interventions.py:69-77`) | **match**; an earlier transposed-J bug (`W_U[t] @ Jᵀ`) was found and fixed, with a guard test + register note. [fact] |
| 6 | Interventions | `h ← h + α v_t`; ablation by projecting out `v_t` (or negative α); coordinate patch `h + V(σ(c) − c)`, `c = V†h` | no intervention code at all; prose protocol only | `add` = `h + α v_t`; `ablate` = project-out with dimensionless α (paper's projecting-out option); `swap` = coordinate patch with σ = 2-cycle (flip) | **match**, with two scope differences: single layer + position ∈ {last, all} vs the paper's band-of-layers protocol; α for `add` in the X9 script is residual-relative (`k·‖h‖·v̂`) per the upstream README convention. [fact; semantic] |
| 7 | Target layer | **Default lens used throughout the paper targets the penultimate layer**; "including the last layer can sometimes increase the number of noisy artifacts" (workspace §app-method-details ~L908-926); the upstream docstring even recommends testing it (`jlens/fitting.py:124-127`) | `target_layer` arg, default final | targets the **final** layer (default; X5 comparing 31 vs 30 was **not run** — `REPORT.md:195,406`) | **deviation, unmeasured** [fact; effect unknown] |
| 8 | Corpus | 1000 × 128-token pretraining-like prompts; "beats baselines with as few as 10 prompts" (workspace §app-method-details ~L938) | none shipped (a WikiText-103 loader as example) | 100 × ≤1536-token samples: WikiText for the S1 text control, LLaVA-generated COCO captions for S2 | **adaptation** — size is defensible [fact: paper's 10-prompt floor]; the caption corpus changes the lens's context distribution by construction (lens is corpus-conditioned; the paper's own "burn-in"/distribution ablations found no meaningful gain from pruning). [fact + inference] |
| 9 | Estimator code reuse | — | `jacobian_for_prompt` with per-layer device-move, logging | duplicated loop; `jacobian_for_prompt`/`valid_position_mask` are dead code here; upstream's per-layer `.to(grad.device)` is not copied; module docstring's "the estimator itself is unchanged" is inaccurate as written | **engineering deviation**, maintainability risk [fact] |
| 10 | Position masks | source `t` all positions; the paper tried excluding the first positions ("no meaningful improvement") | `skip_first=16` hard default (attention-sink rationale), `exclude_last` in the valid mask | `skip_first=1` for multimodal fits (position 0 = BOS; 1-16 are patches), 16 for the S1 text control (passed explicitly), last position always excluded; adds `text`/`image`/`all` + quarter masks | **deliberate, measured deviation** (X3 census: `pos_1_16_sink_like=false`, BOS is the outlier) [fact] |
| 11 | Aggregation weights | per-element mean over prompts | same | per-sample mean over mask positions, then equal-weight over samples/merges (mask-size differences do not shift weights) | **defensible deviation-in-detail** [inference] |
| 12 | `max_seq_len` | 128, truncated | truncate at `max_seq_len` | 1536, over-length **rejected** (placeholder-safe; never truncate) | **adaptation** [fact] |
| 13 | `dim_batch` | n/a (batched "in practice") | rows-per-backward block; replicates the prompt; FLOPs-invariant | same; also used as the memory knob behind the D21 guard change | **match** (memory-only) [fact, mechanics verified] |
| 14 | Resume | — | sums + counts | sums + counts + `next_idx`, order-preserving; test asserts closeness to an uninterrupted fit; fingerprint omits fit dtype / attention implementation / manifest identity | **match**, provenance gap [fact] |

---

## 3. VLM-specific semantics and interpretation risks

1. **The `image` (and dominant-`all`) Jacobians are mostly *intra-block continuation*, not
   verbalization.** With the default `target_mask='all'`, ~94 % (`image`) / ~97 % (`all`) of
   target cotangent mass at image-source rows lands on *later patch positions* — the model
   predicting image placeholders — not on text positions (register V1; `src/vlm_lens/fitting.py`
   docstring; `x1_compare` trims pre-block text that cannot causally reach image targets).
   [measured] Consequence [inference]: those lenses are not "disposition to verbalize" lenses;
   the `text` lens is the paper-shaped one. The X1 experiment (`target_mask='text'`) exists to
   quantify the difference and is pending.
2. **`all` ≈ `image`** for caption samples (576 of ~594 positions are patches; register V3)
   [fact] — the `all` row is descriptive, not a general-purpose lens.
3. **Image-position fidelity has no ground truth** (V4) — lens-vs-model agreement at image
   positions is not "fidelity" in the paper's sense. [inference]
4. **The scorer's default placeholder exclusion collapses the `image` tag to ~1 position per
   sample** (the last patch), so `image` rows have `n ≈ #samples` (V5); with
   `include_placeholders=True` most scored targets are the placeholder id itself (a near-trivial
   continuation). `mask_composition` reports raw mask sizes (≈576/sample), not scored `n`, so
   the two tables can look contradictory. [fact] The `include_placeholders` switch reports both
   — pending S2/X1.
5. **The final-layer row in score tables is synthetic.** `score_lens` unions the scored set with
   the final layer (`evaluate.py:186`) and substitutes the model's own logits there
   (`evaluate.py:146-153`), giving rank 1 / agreement 1 / KL 0 by construction. Reading the
   "last layer" rows as lens evidence is a misread trap (the docstring calls it a "free sanity
   row"). [fact, verified by me]
6. **Direction caveat shared with the paper:** `v_t = W_U[t] J_l` (and the interior of the swap
   basis) omits the final RMSNorm's Jacobian while the readout applies the norm — so `v_t` is
   the gradient of the *linear* readout `W_U[t]·(J_l h)`, not of the model's logits. This is the
   paper's own definition ("up to a data-dependent normalization factor"), registered as E7;
   not a repo-specific error. [fact + inference]
7. **Scope of edits:** the repo intervenes at one layer/position; the paper's protocols steer
   "at every band layer" and across token positions. X9 will therefore test a narrower
   hypothesis than the paper's protocols. [fact] Any nulls under-state the paper's method.

---

## 4. Verification-evidence audit (what is actually proven)

**Tests** (all on the tiny CPU fixture; real-checkpoint paths untested by pytest):

- The "brute-force" reference in `tests/test_fitting_masks.py` is **not independent of the
  estimator's definition**: it imports `build_position_masks` from the code under test
  (`:18`), calls the same `model.forward_mm` (`:51`), and seeds the same "all targets"
  cotangent (`:37-39` vs `fitting.py:183-187`). It independently validates: batch replication,
  the shared-cotangent trick, and per-mask reduction — **not** the estimator semantics, the
  position semantics, or causality. [fact, re-verified by me]
- The readout/intervention tests pin the direction convention and edit algebra; but the
  "transpose guard" compares `lens_vectors` against autograd of the same stored `J`
  (`test_readout_interventions.py:95-109`) — a self-consistency check that **cannot catch a
  globally transposed fit**. The finite-difference row (column-wise, vs perturbed sources) is
  the check that can; it is decoupled from the standard run (D20) and not yet measured on the
  real model. [fact + inference]
- No test covers: real-checkpoint numerics, bf16, the HF layout under generation, resume-after-
  skip, or the causal `t'>=t` relation beyond autograd's own zeros. [fact]

**Campaign evidence** (real checkpoint):

- Equivalence gate passes (tiny exact; real within 1e-5): validates the **forward fork** and the
  final-layer readout chain (`unembed(residual) == HF logits`), not intermediate-layer Jacobians.
  [fact, scope-inference]
- X6 dtype comparison and X3 census are measured; S1's fit is complete; S1's scoring and FD row
  are pending on a GPU window. [fact]
- **Gate row (iii) is vacuous** [verified by me against `evaluate.py:186` + `:146-153` and
  `s1_score.py:291-296`]: because the scored set always includes layer 31 and its J-lens row is
  replaced by the model's own logits, while the logit-lens row at 31 equals the model logits
  too, `iii_last_layer_rank_diff ≡ 0` and the criterion can never fail. The pre-registered
  criterion "J-lens and logit lens agree closely in the last layers" has therefore never been
  tested by that row (the last *fitted* layer, 30, would test it). Companion mismatch:
  `j_not_worse_in_middle` spans all layers below 31, including the noisy first third the
  docstring exempts (`s1_score.py:8-14,292-298`). [fact]
- The published `step2/s1_score.json` predates the gate/FD code (no `gate`, no `check_ii`);
  the FD watcher will republish. [fact]

---

## 5. Defect register (severity-ranked; "fixable" = theoretically, not patched here)

**correctness**
1. `s1_score` check (iii) vacuous (Section 4). Fixable: test at the last fitted layer.
2. `interventions._vectors_for` caches a layer's token→vector table from the *first* edit at
   that layer; a second edit at the same layer with a different token raises `KeyError`
   (`interventions.py:153-169`). Latent — current callers pass one edit per layer. Fixable.
3. Transposed-fit class is guarded only by the (offline) FD row; the unit "guard test" cannot
   catch it (Section 4). Fixable: keep FD row always-on, or add a cheap FD spot-check.

**semantic**
4. Register/docs inconsistencies [fact]: F2 still says "target positions are always the all
   mask" (code has `target_mask`, X1 uses `'text'` — `docs/jlens-vlm-assumptions.md:18` vs
   `fitting.py:113,224`); V3 claims `mask_positions` is in provenance (it is not); the
   `fitting.py` docstring claims the estimator is unchanged (it is a duplicated loop).
5. Paper-default **target layer is penultimate**; the repo targets the final layer and X5 was
   not run — a registered-but-unmeasured deviation from the paper's own "best" recipe
   (Section 2, row 7).
6. Image/all mask semantics (Section 3.1-3.4) — registered; X1/X2 pending.
7. `skip_first` footgun: text manifests fitted via `--manifest` default to 1 unless
   `--skip-first 16` is passed (`fit_llava.py` per audit; `run_step2.sh:19` does pass it).
   A future text fit can silently deviate. Fixable.
8. Transfer asymmetry in `s2_eval` ("caption→WikiText" uses mask `text`, skip 16;
   "WikiText→captions" uses mask `all`, skip 1) mixes corpus, mask, and position protocol in
   one comparison; both are recorded, so it is auditable. [fact]
9. `frequency_control.true_in_top_k` docstring overstates a membership rate as a "guess rate";
   `+1`-smoothed ties make unigram ranks optimistic for unseen tokens. [fact + inference]
10. `x7_x9`: `'degenerate'` flag hardcodes `cond > 1e3` while filtering uses `--cond-max`;
    `cond` of the *raw* 2-column basis conflates scale imbalance with collinearity (cosine and
    norms are logged alongside, so diagnosable). [fact + inference]

**risk**
11. Artifact provenance lacks fit dtype and per-mask position counts; the resume fingerprint
    omits dtype/attention-implementation/manifest identity. [fact]
12. Quarter masks assume `tensor_split` yields equal chunks (true for 576, untested generally).
13. `check (ii)` gate tolerance 5 % vs docstring "1-2 %"; absolute ε not rescaled per layer.
    [fact]
14. X9/paper scope mismatch in interventions (Section 3.7).
15. Verification artifacts can be stale relative to code (published `s1_score.json`); the
    register's own stale rows (Sections 5.4) show this class recurs. [fact]

**no-blocker note**
16. Upstream `jlens` cannot forward image-conditioned batches at all (`jlens/hf.py:163-164`);
    the repo's fork is the workaround and is equivalence-verified. This is an engineering
    limitation of the *reference implementation*, not of the method. [fact]

---

## 6. Facts vs inferences — the load-bearing distinctions

- **Fact:** the paper defines and validates the J-lens only on text-only Claude models; no
  multimodal experiment exists in it. **Inference:** extending it to LLaVA is scientifically
  plausible (the object is definable; positive precedents exist for other lens types) but is
  *unvalidated by the paper* and must be treated as an extension claim.
- **Fact:** the repo's estimator, readout (norm placement), and vector orientations match the
  paper's definitions as implemented upstream. **Inference:** therefore no *math* defect
  invalidates the fitted lenses; the risks are semantic (what the numbers mean) and coverage
  (what has been measured).
- **Fact:** with `target_mask='all'`, image-source rows are dominated by patch-continuation
  targets. **Inference:** an "image lens" read from those rows describes the model's
  placeholder dynamics, not verbalization; the `text` lens is the paper-shaped readout.
- **Fact:** the paper's own default lens targets the penultimate layer; the repo targets the
  final layer and did not run the 31-vs-30 comparison. **Unknown:** the size/direction of the
  effect on this model's lens quality.
- **Fact:** S1's check (iii) is vacuous and the real-model FD row is not yet measured.
  **Inference:** the strongest currently-available correctness evidence for the fitted J is
  the tiny-fixture equivalence (vectorization/sharding) plus the campaign's forward-fork gate —
  the estimator-definition-level evidence at 7B scale is still pending.

---

## 7. Open questions (and what closes each)

1. Do image/all lenses materially change under `target_mask='text'`? → X1 (`x1_compare`), pending.
2. Is the image block internally structured (quarters) in lens terms? → X2, pending.
3. Does targeting layer 30 (paper default) change lens quality on LLaVA? → X5 (deferred; run a
   small paired fit).
4. Does the FD row confirm the fitted J at 7B scale? → FD watcher (D20), pending GPU window.
5. Is next-token fidelity (rank/KL/top-1) even the right sanity metric for J-lenses? → The paper
   judges lenses by *intermediate recovery + causal effect*, and says the J-lens is deliberately
   worst at next-token prediction near the output ("motor regime"). The repo's S1 gate is a
   sanity check on that axis; it should not be advertised as the paper's validation paradigm.
   [fact + inference]
6. Do the paper's estimator variants (present-only, frozen-QK, median aggregation) change the
   VLM conclusions? → Not implemented upstream; the paper reports qualitative robustness; out of
   scope unless a result hinges on it. [fact]

---

## 8. Sources

- Paper: https://transformer-circuits.pub/2026/workspace/ — §methods-jlens (definition, readout,
  single-token probe, sparse decomposition), §methods-technical-details (steering/ablation/
  coordinate-patch formulas), §app-method-details ~L904-966 (penultimate-layer default, variants,
  corpus sweep, pseudocode), §Related work ~L639 (logit-lens on image patches), §discuss-limitations.
- Upstream: `third_party/jacobian-lens/README.md`; `jlens/fitting.py` (estimator, `_check_layer_indices`,
  `jacobian_for_prompt`), `jlens/lens.py` (transport/merge), `jlens/hf.py:163-174` (text-only
  forward; `unembed = lm_head∘final_norm`), `data/experiments/README.md` (prose intervention protocol).
- Model: `llava-hf/llava-1.5-7b-hf` `config.json` (text_config `lmsys/vicuna-7b-v1.5`,
  `max_position_embeddings=4096`; vision CLIP ViT-L/14-336; LLM dims only implicitly inherited —
  [inference]); `processor_config.json`; HF `modeling_llava.py` (`masked_scatter` fusion);
  installed `transformers` copy used for line-level verification.
- Precedents (citations verified by the web unit): logit lens (nostalgebraist 2020); Tuned Lens
  (Belrose et al., arXiv:2303.08112); VLM lenses: arXiv:2410.07149, arXiv:2602.01530,
  arXiv:2608.30705, arXiv:2411.19187, arXiv:2608.07302; CLIP interpretation (arXiv:2310.05916).
- Repo evidence: `docs/jlens-vlm-assumptions.md` (V1/V3/V4/V5/V6, D3/D5/D7, E3/E7, F2),
  `results/validation_2026-10-01/REPORT.md` (§§2-4, 9-12; X3/X6 results; X5 not run),
  `DECISIONS.md`, `results/validation_2026-10-01/code/*.py`.
