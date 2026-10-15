# J-lens on LLaVA-1.5: experiment atlas (2026-10-01 campaign + extensions)

The instrument is a Jacobian lens: for each source layer `l` it fits the average Jacobian `J_l` of the final
residual stream with respect to the layer-`l` residual state over a corpus of multimodal samples, and reads out
`z_l = unembed(J_l h_l)` — a per-position, per-layer "what is this state disposed to say next" distribution on
the multimodal residual stream of `llava-hf/llava-1.5-7b-hf`. The question driving the campaign: LLaVA
hallucinates objects that are not in the image — can lens readouts, plus causal edits from the same lens
vectors, explain where and why the visual signal loses to the language prior, and cure it without training?
Headline conclusion after four stages (decisions D0–D47 in [`DECISIONS.md`](DECISIONS.md)): the J-lens is a
validated disposition + intervention instrument, not a universal detector; hallucination is the language prior
winning the visual→verbal handoff — the prior leads 7× at L0 (rank ~787 vs ~5358), grounding overtakes it only
at L24–26, and the handoff is text-mediated at the last prompt position; training-free pre-generation detectors
reach 0.985 AUROC on POPE object presence (0.895 for catching the model's own hallucinated yes); a
mean-replacement ablation of the hallucinated concept's J-lens direction at L30 removes 0.90 of hallucinations
with grounded retention 1.00 (the control ablation destroys 0.97→0.35); and the global-workspace structure of
Gurnee et al. 2026 replicates on this VLM, with a novel persistent non-verbal sensory band at image positions.

## How to read this document

- Four stages — **0 — Foundations** (trust the estimator), **1 — The instrument**, **2 — Improving the
  lens** (model-class ladder, scaling), **3 — Hallucination mechanisms**, **4 — Global-workspace
  replication + the VLM novelty** (x20–x23); then operations lessons, in-flight work, licensing claims,
  and the hypothesis register at the end.
- **D-numbers** (D0–D47) point into [`DECISIONS.md`](DECISIONS.md), the authoritative per-experiment record
  (campaign-local; the repository register's D-table is unrelated). Experiment codes X*/S*/P\* follow the
  campaign registers; V1 etc. are the repo register's hypothesis identifiers.
- Every figure is regenerable via `results/validation_2026-10-01/code/figures_instrument.py` (figA*),
  `figures_hallucination.py` (figB*), `figures_workspace.py` (figC*), `figures_d_improve.py` /
  `figures_d_detect.py` / `figures_d_campaign.py` (figD*, the second visualization round covering X1/X7,
  the improvement mechanics, detection details, S2 design, and the operations timeline), and each is
  embedded exactly once below.

---

## Stage 0 — Foundations (why the instrument can be trusted)

### Equivalence gate

- **WHY.** `vlm_lens` re-encodes images (placeholder-safe encoding) and applies modality masks before fitting;
  if either perturbed the model, every Jacobian would be fitted to a different function than the deployed model.
- **HOW.** `scripts/check_equivalence.py` compares our multimodal forward against HuggingFace's own: the
  recorded residual stream must satisfy `unembed(forward_residual(batch)) == hf_model(...).logits`, and the
  stack output (`last_hidden_state`, post final norm) must match. Bit-exact on the tiny CPU fixture;
  tolerance-gated on the real checkpoint; run before every fit chain, which aborts on failure.
- **RESULT.** `EQUIVALENCE PASS` before every fit chain; on the real checkpoint the LM-stack max abs diff
  measured 0 (REPORT §9). The multi-arch gates (phase28) reuse the same script.
- **INTERPRETATION.** The estimator measures the deployed model, not a re-implementation; every downstream
  number inherits this anchor.

### Corpora + cost model

- **WHY.** Fit and score on-policy (LLaVA's own greedy captions), with a text-only control and frozen splits,
  budgeting every step from measured cost.
- **HOW.** COCO val2014: 100 fit / 30 held-out images (`seed=0`, disjoint by construction; a 10-question bank
  cycles so each question has 10 samples). WikiText-103 text control: 100 fit / 30 held-out prompts. Splits
  frozen on disk (`step0/corpus-split.json`, `step0/text-corpus.json`); fits offline. Cost measured at
  `dim_batch=8`, bf16, all 31 layers, 3 masks.
- **RESULT.** Image sample (660 tokens = 82 text + 576 image): **288.2 s/sample, peak 36.4 GiB**; text sample
  (190 tokens): **85.9 s/sample, peak 19.9 GiB** — replacing the 355 s projection, keeping S1 at all 100 prompts
  (D5).
- **INTERPRETATION.** The image block is 88% of tokens and ~3.4× the cost; mask budget, `dim_batch`, and fit
  shards all trace back to this measured geometry.

### X3 census and mask budget (figA3)

- **WHY.** Massive-activation outliers and sink positions decide what a fit should skip: are the leading
  positions sink-like, as in text models that use `skip_first=16`?
- **HOW.** Residual-norm census over 50 real COCO fit samples at L0/L16/L24/L31; groups = BOS, positions 1–16,
  text before/after the image block, first 16 patches, patches 17–end (`step1/x3_norms.json`).
- **RESULT.** BOS is the only massive-activation outlier: mean ‖h‖ 8.3 at L0 → **1568.8 at L16 (188×)**, ~42×
  the text-post norm, decaying to 633.7 by L31. Its dominant dims are disjoint from every other group (Jaccard
  0.111) and positions 1–16 track the patch/text groups — `pos_1_16_sink_like=false`. Mask positions: text
  **3604**, image **28800**, all **32454** (image 88.7%).
- **INTERPRETATION.** `skip_first=1` stands for multimodal fits (16 for the text control); the conditional X4
  sweep was never triggered (D6). Image positions dominate the fit numerically, making the target-mask question
  (X1) structural.

![figA3](figures/figA3_census_masks.png)
_figA3 — position census over 50 COCO samples: BOS is the sole massive-activation outlier (8.3 → 1568.8, 188×,
then 633.7 at L31); image patches are 88.7% of the 32,454 positions (mask counts text 3604 / image 28800 / all 32454) (`step1/x3_norms.json`)._

### X6 dtype gate (figA1)

- **WHY.** bf16 gradients would make fitting ~13% cheaper — if numerically adequate; a pre-registered rule
  decides before any production fit.
- **HOW.** Same 8 COCO samples, layers 0/8/16/24/30, both legs differing only in `--dtype`: bf16 vs fp32+TF32
  (true fp32 measured ~15× slower un-flagged, D10). Comparator: relative Frobenius, cosine, best-fit scale,
  residual after scale removal. Rule: fp32 if median > 3% or max > 10%.
- **RESULT.** Per-mask medians agree: **1.20% (all), 1.26% (image), 1.48% (text)**. But text/L0 hits **49.2%**
  (cosine 0.88 there; `rel_after_scale` 48.1% — not a scale artifact); all/L0 20.1%, image/L0 4.35%. Rule fires:
  `use_fp32=true`, at 2486.5 s vs 2205.8 s for the same 8 samples = **1.13×**.
- **INTERPRETATION.** Production fits run fp32+TF32; S1's finished bf16 lens stays as a recorded deviation (D18)
  — its gate claim is a ranking claim a 1.5% perturbation cannot flip. The one catastrophic cell is where means
  are uncentered (L0 text), foreshadowing P1.

![figA1](figures/figA1_dtype_precision.png)
_figA1 — bf16 vs fp32+TF32 Jacobian agreement: per-mask medians 1.20/1.26/1.48% by L16, but text/L0 at 49.2%
(cosine 0.88, not a scale artifact) fires the pre-registered rule → production = fp32+TF32 at 1.13× cost
(`step1/x6_dtype.json`)._

---

## Stage 1 — The instrument (fit + validate + first causal handle)

### S1 text-control fit and gate (figA2)

- **WHY.** Validate the fit→score pipeline on a text-only corpus against pre-registered checks, including the
  strict "never worse than the untrained logit lens in the middle layers" criterion.
- **HOW.** WikiText-103, `skip_first=16`, mask `all`, 31 layers, bf16, 100 prompts; scored on the frozen
  30-prompt held-out shard (`step2/s1_score.json`): identity check (J=I must reproduce the logit lens exactly),
  depth trend, last-layer alignment, unigram frequency control, mid-layer comparison.
- **RESULT.** Identity exact **0.0**; depth trend 885.54 (L16) → **27.31** (L31) mean true-token rank;
  last-layer rank difference **0.0**; frequency control equal to the model at L31 (top-1-in-top-50 0.450, mean
  rank 27.3). The J-lens wins mean rank at L2–16 (L14: 1101.2 vs 5178.4) and L26–30 (L30: 46.5 vs 75.2) — but
  the **strict mid-layer criterion FAILS**: at L17/20/21/23/24 it loses to the logit lens by +379, +813, +331,
  +99, +271 ranks (chance 16032.5; mid-band means 2228.8 vs 2119.3). By KL/top-1 the untrained lens leads L17–30
  (L30 top-1 0.762 vs 0.718): the rank-vs-KL tension.
- **INTERPRETATION.** Conditional pass, the campaign's honest limitation (D19): a comparative nuance, not a
  validity failure. P1 later shows the mid-layer deficit was substantially the uncorrected mean shift — logged
  before the fix, not rounded away.

![figA2](figures/figA2_depth_fidelity.png)
_figA2 — S1 depth fidelity on held-out WikiText: the J-lens beats the untrained logit lens on mean rank at every
layer except the L17–24 bump (2228.8 vs 2119.3), exact at L31 (rank 27.31 = model, KL 0.0, top-1 1.00)
(`step2/s1_score.json`)._

### S2 caption lens

- **WHY.** Does the lens survive on multimodal, on-policy data with held-out scoring and cross-half controls?
- **HOW.** The 100 fit images split by instruction half (A = questions 1–5, B = 6–10; 50 image-disjoint samples
  each); two half-fits then `--merge` (weights by `n_prompts`, so the merge is bit-equivalent to a single
  100-sample run, D7); scored on **300 held-out captions** (`step3/s2_eval.json`); halves cross-scored per
  question-half (no image leakage); cross-corpus transfer both ways.
- **RESULT.** Text-tag rank: L0 26120.6 → L16 **1307.6** → L30 **215.2** → L31 = model (42.62; identity exact,
  agree 1.000, KL 0.000). L16 sits at 30.7× the model ceiling vs S1's 32.4× — the caption lens tracks the text
  lens. Halves are symmetric: lens_a own 233.00 vs cross 231.83; lens_b own 197.81 vs cross 203.15 (correct
  in-domain sign, no leakage; lens_b ~15% stronger — question-half distribution). An L20 bump (2753, +1446 over
  L16) echoed S1's mid-band weakness.
- **INTERPRETATION.** X11 attributed the bump to the **fitted map, not the model** (D37, revising D35/D36): the
  untrained logit lens is monotone through L17–24 (2201.8 at L16 → 1557.9 at L20 → 284.8 at L30) while both
  fitted lenses bump — A2's single-linear-map approximation degrades mid-band, Stage 2's target. Transfer keeps
  the rank-vs-KL tension honest: S1 wins L30 rank on captions (109.49 vs 215.21), S2 wins agree/KL (0.729/0.428
  vs 0.706/0.474).
  ![figD9](figures/figD9_s2_halves_transfer.png)
  _figD9 — S2 caption lens (`step3/s2_eval.json`): merged 100-sample lens text-tag rank 26120.6 → 1307.6 (L16) →
  215.2 (L30) → 42.62 at L31 (identity exact); instruction halves symmetric (lens_a own 233.00 vs cross 231.83;
  lens_b 197.81 vs 203.15 — merge ≡ single fit, D7/D35); cross-corpus transfer: the caption lens reads WikiText
  at 2.49× ceiling (L30 67.9 vs 27.28), the text lens reads captions at 109.49 rank — and the L20 bump appears in
  both transfers, the fitted-map signature X11 pinned on the map, not the model (D37)._

### X1 target-mask separation (V1)

- **WHY.** Register V1: text sources after the image block cannot causally reach image targets, so the cotangent
  seeded at image targets must live in different rows than the next-text cotangent — otherwise image-position
  readouts would be next-text predictions.
- **HOW.** The same 20-image shard fitted twice: `target_mask=all` vs `target_mask=text` (cotangent seeded only
  at requested positions; D2). Hard criterion: text rows `torch.equal`-identical between variants, image rows
  different; a lens-level table quantifies image-row movement (`step4/x1_targetmask.json`).
- **RESULT.** The bit-identity criterion failed as stated — the variants are separate fit invocations, so
  full-variant text rows differ only at the TF32/atomics run-to-run noise floor (rel_fro 1.37e-2 at L0 → 1.75e-6
  at L30, the X6-era floor); the unit test on the causally-disjoint subset passes (D3/D36), and the float64
  compare re-run was deferred after three OOMs, its analytic justification standing (D38). The real signal: with
  the image block before the text, **image rows move by rel_fro 24.5 (L0) → 27.4 (L8) → 38.1 (L16) → 103 (L24) →
  255 (L30) while cosine collapses 0.39 → 0.09** — a ~180–1500× separation from text-row movement. Lens-level:
  image J moves 23.7 → 115.6 with best-fit scale 8–28× (the image→image cotangent mass is 8–28× larger,
  near-orthogonal); aggregate text rows differ 2.12 (L0) → 0.17 (L30), explained by `USER:` tokens preceding the
  image block, which do reach placeholder targets (pre-registered, D3).
- **INTERPRETATION.** **V1 confirmed**: image rows are not next-text predictors; a readout at an image position
  is a first-order disposition to verbalize, not a prediction — binding Stages 3–4.

![figD1](figures/figD1_targetmask_conditioning.png)
_figD1 — target-mask separation and swap-basis conditioning (`step4/x1_targetmask.json`, `x7_x9.json`): image
rows move rel_fro 24.5 → 255 (L0 → L30) while cosine collapses 0.39 → 0.09; text rows differ only at the TF32
noise floor (1.37e-2 → 1.75e-6; the `torch.equal` criterion failed by construction, D36) — V1 confirmed. Right:
all 10 COCO pair bases are well-conditioned (cond 1.30–2.89 at L8/16/24, ~345× below the 1e3 exclusion line)._

### X7 conditioning and X9 edit sweep (figB1)

- **WHY.** Are the lens's directions (rows of `W_U J_l`) a live causal handle: well-conditioned swap bases, and
  generations that actually change?
- **HOW.** X7: condition number of the 2-column swap basis `[v_s, v_t]` for 10 COCO concept pairs at L8/16/24
  (exclude at cond > 1e3). X9: greedy generations on held-out captions with add/ablate/swap at L16/24 over an
  alpha grid; x9b extends to 2800 generations (D40); directionality mining on the 215 changed generations (D39).
- **RESULT.** X7: 31 entries, **cond ∈ [1.30, 2.89]** — all far below the exclusion threshold; cosine 0.25–0.78,
  norms 0.83–1.35, `degenerate=false` everywhere. X9: **215/1400 generations changed** (15.4%; add 161, ablate
  32, swap 22); edits bite early (first-diff tokens cluster at caption positions 10–14 and 1–3; add@L16 91 vs
  add@L24 70). Dose-response is monotone (figB1): ablate 7/16/24/34% and swap 7/12/24/39% at alpha 0.5/1/2/4;
  swap@L24 reaches 35% at alpha 4. But **direction is rare**: 7 of 215 changed generations (3.3%) are
  concept-directed, needing alpha ≈ 5–14× the residual norm (e.g. `man→woman add@L24 α=13.81` → "The woman is
  the main focus"); at alpha ≈ 1–3×, 85% of edits leave the greedy path unchanged.
- **INTERPRETATION.** The E5 minimal bar passes; the strong bar (cheap directed control) fails at this scaling —
  a J-vector edit moves the state mostly along directions with weak logit effect. P5's span-overlap measurement
  supplies the mechanism; H1/H2 propose the fixes.

![figB1](figures/figB1_dose_response.png)
_figB1 — edit dose-response and directedness (X9 + x9b, 2800 generations): caption change is cheap and monotone
in alpha (ablate 7/16/24/34%, swap 7/12/24/39% at α = 0.5/1/2/4), but concept-directed change is rare (7/215 =
3.3%) and needs α ≈ 5–14× the residual norm; α ≥ 27 degenerates into repetition collapse, not injection
(D39/D40)._

---

## Stage 2 — Improving the lens (the model-class ladder)

### Metric redesign

- **WHY.** Mean rank over a 32k vocabulary is tail-dominated; if the lens is good for the typical position,
  means hide it.
- **HOW.** `lens_zoo_eval.py` scores mean/median rank, top-1/5/10/50 agreement, mean KL and a composite LQS on
  the 300-sample held-out set; cross-run determinism checked against historical scores.
- **RESULT.** Medians unmask the lens: L16 mean 1307.6 but **median 15**; L24 mean 1047.5, median 4; L30 mean
  215.2, **median 1**, top-10 agreement 0.886. The re-scored merged lens reproduces the historical ranks exactly
  (L30 215.21).
- **INTERPRETATION.** For the typical position the lens nearly solves next-token prediction from L16 on; later
  model-class comparisons use this multi-metric view.

### P1 affine correction from a moment census

- **WHY.** X6's L0 cell and S1/S2's mid-band bump both look like an uncorrected mean shift between intermediate
  and final residual states, not a variance limit of the linear map.
- **HOW.** `moment_census.py` measures per-layer `E[h]` and `E[h_final]` on 100 corpus samples (text mask, 7144
  positions); the readout applies `h_final ≈ b_l + s_l (J_l h_l)` with `b_l = E[h_f] − J_l E[h_l]`, `s_l = <t,
h_f>/||t||²`; a clamp variant clips `s_l` to [0.5, 2.0] (the raw census scale is noisy early: `s_0 = −3.19`).
- **RESULT.** On the 300-sample held-out (tag=text): L20 rank 2753.0 → **811.3**; L30 215.2 → 76.9; the bump
  ratio (L20/L16) collapses **2.11 → 0.68**; L30 ratio 5.05 → 1.80. The clamp fixes the early blowup: KL at L0–4
  **21.9 → 7.0** (better than the unbiased 9.6–15.6), rank ratio 706 → 46; LQS climbs **−0.102 → +0.335
  (unbiased) → +0.766 (clamped)**; layers 5–30 unchanged; one mild KL regression at L9–16 (+0.4–0.7).
- **INTERPRETATION.** The D19 limitation was substantially the uncorrected mean shift: with bias+clamp, L17–24
  becomes the lens's _best_ zone (KL 2.6–4.0 vs unbiased 5.0–6.6; L20 rank ratio 19.0 vs 64.6), superseding
  D39(c). What survives is A2's true variance limit — the translator class's job.

![figD2](figures/figD2_affine_transport.png)
_figD2 — the affine correction and the transport decomposition (`step4/lens_zoo{,_biased,_biased_clamped}.json`,
`step5/lens_zoo_{ident,cal}.json`): bias+scale collapses the mid-band bump (L20 rank 2753.0 → 811.3; ratio
2.11 → 0.68) and the clamp fixes the L0–4 blowup (KL 21.9 → 7.0; LQS −0.102 → +0.335 → +0.766); transport is
worthless at L0–7, peaks across L17–28, and delivers −0.508 at L30 (rank 127.7 → 76.9) — affine does the early
work, transport the late work (D40/D41)._

### P5 model-class ladder (figA4)

- **WHY.** Where does the J's value live — transport (`J_l` vs identity), affine correction, or calibration? Can
  cheaper classes (diagonal, shifted, rank-truncated, distilled translators) do the same job?
- **HOW.** Complete ladder on the 300-sample held-out (tag=text), zero new Jacobian fits: J-structure analysis
  (`jacobian_structure.py`), transport decomposition (identity lens with the same payload), span overlap against
  `W_U`'s readout span, calibration ablations.
- **RESULT.** LQS ladder: untrained logit lens 0.000; scaled identity 0.000 (exact); rank64/ rank256 truncated J
  −0.587/−0.284; full J uncorrected −0.102; **diag(J) +0.374**; shiftP4 +0.524; ident+bias +0.647; **J + census
  bias+scale +0.766**; **+logit_bias +0.922** (a win at every layer, 2–23%, after fixing a wrong-sign first
  attempt that cost −0.100); **lowrankJ (rank-32 KL translator on `J_l h`) 1.714 ≈ lowrankRaw (pure tuned lens)
  1.718**. Transport decomposition: identity with the same payload scores +0.647 vs J's +0.766 — transport adds
  +0.119 (16% of the fix); worthless at L0–7, growing through L8–16, peaking **−0.508 at L30 (127.7 → 76.9
  rank)**: affine does the early work, transport the late work. Structure: `trace(J)/d` crosses 1 at L20–23
  (0.05 at L0, 1.04 peak at L25); top-64 singular directions carry 87.5% of the energy at L0 but 9.9% at L30;
  diag carries 0.04–5.3% of the Frobenius mass. **Rank compression loses** — truncating J strips the
  near-identity mass (its ~4096 equal singular values), leaving exactly the uncorrected off-diagonal; the J's
  value is not low-rank. Span overlap: the J's top-64 directions carry only **3.5–10.6% energy in `W_U`'s
  readout span** (random baseline 3.1%) — at chance early, **3.4× chance at L30** — mechanistically explaining
  why J-vector edits need ~10–30× residual-level magnitude to move logits (D39's alpha 5–15×, x9b's α≥27
  breakdown). Cross-layer span overlap vs the final layer is near chance (0.0002–0.006) until L16, then
  0.018–0.050 at L20–26 (75–200× chance — the tuned lens's covariance drift, quantified).
- **INTERPRETATION.** The ladder is complete: the J-lens's unique value over the tuned lens is late-layer
  transport; the translator's value is mid-layer; the affine correction (bias+scale+logit_bias) is necessary
  everywhere. Best J readout at this point: +0.922 (L30 ratio 1.79) — until the tmtext refit and the scaling
  revision below.

![figA4](figures/figA4_model_class_ladder.png)
_figA4 — the model-class ladder (LQS on 300 held-out captions): diag(J) +0.374, shiftP4 +0.524, ident+bias
+0.647, J+bias+scale +0.766, +logit_bias +0.922, tmtext50+payload +0.947; rank-32 translators lead overall
(1.714/1.718) but lose 3× at L30 (5.62–5.82 vs 1.30–1.79) (DECISIONS.md D41/D42)._

![figD3](figures/figD3_jacobian_structure.png)
_figD3 — the average-Jacobian becomes identity-like (`step5/jacobian_structure.json`): α_ls = trace(J)/d crosses
1 across L20–23 (0.046 → 1.043); the early J is a few dominant off-diagonal directions (top-64 energy 87.5% at
L0 → 9.9% at L30; symmetry 1.41 → 0.51; diag mass 0.04 → 5.3%) — which is why rank compression loses: truncation
strips exactly the near-identity mass that carries the late-layer value (D41)._

![figD4](figures/figD4_span_overlap.png)
_figD4 — span overlap explains the edit thresholds (`step5/span_overlap.json`): the J's top-64 right-singular
directions carry only 3.5–10.6% energy in `W_U`'s readout span (chance 3.125%), so a residual-level edit needs
~10–30× magnitude to move logits (D39's α 5–14×; x9b's α ≈ 27 → 82% change); cross-layer overlap vs the final
layer stays near chance (2.4e-4) until L16, then 0.018–0.050 across L20–26 — 75–200× chance, the tuned lens's
covariance drift quantified (D41)._

### P3 alpha sweep (figB1, above)

- **WHY.** Map the sane operating window: where does "change the caption" stop being "direct the caption"?
- **HOW.** x9b: 2800 greedy generations, add/ablate/swap at L16/L24; directedness flags validated against an
  independent word-boundary recomputation.
- **RESULT.** Change is monotone in alpha (figB1). The add grid ran at alpha ≈ 27–1035 (its "k=1" label is
  really α≈27 at L16, 58 at L24) — combined with D39's add grid (α 1.34–14.68, 10–80% change) the full curve is
  monotone. Within the sane window (α ≤ 4) directedness is **0.5–3%** (ablate 1–2/200, swap 0–6/200). At **α ≥
  27** the "directed" rows are degeneration: `dog→cat add@L16` yields "cat cat cat cat ...", a repetition
  collapse — breakdown, not injection.
- **INTERPRETATION.** Change is cheap, direction is rare (D39's 7/215 remains the reference), large-alpha
  effects are model breakdown; protocols must report residual-relative alpha, never the k label (D40).

### target_mask=text refit (D42)

- **WHY.** D40's n=20 probe flagged `target_mask=text` (excluding the image block's next-image-token targets) as
  the best data lever; does it hold at n=50, and does it compose with the calibration payload?
- **HOW.** 23.7-h refit: 50 samples, all layers, fp32+TF32, `target_mask=text`; scored bare and composed with
  the best payload (census bias+scale + logit_bias) on the 300-sample held-out
  (`step4/lens_zoo_tmtext{,_cal}.json`).
- **RESULT.** Bare: LQS **+0.337**, L30 ratio 2.667 — **+0.44 LQS over the all-target merged lens**, stable
  across sample counts. Composed: **LQS +0.947, L30 ratio 1.30** — the best late-layer readout measured, J or
  translator (rank-32 translators sit at L30 5.62–5.82). Mid-layer stays behind (L16 37.7, L20 30.8 vs
  merged+payload 21.4/15.3).
- **INTERPRETATION.** Image-target exclusion sharpens exactly the late image→text handoff layers and costs
  mid-layer sharpness — the mask choice is a dial between regimes, and the deployment recipe splits by regime.

### Data scaling (figA5)

- **WHY.** The J-class looked flat from n=20 to n=100 (D40): is that sample-count saturation or the model class?
  And does the winning translator class keep scaling?
- **HOW.** Two scaling curves on disjoint nested corpora, scored on the 300-sample held-out (tag=text). J-class:
  an online probe (`fit_masked --probe-every`) scores the running mean-J on 16 held-out samples every 5 fit
  samples. Translator class: rank-32 KL distillation to the model's own logits from a CPU-resident activation
  cache, at n = 40/100/500/1000.
- **RESULT.** J-class saturates at ~100: intra-fit text-KL 6.305 (n=5) → 6.043 (10) → 5.989 (15), decelerating;
  image-position KL flat throughout (11.51/11.58/11.53 — V1's echo); its value is the moment census, converging
  ~1/sqrt(n) over 7k positions by n≈100. The translator class does **not** saturate: fit-side val_kl 1.424 (40)
  → 0.888 (1000), but held-out LQS keeps rising ~log-linearly: **1.714 (40) → 2.043 (100) → 2.691 (500) → 3.009
  (1000)** (~0.4 LQS per e-fold; rung deltas +0.329/+0.648/+0.318); fit-KL decouples from the rank metric (the
  D19 tension at scale). Per layer at n=1000 the translator reaches L30 ratio **1.30** — parity with the best
  J-lens (1.30), from 5.82 at n=40 (3.90/2.05 in between).
- **INTERPRETATION.** **REVISION (supersedes D41/D42): the late-layer transport advantage was a small-data
  artifact** — the translator's late-layer weakness decays with data exactly to the J-lens's level. Production
  recommendation: rank-32 KL translator at n ≥ 1000 for readouts (keep scaling; ~1–2 h per 1000 samples with the
  CPU cache); the J-lens (zero-fit, ~1 h total with the moment census) as the cheap fallback (LQS 0.92–0.95);
  the J keeps the causal-edit handle.

![figA5](figures/figA5_data_scaling.png)
_figA5 — data scaling: the J-class is saturated by n≈100 (LQS 0.92–0.95) while the rank-32 translator's held-out
LQS rises ~0.4 per e-fold of data (1.714@40 → 2.043@100 → 2.691@500 → 3.009@1000), reaching J-lens parity at L30
(1.30) — the late-layer transport advantage was a small-data artifact (DECISIONS.md D43)._

---

## Stage 3 — Hallucination mechanisms (the causal story)

### X12 commitment boundary (figB3)

- **WHY.** Until which layer is visual content causally live at image positions — the window in which a
  hallucination could be prevented rather than cured?
- **HOW.** Cross-image patching: 8 disjoint-object COCO pairs; replace A's 576 image-token rows with B's at a
  single layer; measure caption change, A-object vanishing, B-object appearing; single-position
  (last-image-token) and text-position controls (`step10/x12_cross_image_patch.json`).
- **RESULT.** All-576-patch changes flip the caption at rate **1.00 / 1.00 / 1.00 / 1.00 / 0.875 / 0.125 / 0.50
  / 0.25 across L0/8/12/16/20/24/28/30**; A-objects vanish 1.00 → 0 by L24; B-objects appear 0.406–0.490 only
  through L16, then ~0. Single-token patches do nothing (**≤ 0.25**); text-position controls sit at the
  0.12–0.37 floor.
- **INTERPRETATION.** The commitment boundary sits at **L16–20**: re-writing image tokens after L20–24 no longer
  changes the caption, and the content is distributed — no single handoff token stores it. Prevention must
  happen mid-stack; cure at the handoff layers.

![figB3](figures/figB3_commitment_boundary.png)
_figB3 — cross-image patching (8 pairs): patching all 576 image tokens changes the caption at rate 1.00 → 0.125
across L0–24 (A-objects vanish by L24; B-objects appear only through L16); single-token patches do nothing (≤
0.25) — visual content is causally live at image positions only until ~L16–20
(`raw/step10/x12_cross_image_patch.json`)._

### X14 prior-vs-grounding race and the text-mediated handoff (figB4)

- **WHY.** Is a hallucinated word suppressed knowledge ("knows but commits anyway") or the prior's default that
  grounding must overtake — and where does image content reach the verbal stream?
- **HOW.** 24 images, calibrated logit lens (identity transport + moment payload — zero fitting beyond moments);
  lens-rank trajectories of hallucinated vs grounded generated words, of the image's true categories at the last
  image-token vs the last prompt position, and of truth at the hallucinated words' decoding positions
  (`step10/x14_workspace_probe.json`).
- **RESULT.** The prior leads from the start: hallucinated generated words are high-ranked **from L0 (rank ~787
  vs grounded ~5358 — 7×)**, while grounded words are built up monotonically (5358 → 1.18). The trajectories
  **cross at L24–26** (L24: grounded 4.86 vs hallucinated 6.5; L26: 1.18 vs 4.5). The handoff is text-mediated:
  the image's true categories rank ~1200–2000 at the last IMAGE-token position at every layer (never decodable
  there), but at the **last PROMPT position** the best-per-image true object rank falls **1423 (L16) → 34 (L20)
  → 6–15 (L22–25)**; then the L31 readout flips to the imminent token (mean rank 3464 — the motor regime).
  DOES-THE-MODEL-KNOW is **negative**: at the hallucinated words' decoding positions the image's actual
  categories rank ~16–35 through L20–30, vs the hallucinated word's 2–6. Separability at L30: grounded 1.18 vs
  hallucinated 6.0.
- **INTERPRETATION.** Hallucination is the prior winning, not suppressed knowledge: the model does not represent
  the truth competitively at the commitment positions. Grounding is a computation that overtakes the prior in a
  narrow band (L20–25) at a specific location (last prompt position) — exactly the X12 boundary and the D41–D43
  transport zone.

![figB4](figures/figB4_prior_grounding_race.png)
_figB4 — the prior-vs-grounding race (24 images): hallucinated words rank ~787 vs grounded ~5358 at L0;
grounding overtakes at L24–26; the image's true objects surface only at the last prompt position (1423 → 34 →
6–15 across L16→L20→L22–25) and the L31 motor flip moves the readout to the imminent token (mean rank 3464)
(`raw/step10/x14_workspace_probe.json`)._

### X13 cure test (figB2)

- **WHY.** D39's open item: does ablating the hallucinated concept's J-lens direction remove the hallucination —
  and is the ablation surgical?
- **HOW.** 39 images, 10 hallucination arms vs COCO-instances ground truth; mean-replacement ablation of the
  hallucinated category's J-lens-vector direction at alpha=1, at L16/24/28/30; control = ablating a GROUNDED
  category's direction (36 control arms) (`step10/x13_cure_test.json`).
- **RESULT.** Hallucinations removed **0.50 / 0.60 / 0.70 / 0.90 at L16/24/28/30** with grounded-category
  retention **1.00 / 1.00 / 1.00 / 1.00** on the same arms. The control is destructive: grounded retention falls
  **0.97 → 0.79 → 0.52 → 0.35** across the same layers.
- **INTERPRETATION.** A training-free, single-direction **90% surgical cure at L30**: the hallucinated concept's
  direction is causally load-bearing for the unsupported token while grounded content survives it. The layer
  dose-response matches the X14 crossing — the cure improves as grounding overtakes the prior.

![figB2](figures/figB2_cure_test.png)
_figB2 — the cure test (39 images): mean-replacement ablation of the hallucinated concept's J-lens direction
(α=1) removes 0.50/0.60/0.70/0.90 of hallucinations at L16/24/28/30 with grounded retention 1.00, while ablating
a GROUNDED category's direction destroys retention 0.97 → 0.35 — a training-free 90% surgical cure at L30
(`raw/step10/x13_cure_test.json`)._

### Scale-ups at power (D45/D46)

- **WHY.** D44's round left held-out predictions; the scale-ups test them at adequate n: pre-generation
  detection, sink clamping, the attention mirror, hesitation, and image-level fusion.
- **HOW / RESULT.** One sequential chain, all training-free, single-forward unless noted (`step11/*.json`):
- **X18 POPE pre-generation gap**: 49 images, 185 questions (87 present / 98 absent); model POPE accuracy 94.1%
  (present 96.6 / absent 91.8), 8 hallucinated yes. One forward at the last prompt position + the calibrated
  lens: **AUROC(present vs absent) = 0.985 at L28** (0.982–0.985 across L20–30); mean gap +5.28 (present) vs
  −4.97 (absent). The lens disagrees with the model exactly on its wrong affirmations: **0.895 as a low-gap
  detector of the model's own hallucinated yes**.
- **X16 sink clamp**: 20 images, top-16 dims per (layer × position group), zero-clamp at L16/24/28/30. The **BOS
  sink is load-bearing**: clamping it collapses grounded retention (0.278 at L16 BOS; 0.222 all-groups) and
  explodes the handoff true-object rank (+1433 at L16, +550 at L28). **Image-token massive dims are nearly
  inert** (retention 0.94–1.00; rank deltas −43..+330). The BOS dim set is **identical across L16–30**
  (`[2533, 1415, 1076, 1512, 2789, 2298, 2158, 3431, 339, 3209, 2469, 257, 4071, 2350, 788, 2235]`);
  image/text groups share 1512/2789/1076 and drop the BOS-specific 2533/1415 — the sink is position-typed.
- **X17 attention mirror**: CLS→patch vs text→image Spearman. At n=200 the curve **replicates**: rho 0.105 (L0)
  → 0.341 (L4) → 0.463 (L12) → 0.577 (L16) → **0.645 (L20)** → 0.403 (L24) → 0.199 (L30) — the LLM's access to
  the image first mirrors the vision tower, then the verbal workspace takes over. But the **hallucination join
  is NULL at power** (27 hallucinating / 170 clean): share_mid AUROC **0.501**, r_pb −0.10 on a 0.4–0.57 mean
  — attention allocation does not predict hallucination.
- **X15 hesitation**: at n=200 (385 grounded / 39 hallucinated word-completion steps), hallucinated steps show
  entropy **0.963 vs 0.620**, margin 4.53 vs 5.55, logprob −0.384 vs −0.230. AUROC (hallucinated positive):
  model entropy 0.593, margin 0.426; **calibrated-lens entropy beats the model's own (L7: 0.730**; L4–7
  0.63–0.73; near chance L11+). Direction confirmed at power; as a per-word detector it is modest.
- **X19 image-level fusion**: 197 images (27 hallucinating): mean per-step lens entropy AUROC 0.620 (L28);
  pre-generation true-category gap 0.632 (L30); **fused z(mean entropy) + z(gap) 0.666 (L30)** — the fusion
  beats its components.
- **INTERPRETATION.** The pre-generation object-presence question is nearly solved by a single forward;
  attention is exonerated; per-word uncertainty is real but weak; the components of image-level risk are
  complementary (H8).

![figD5](figures/figD5_pope_handoff.png)
_figD5 — the pre-generation POPE handoff (`step11/x18_pope_handoff.json`): the calibrated lens's z[yes]−z[no]
gap at the last prompt position separates present (mean +5.28, n=87) from absent (−4.97, n=98) questions at
AUROC 0.985 (L28), and the model's 8 hallucinated yes-answers sit low-gap (halluc-positive AUROC 0.106 → 0.894
as a low-gap detector) — a training-free, single-forward truth readout (D45/D46)._

![figD6](figures/figD6_uncertainty_signals.png)
_figD6 — decoding-time uncertainty (`step11/x15_hesitation_n200.json`): hallucinated word-completion steps are
measurably more hesitant (entropy 0.963 vs 0.620, margin 4.53 vs 5.55, logprob −0.384 vs −0.230; n = 39/385),
and the calibrated lens's entropy is the better signal — AUROC 0.730 at L7 vs the model's own 0.593 — with
signal concentrated at L4–7 (D46)._

![figD7](figures/figD7_mirror_and_sink.png)
_figD7 — attention mirrors, sinks anchor (`step11/x17_attention_mirror_n200.json`, `x16_sink_clamp.json`): the
CLS→patch ↔ text→image Spearman mirror replicates at 10× n (0.105 → 0.645 at L20 → 0.199 at L30: vision tower
early, verbal workspace late) but does not predict hallucination (join AUROC 0.501 — clean null). The BOS
massive-activation sink is load-bearing (clamping collapses grounded retention to 0.278 at L16 and explodes the
handoff rank +1433) while image-token massive dims are inert (0.94–1.00); the BOS clamp set is stable across
L16–30 (14/16 dims shared, recomputed from the JSON; D46 records the set)._

### The detector ledger (figB5)

- **WHY.** One comparable table for every training-free hallucination signal measured in Stages 3–4. All signals
  are moments-only, single pass unless noted; ns from D45/D46.

| Signal (training-free)                            | AUROC               | n                                                              |
| ------------------------------------------------- | ------------------- | -------------------------------------------------------------- |
| Pre-generation yes/no gap, per question (L28)     | **0.985**           | 185 questions (0.895 catches the model's own hallucinated yes) |
| Pre-generation true-category gap, per image (L30) | 0.632               | 197 images                                                     |
| Per-step lens entropy (L7), per word              | **0.730**           | 39 hallucinated / 385 grounded steps                           |
| Per-step model entropy, per word                  | 0.593               | same 39/385                                                    |
| Image mean lens entropy, per image (L28)          | 0.620               | 197 images                                                     |
| Fused mean entropy + category gap (L30)           | 0.666               | 197 images                                                     |
| Cross-modal attention share / mirror rho (join)   | 0.501 / 0.41 (NULL) | 197 images                                                     |

![figB5](figures/figB5_detector_ledger.png)
_figB5 — the detector ledger: workspace CONTENT predicts hallucination (pre-generation gap 0.985; per-word lens
entropy 0.730; fused image-level 0.666), attention allocation does not (0.501 NULL); caption-level risk is a
0.6–0.73 problem, pre-generation yes/no is nearly solved (`raw/step11/` JSONs, D46)._

---

## Stage 4 — Global-workspace replication + the VLM novelty

### Paper frame and band mapping

- **WHY.** Gurnee et al. 2026, "Verbalizable Representations Form a Global Workspace in Language Models"
  (transformer-circuits.pub/2026/workspace) claims a mid-layer verbalizable band with sensory/workspace/motor
  trichotomy in frontier text models, with only an anecdotal multimodal check. Does the structure replicate on a
  7B VLM, and what is VLM-specific?
- **HOW.** The upstream repo vendors the paper's experiment data
  (`third_party/jacobian-lens/data/{experiments,evaluations}`); ten scripts (`code/x20`–`x23`) rerun the paper's
  arms on LLaVA-7B plus VLM-only arms (`step12/`, 11 JSONs). The paper's 0–100 band **[38, 92]** maps to
  LLaVA-7B's 32 layers as ~**[12, 29]**.

### Band structure replication (figC1)

- **WHY / HOW.** x22 structure arm (200 prompts): four band metrics — top-1 next-token accuracy, readout
  kurtosis, top-1 autocorrelation, effective dimensionality — plus the multihop depth-precedence and flexgen
  arms on the vendored items.
- **RESULT.** Onsets: autocorrelation **L12** (argmax 24), effective dimensionality **L17** (0.327 → 0.702 in
  band), top-1 accuracy **L20** (argmax 29), kurtosis **L28** (max 3.72) — the four metrics **bracket a band
  ~L17–29** inside the mapped paper band. Depth precedence **replicates almost exactly**: intermediate-swap vs
  answer-swap effective-onset gap **−0.176** vs the paper's ~−0.17 (intermediates are computed and causally used
  before answers); multihop probe pass@1 0.269. Absolute swap success is low (late-band 9.3% vs the paper's
  54–70%), and flexible generalization via J-lens vectors is weak: flexgen **5% at alpha=1 and alpha=2** vs the
  paper's 40/53% (loading-success Pearson +0.256, right sign).
- **INTERPRETATION.** The sensory/workspace/motor trichotomy is present at 7B; causal leverage on it is not
  frontier-grade.

![figC1](figures/figC1_band_structure.png)
_figC1 — band structure on LLaVA-7B: four onset metrics (autocorr L12, effdim L17, top-1 L20, kurtosis L28)
bracket a workspace band L17–29 inside the mapped paper band [12, 29]; depth precedence replicates (−0.176 vs
paper −0.17); flexgen is weak (5% vs 40/53%) (`raw/step12/x22*reasoning_structure*_.json`).\*

### Lens beats logit lens at mid-band (figC2)

- **WHY.** The paper's methodological claim: the J-lens reads workspace intermediates better than the logit lens
  in the band. Does it transfer to the VLM?
- **HOW.** x20: six vendored eval suites, J-lens vs calibrated logit lens, pass@10 per layer.
- **RESULT.** The J-lens wins **5/6 suites**: multihop **0.312 vs 0.054 pass@10 at L20** (best 0.484 at L28 vs
  0.366 at L29); multilingual 0.265 vs 0.058 at L20 (best 0.280 vs 0.235); typo 0.125 vs 0.010 at L20 (best
  0.125 vs 0.031); order-ops 0.436 vs 0.327; poetry 0.041 vs 0.020. The **association suite is NULL on both
  variants** (pass@10 = 0 at every layer; gold mean rank ~20,660 of ~32k at L20) — that suite's associations are
  not lens-readable on this model.
- **INTERPRETATION.** The J-lens advantage at workspace mid is architecture-robust (text and multimodal) but
  suite-dependent; the NULL row bounds the claim.

![figC2](figures/figC2_lens_vs_logit_suites.png)
_figC2 — J-lens vs calibrated logit lens on six vendored suites (pass@10 per layer): the J-lens wins 5/6
(multihop 0.312 vs 0.054 at L20; multilingual 0.265 vs 0.058; typo 0.125 vs 0.010), while the association suite
is NULL on both variants (gold rank ~20.7k of 32k) (`raw/step12/x20_lens_evals.json`)._

### Verbal report and introspection (figC3)

- **WHY.** The paper's strongest subjective reports: lens order tracks output order, injected "thoughts" are
  reportable from user-turn (not assistant) positions, directed modulation shows the white-bear effect. Which
  survive at 7B?
- **HOW.** x21 over 14 vendored categories: pooled Spearman between lens order and output order per layer;
  injected-thought introspection with alpha dose; directed modulation; lens-coordinate swaps.
- **RESULT.** Spearman(lens order, output order) at the colon rises **0.475 (L12) → 0.865 (L20) → 0.958 (L29)**
  — the paper's rise replicates. User-turn injection reports the concept **0.65 at alpha=1.0** (0.6 at 0.5,
  dose-responsive) while the **assistant-span control is 0.00 at every strength** — position selectivity
  replicates sharply. Directed modulation: focus **1.00** > ignore **0.60** > baseline **0.20** — including the
  white bear (ignoring primes above baseline); line-count focus 0.40. **Math intermediates (nine→seven) NEVER
  surface** (0.0 in all conditions) — a real capability difference from the paper's frontier models.
  Lens-coordinate swaps flip **9/42 answers (21%)** vs the paper's 59–88%, and never reach top-5 of the
  immediate next-token distribution (flips act through generation).
- **INTERPRETATION.** Report/introspection phenomena are intact at 7B, with weaker swap-based control and a
  genuine capability gap (no math intermediates in the lens).

![figC3](figures/figC3_report_introspection.png)
_figC3 — verbal report and introspection: Spearman(lens order, output order) rises 0.475 → 0.865 → 0.958
(L12→L20→L29); user-turn injection reports the injected concept 0.65 at α=1.0 vs assistant control 0.00;
modulation shows the white-bear ordering (focus 1.00 > ignore 0.60 > baseline 0.20); math intermediates never
surface (`raw/step12/x21\__.json`).\*

### VLM-only: ignition and modality-split bands (figC4, figC5)

- **WHY.** Two VLM-native questions the paper never tested: does the visual workspace commit all-or-none
  (ignition), and do the band statistics split by modality?
- **HOW.** x23: (A) image-mixing ignition — blend two images' projected features `f = (1−a) f_A + a f_B`, sweep
  alpha (20 pair-trials × 41 steps), compare the J-space component's 10–90% transition width against the full
  activation's, and correlate the caption's committed-class flip alpha with the lens transition; (B)
  modality-split band statistics — stickiness of top-1 lens content and object-in-top10 rates at image vs text
  positions; (C) cross-modal report swaps (image-sourced concepts) at text/all/image scopes; (D) modulation
  during captioning (590 instructed runs).
- **RESULT.** (A) **Ignition**: the lens transition is **sharper than the full activation's at every layer**
  (width 0.42 vs 0.63 at L12 → 0.28 vs 0.38 at L31) and sharpens with depth; at L28 the caption's committed
  class flips at the **same alpha as the lens transition (Pearson 0.9887**; threshold alpha mean 0.395 vs
  caption flip 0.479; 3/20 trials show a flip within the sampled alphas). (B) **Modality-split bands**: image
  positions carry highly persistent top-1 lens content at every layer (**stickiness z 9.2 at L12 → 18.6 at
  L31**) yet objects appear there **only in the motor regime** (object-in-top10: 0.000 band-wide → 0.006 at L20,
  0.022 at L24, 0.047 at L28, **0.069 at L31**), while text positions stay token-local (z 0.2–2.0) and hold the
  object workspace. (C) Swap-scope ordering **TEXT 1.7% / ALL 2.1% / IMAGE 0.0%** flip rates — the predicted
  ordering holds (image-position swaps do nothing) but absolute rates are tiny. (D) Behavioral concept-mention
  0.37 under instruction; the concept-vs-visual co-occupancy analysis (z 1.35–1.69 in-band over 590 runs) did
  **not** clearly replicate the paper's task-exclusion pattern at this n.
- **INTERPRETATION.** The visual workspace commits all-or-none with a behaviorally aligned readout, and the band
  splits by modality: a **persistent, non-verbal sensory workspace at image positions** whose contents verbalize
  only at the motor regime, with the object-level workspace living at text positions throughout — the band-level
  quantification of D44's text-mediated handoff.

![figC4](figures/figC4_ignition.png)
_figC4 — image-mixing ignition (20 pair-trials × 41 alpha steps): the J-space component's 10–90% transition is
sharper than the full activation's at every layer (0.42 vs 0.63 at L12 → 0.28 vs 0.38 at L31) and sharpens with
depth; at L28 the caption's committed class flips at the same alpha as the lens transition (Pearson 0.9887)
(`raw/step12/x23_vlm_workspace_ignition.json`)._

![figC5](figures/figC5_modality_split.png)
_figC5 — modality-split bands: image positions carry persistent non-verbal lens content (stickiness z 9.2 → 18.6
across L12→L31) while objects appear there only in the motor regime (object-in-top10 0.000 → 0.069); text
positions stay token-local (z 0.2–2.0) and hold the object workspace; image-sourced swaps flip reports 0.0% vs
text 1.7% / all 2.1% (`raw/step12/x23_vlm_workspace_band.json`, `x23_vlm_workspace_report.json`)._

![figD8](figures/figD8_workspace_dynamics.png)
_figD8 — workspace reasoning dynamics (`step12/x22_reasoning_structure_multihop.json`,
`x22_reasoning_structure_flexgen.json`, `x23_vlm_workspace_modulation.json`): two-hop intermediates surface in
the lens and are causally used before answers (effective-onset gap −0.176 vs the paper's ≈−0.17; probe pass@1
0.269); broadcast-style flexible generalization stays weak (5% at both α vs the paper's 40/53%; loading
Pearson +0.256); modulation-during-captioning: concept mention 0.37 of 590 instructed runs, concept-vs-visual
co-occupancy elevated in-band (z 1.3–1.7) without the paper's clear task exclusion (D47)._

### Interpretation

- **STRUCTURE YES, LEVERAGE WEAK.** LLaVA-1.5-7B possesses the J-space's structural workspace properties (band,
  lens > logit lens at mid, report-correlation rise, depth precedence, white-bear modulation, introspection
  selectivity, image-side ignition with caption alignment) while causal control via lens-vector swaps transfers
  poorly at 7B scale (report flips 21% vs 59–88%; flexgen 5% vs 40/53%; image-scope swaps 0.0%).
- **THE STRUCTURAL NOVELTY** is the persistent non-verbal sensory band at image positions — stable workspace
  statistics (stickiness z up to 18.6) that stay object-free until the motor regime — plus the demonstration
  that the object workspace is text-side, quantifying the Stage-3 text-mediated handoff at the band level.

---

## Operations lessons

| Incident (D-ref) | Mechanism | Mitigation evolution |
| --- | --- | --- |{"type":"excalidraw/clipboard","elements":[{"id":"bumXabRY","type":"text","x":1292.780289418186,"y":4442.285960535139,"width":23.0159912109375,"height":35,"angle":0,"strokeColor":"#1e1e1e","backgroundColor":"transparent","fillStyle":"hachure","strokeWidth":4,"strokeStyle":"solid","roughness":0,"opacity":100,"groupIds":[],"frameId":null,"index":"bBS","roundness":null,"seed":384637248,"version":23,"versionNonce":910963392,"isDeleted":false,"boundElements":null,"updated":1791757139809,"locked":false,"text":"...","rawText":"...","fontSize":28,"fontFamily":5,"textAlign":"left","verticalAlign":"top","containerId":null,"originalText":"...","autoResize":true,"lineHeight":1.25,"hasTextLink":false}],"files":{}}
| OOM under co-tenancy (D11/D12) | Co-tenant held 57–73 GiB; the OOM'd fit process stayed resident holding ~28 GiB | Memory-guarded chain (`run_campaign.sh`, per-step windows, 30 s polls), then `gpu_guard.sh` |
| External SIGKILLs (D14/D15) | Neighbours reclaim-then-launch (a `VLLM::EngineCore` started 13 s after our kill); fits died before their checkpoint | Transient-unbounded retries (`MAX_ROUNDS=60`, stop only on 3 consecutive non-transient), checkpoint cadence 5 (was 20/25; briefly 2 under kill storms, D27), 36/36 supervisor regression checks |
| Speed derate (D22) | Co-tenant at 100% util pins TF32 matmuls to 41–49 TFLOPS (~8–10% of peak) while 14.8–38 GiB stay free — memory-free windows are still ~10× slow | `gpu_guard.sh` probes TFLOPS + free memory before starts; later superseded by per-step window gates + in-step watchdog on the 8×H100 box (D32) |
| ENOSPC (D24/D46) | `/data` at 100%: `torch.save` died in its zip trailer; even ~3 KB saves failed; second full-disk hit in round 3 | Path-aware `DISK_NEED` disk gates + output relocation to a writable mount (D25/D30, `/home/nvidia` in round 3) |
| Fingerprint rigidity (D13) | Checkpoint settings fingerprints include `dim_batch`, so resuming at a different value hard-fails even though it is mathematically inert | Keep `dim_batch` fixed per run; docs note resume requires the same value |
| Estimator memory shape (F10/D21) | One forward on a `dim_batch`-fold expanded batch; the graph is retained from `min(source_layers)` across all `ceil(d_model/dim_batch)` backwards — at `dim_batch=8` the expanded graph (~30 GB at the 13B geometry) survives all 640 backwards | `dim_batch` is the dominant OOM lever (a memory knob, not compute); `dim_batch=1` fits under 36 GiB guards |

![figD10](figures/figD10_campaign_timeline.png)
_figD10 — the campaign timeline (2026-10-01 → 10-11, parsed from DECISIONS.md D-headers): phase lanes and the
14 operational incidents that shaped the tooling (OOM storms, external SIGKILLs, the TFLOPS derate, two ENOSPC
events, two box relocations); inset: reported GPU-hours by phase from REPORT.md §9 (measured vs derived
marked). Schematic spans are disclosed in the figure footnote._

Supervisor lineage: memory guards → `gpu_guard.sh` (occupancy + speed probe + hang trip) → `fit_supervisor.sh`
for the multi-arch shards (wait-for-slot polling; early-OOM load-window trial — a candidate GPU is accepted only
if the process survives `EARLY_SECS=420` without an OOM marker, else kill and try the next GPU; `FIT_EXTRA` passthrough).

## In-flight work (not results)

The **multi-arch extension** — `llava-1.5-13b-hf` (40 layers) and `Qwen2-VL-7B-Instruct` (28 layers, new
`Qwen2VLLensWrapper` handling variable-resolution image tokens) — is **built and gated but not fitted**:
`phase28_gates.sh` runs the real-checkpoint equivalence gate for both architectures before any fit;
`phase29_fit13b.sh` / `phase30_fitqwen.sh` fit 100 COCO captions (masks text/image/all, two shards on different
GPUs, then merge) under 55/45 GiB window guards; `phase31_evals.sh` / `phase32_chain.sh` stage the downstream
evals (`vlm_registry.py` holds the model specs). All fits are queued on GPU availability — **no numbers from
these runs are results yet**; hypotheses referencing them (H6, H7) are pre-registrations.

## What these results license (and what they do not)

**Can** (REPORT §12, updated by later rounds): token-level disposition readouts on **text positions** — per
layer, on fit corpora and held-out shards — including the **last-patch/last-prompt handoff**, where the
hallucination-prone commitment forms; comparative statements across layers, masks and corpora (J-lens vs logit
lens vs translators; text vs image masks; captions vs WikiText), given the metric caveats; a **causal edit
handle** (add/ablate/ swap with well-conditioned 2-column bases, cond ≤ 2.89; 215/1400 caption changes; the 90%
surgical cure at L30); training-free hallucination readouts (pre-generation gap 0.985, per-word lens entropy
0.730, fused image-level risk 0.666 — all single-forward, moments-only); and workspace structure claims
replicated on a VLM (band, lens>logit at mid, report rise, depth precedence, ignition, modality-split bands).

**Cannot**: "the J-lens dominates everywhere" — mid-layer needed the affine correction (D40 revising
D19/D39(c)); image rows as text predictors (V1/D36: the image→image cotangent mass is 24–255× larger, cos 0.39 →
0.09) or "the model sees X" claims from image-position readouts; absolute hallucination rates from rank/KL
values (corpus- and instruction-conditional); ranking claims from mean rank alone (the rank-vs-KL tension,
D19/D43); attention-based hallucination prediction (NULL at power, D46).

## Hypothesis register

Numbered, testable, each with its evidence pointer; predictions are directional; tests are the cheapest
discriminating run on existing tooling.

- **H1 — Grounding-overtake intervention.** Boosting grounded categories at L20–25 text rows converts would-be
  hallucinations. Evidence: D44 X14 — true-object rank 1423 → 34 → 6–15 at the last prompt position
  (L16→L20→L22–25), crossing at L24–26; D44 X13 — ablation at those layers already removes 0.50–0.70.
  Prediction: an add-arm (the mirror of X13's ablate-arm) at L20–25 turns hallucinated captions grounded more
  often than decoding noise, below the α≈27 degeneration boundary (D40). Cheapest test: add an add-arm to
  `x13_cure_test.py` on the same 39 images (~1 GPU-h, no fitting).
- **H2 — Edits through the n=1000 translator.** Composing edit directions as `W_U[t] @ A_l J_l` matches J-vector
  edit efficacy and enables mid-layer cures at lower alpha. Evidence: D41 span overlap — J-vector top-64
  directions carry only 3.5–10.6% of `W_U`'s readout energy (3.4× chance at L30), explaining the 5–15× alpha
  cost; D43 — the n=1000 translator matches the J-lens at L30 (1.30) and dominates mid-layer (L16 1.45 vs 21.4).
  Prediction: the composed direction reaches the same change rate at lower alpha, and mid-layer (L16–24) cures
  become concept-directed rather than change-only. Cheapest test: recompose X13's mean-replacement directions
  through the `step8` n=1000 translator on the same 39-image cure set and compare dose-response arms.
- **H3 — Image-position probe distillation fails mid-band, succeeds at L28+.** Evidence: D47 finding B — image
  positions are sticky (z 9.2–18.6) but object-free until the motor regime (object-in-top10 0.000 → 0.069 at
  L31); D36 — image rows carry image→image cotangent mass (24–255×). Prediction: a linear probe on
  image-position states predicts the image's object set at chance through the mid band and above chance only at
  L28–31. Cheapest test: train a small probe on cached image-position states (D43's CPU-cache tooling) over ~200
  COCO images with COCO category labels.
- **H4 — Ignition-threshold micro-nudges.** Alpha just below a trial's flip threshold changes the committed
  class with minimal collateral. Evidence: D47 finding A — caption flip co-locates with the lens transition
  (Pearson 0.9887 at L28; threshold α 0.395 vs flip α 0.479). Prediction: sub-threshold perturbations flip the
  caption's committed class while leaving most tokens unchanged. Cheapest test: rerun the x23 harness with a
  fine alpha grid around each trial's threshold; count class flips vs collateral token edits.
- **H5 — The pre-generation gap generalizes beyond the fit question bank.** Evidence: D45/D46 X18 — AUROC 0.985
  on 185 questions of the 10-question bank (0.895 on the model's own hallucinated yes). Prediction: the gap
  signal holds across POPE subsets (random/popular/adversarial) and A-OKVQA yes/no, degrading only on
  adversarial splits. Cheapest test: run `x18_pope_handoff.py` on the vendored POPE splits (`data/pope.py`); one
  forward per question.
- **H6 — The sink-dim set is architecture-universal.** Evidence: D46 X16 — the BOS top-16 dim set is identical
  across L16–30, position-typed (image/text share 1512/2789/1076), BOS-clamp destructive (retention 0.278) while
  image dims are inert (0.94–1.00). Prediction: qwen2vl's census will show a single massive outlier group with a
  position-typed, band-stable dim set. Cheapest test: run the X3 census + top-16 dim extraction on the queued
  qwen2vl fit (phase30) — zero new method code.
- **H7 — The band boundary shifts with depth.** Evidence: D47 — 32-layer LLaVA-7B band L17–29 within the mapped
  [12, 29]; the paper's deeper frontier models band [38, 92]. Prediction: on llava-13b (40 layers, phase29 fit)
  the onset/offset layers shift deeper in absolute terms at roughly constant fractional band position. Cheapest
  test: rerun the x22 structure arm on the 13b lens; compare fractional onsets (layer / n_layers).
- **H8 — Fusion of lens entropy + category gap improves with more images.** Evidence: D46 X19 — fused 0.666 at
  n=197 (27 positives), the components alone 0.620 and 0.632, failing on different images. Prediction: the
  z-scored fusion's AUROC rises with n as the components' errors decorrelate. Cheapest test: extend
  `x19_halluc_risk.py` to the 300 held-out captions plus further COCO images with the existing single-pass
  harness and refit the fusion.
- **H9 — The J-lens's L30 exactness makes it the preferred calibration reference for any translator.** Evidence:
  D19/D34/D42 — the L31 readout is the model exactly (identity 0.0; rank 27.31; KL 0.000) and the
  tmtext50+payload lens holds L30 ratio 1.30; D43 — translator quality is limited by the distillation target's
  fidelity. Prediction: distilling the translator against the L30 J-lens readout converges faster per sample at
  equal n. Cheapest test: refit the rank-32 translator at n=40/100 with the L30 J-lens output as target and
  compare the LQS curve to D43's 1.714/2.043.
