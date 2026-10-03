# VLM visualization research — readout, attribution, knockout, steering (LLaVA-1.5-7B)

Research notes for a local interactive tool that explains captioning hallucination in LLaVA-1.5-7B
with a Jacobian lens (J-lens), patch attribution, causal knockout and steering.

**Method.** Every external URL below was fetched and read on 2026-10-03; nothing is cited from memory.
Two frequently miscited IDs are corrected here: the Chefer CVPR'21 relevance method is
`arXiv:2012.09838` (not 2012.12241) and VL-InterpreT is `arXiv:2203.17247` (not 2205.07960).
Two arXiv items named in early notes (`2310.05916`, `2406.11193`) are deliberately **not** cited
because their titles/claims were not re-read. In-repo grounding is cited as `[A <id>]` =
`docs/jlens-vlm-assumptions.md`, `[U §n]` = `docs/jlens-vlm-audit.md`, `[R]` =
`third_party/jacobian-lens/README.md` (workspace `vlm-lens` repo; measured pilot data are A §5.1).

**Views.** V1 session+generate · V2 caption-token trace (per-layer top-k + prob-vs-layer) ·
V3 24×24 patch-grid lens heatmap + quadrant aggregation (image-q0..q3) · V4 causal patch knockout ·
V5 attention-rollout overlay · V6 interactive steering.

**Status tags.** `[PROVEN STANDARD]` = established method with replicated evidence (may still be a
documented adaptation to LLaVA). `[ADAPTATION]` = the *mechanistic claim* is unvalidated on LLaVA;
J-lens itself is `[ADAPTATION]` because the workspace paper contains zero multimodal experiments
`[U §2]` and the repo targets the final layer while the paper's readouts default to the penultimate
one `[U §2 #7]`. `[CAVEAT]` = read as a warning for a technique, not a view candidate.

## 1. Technique → view map

Cost convention: a "generation pass" for 12 tokens is 0.6–1.0 s on the pilot H100 `[A M10]`; lens
arithmetic marked `[arith]` is my own multiplication, everything else is sourced.

| Technique | Source (verified URL, year) | Status | Views | Mechanistic read for hallucination | Cost (forwards / VRAM) |
| --- | --- | --- | --- | --- | --- |
| Logit lens | https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/interpreting-gpt-the-logit-lens (2020) | [PROVEN STANDARD] text LLMs | V2 | Decodes intermediate residuals into vocab space; late layers approach the model distribution, early layers are brittle — read late layers and measure fidelity | 1 matvec/layer/position over W_U; residual cache 32·594·4096·2 B ≈ 156 MB/gen [arith] |
| Tuned lens | https://arxiv.org/abs/2303.08112 (2023) — code https://github.com/AlignmentResearch/tuned-lens | [PROVEN STANDARD] text; [ADAPTATION] MLLM via https://arxiv.org/abs/2608.30705 | V2 | Per-block affine probes are more predictive/reliable than logit lens in early layers; causal experiments show it uses model-like features; correlational and can "skip ahead" (https://transformer-circuits.pub/2026/workspace/index.html) | training: corpus forward + probes; storage 32·4096²·2 B ≈ 1.07 GB [arith] |
| J-lens | https://transformer-circuits.pub/2026/workspace/index.html (2026) + [R] | [ADAPTATION] (no multimodal experiments [U §2]) | V2,V3,V6 | lens_l(h)=unembed(J_l·h), J_l=E[∂h_final/∂h_l] fitted on 1000×128-token text prompts [R]; causal (first-order transport), interventions run in lens coordinates | fit 288.2 s/sample, 36.4 GiB peak, 8.0 h/100 images [A M13]; artifact 1.07 GB/mask [A M4]; dim_batch memory-only, 266 s at 8/16, OOM at 32 [A M3] |
| Token-restricted lens vectors v_t | [A §7] (2026) + [A F15] | [ADAPTATION] | V6 (scoring for V3) | v_t = W_U[t]·J_l is exactly the gradient of the lens score of token t; restricting to a token set costs one backward per token instead of one per hidden dim | ceil(|T|/dim_batch) backward passes; e.g. 40–512× cheaper for few tokens [A §7] |
| Attention rollout / flow | https://arxiv.org/abs/2005.00928 (2020) | [PROVEN STANDARD] text; [ADAPTATION] to a 594-token multimodal sequence | V5 | Attention weights are mixed across layers → "unreliable as explanation probes"; rollout/flow correlate better with ablation importance and input gradients than raw attention | 1 forward with attention capture; all-heads cache 32·32·594²·2 B ≈ 0.72 GB, last-row-only ≈ 2.4 MB [arith] |
| Chefer relevance (CVPR'21) | https://arxiv.org/abs/2012.09838 (2021) — code https://github.com/hila-chefer/Transformer-Explainability | [ADAPTATION] (ViT classifier → decoder-only) | V5 | Gradient×attention relevance with layer-wise propagation; class-specific maps as a rollout alternative | one backward per target token/layer — spot checks only |
| Chefer bi-modal (ICCV'21) | https://arxiv.org/abs/2103.15679 (2021) — code https://github.com/hila-chefer/Transformer-MM-Explainability | [PROVEN STANDARD] encoder-decoder VL (DETR/LXMERT/CLIP) | V5 (vision tower only) | Cross-modal co-attention relevance; the reference if vision-tower co-attention ever enters scope | per-sample gradient passes |
| Attention ≠ explanation | https://arxiv.org/abs/1902.10186 (2019) | [CAVEAT] | V5 | Adversarial attention distributions leave predictions intact → attention weights are not causal evidence | — |
| Attention as diagnostic | https://arxiv.org/abs/1908.04626 (2019) | [CAVEAT] | V5 | "Not not explanation": attention is usable if paired with controls — motivates rollout + knockout side by side | — |
| ROME causal tracing | https://arxiv.org/abs/2202.05262 (2022) | [PROVEN STANDARD] | V4 | Corrupt→restore localizes factual recall to specific mid-layer states; the template for mask/restore on image tokens | 1 forward per condition (batched), no gradients |
| Causal mediation | https://arxiv.org/abs/2004.12265 (2020) | [PROVEN STANDARD] | V4 | Total/natural direct/indirect-effect decomposition; small component sets carry the effect → patch effects need not be additive | 2 forwards per factor |
| Activation-patching practice | https://arxiv.org/abs/2309.16042 (2023) | [PROVEN STANDARD] methodology | V4 | Metric and corruption-baseline choices can flip conclusions → freeze a protocol (Δ logprob on baseline caption, greedy decode) and report zero/mean variants | 1 forward/condition pair |
| Attention knockout (edges) | https://arxiv.org/abs/2304.14767 (2023) | [PROVEN STANDARD] text | V4 | Attention-edge knockout reveals subject→attribute routing; upgrade path when embedding-level masking is too coarse | 1 forward per edge set |
| POPE | https://arxiv.org/abs/2305.10355 (2023) | [PROVEN STANDARD] eval | V4,V6 | Object hallucination tracks instruction frequency and co-occurrence priors; binary polls are the cheap scoring harness for edits | generations per question set |
| CHAIR | https://arxiv.org/abs/1809.02156 (2018) | [PROVEN STANDARD] eval | V4 | Caption-level metric; hallucinated objects reflect training label co-occurrence | 1 generation/caption |
| VCD | https://arxiv.org/abs/2311.16922 (2023) | [PROVEN STANDARD] decoding | V6 | Over-reliance on statistical bias + unimodal priors; contrast image-conditioned vs distorted-image distributions | 2 forwards/step (or 2 generations) |
| OPERA | https://arxiv.org/abs/2311.17911 (2023) | [PROVEN STANDARD] decoding | V5,V6 | Hallucinations follow "knowledge aggregation patterns" (over-trust of few summary tokens) and neglect of image tokens | beam search × penalty; behavioral baseline only |
| IBD | https://arxiv.org/abs/2402.18476 (2024) | [PROVEN STANDARD] decoding | V6 | Over-reliance on linguistic priors; contrast with an image-biased branch to amplify image-correlated tokens | 2 forwards/step |
| LVLM hallucination survey | https://arxiv.org/abs/2404.18930 (2024) | [PROVEN STANDARD] | all | Taxonomy of causes (visual encoding, language prior, data/alignment) and evaluations | — |
| LVLM-Interpret | https://arxiv.org/abs/2404.03118 (2024) — code https://github.com/IntelLabs/lvlm-interpret | [PROVEN STANDARD] tool | V3,V4,V5 | Interactive patch-relevancy + grounding assessment with a LLaVA failure case study; the Gradio app runs `llava-hf/llava-1.5-7b-hf` | their server: 1 forward + relevancy pass per request |
| VL-InterpreT | https://arxiv.org/abs/2203.17247 (2022) | [PROVEN STANDARD] tool (encoder-decoder era) | V2,V5 | Head statistics across layers, cross/intra-modal attention heatmaps, hidden-representation trajectories | dashboards over 1 forward |
| CircuitsVis | https://github.com/TransformerLensOrg/CircuitsVis (MIT; LICENSE.txt pinned) | [PROVEN STANDARD] components | V2,V5 | React/Python visualization components (token coloring, attention patterns) reusable in the front-end | none |
| nnsight / NDIF | https://github.com/ndif-team/nnsight (MIT) + https://nnsight.net/ + https://ndif.us/ | [PROVEN STANDARD] infra | V1,V4,V6 | Deferred traces let you hook and intervene on internals without rewriting model code; remote execution option | local: one model load; remote: round-trip |
| TransformerLens | https://github.com/TransformerLensOrg/TransformerLens (MIT) | [PROVEN STANDARD] infra | V2 | Hook/cache patterns worth reusing; wrapping a VLM is non-trivial (vision tower) | — |
| PruMerge | https://arxiv.org/abs/2403.15388 (2024) — code https://github.com/42Shawn/LLaVA-PruMerge | [ADAPTATION] | V3 | CLS-attention sparsity exposes spatial redundancy: 14× compression on LLaVA-1.5 with comparable scores → most of the 576 tokens are redundant, so quadrant aggregates are defensible | offline selection (1 CLIP pass) |
| FastV | https://arxiv.org/abs/2403.06764 (2024) | [ADAPTATION] | V3,V5 | "Visual token attention is extremely inefficient in deep layers" (LLaVA-1.5 in scope); half the tokens can be dropped after layer 2 without hurting performance → a low late-layer image-attention share is expected, not evidence of ignoring the image | 1 forward + pruning bookkeeping |
| Neo et al. | https://arxiv.org/abs/2410.07149 (2024) | [ADAPTATION] — LLaVA-1.5 specifics | V3,V4 | Removing object-specific tokens drops object identification >70%; visual-token representations become increasingly vocabulary-aligned with depth; read out at the last token position | ablation = 1 forward/set |
| Jiang et al. | https://arxiv.org/abs/2410.02762 (2024) | [ADAPTATION] | V3,V6 | Vocabulary projection of image representations: real objects decode more confidently than hallucinated ones; linear orthogonalization removes up to 25.7% of hallucinated objects | 1 forward + projections |
| EAH | https://arxiv.org/abs/2411.09968 (2024) | [ADAPTATION] | V5 | Attention sinks inside image tokens: dense shallow, sparse deep; boosting high-density image-sink heads alleviates hallucination | attention stats from 1 forward |
| DAMRO | https://arxiv.org/abs/2410.04514 (2024) | [ADAPTATION] | V3,V5 | Decoder image-token attention mirrors the encoder's CLS attention and concentrates on background "outlier" tokens; CLS filter re-scores | 1 forward |
| Redundancy→Relevance | https://arxiv.org/abs/2406.06579 (2024) | [ADAPTATION] | V5 | Attention+LLaVA-CAM: information flow converges shallow and diversifies deep; a context-dependent "information-flow cliff" explains why truncating layers breaks grounding | 1 forward |
| Text inertia | https://arxiv.org/abs/2407.21771 (2024) | [ADAPTATION] | V5,V6 | Same outputs with and without the image; amplifying image attention and subtracting text-only logits reduces hallucination | 2 forwards/step |
| MemVR | https://arxiv.org/abs/2410.03577 (2024) | [ADAPTATION] | V2,V6 | Decoder "amnesia" about visual content: re-injecting visual tokens mid-stack mitigates hallucination → annotate where visual evidence peaks/dips in V2 | 1 forward with re-injection |
| AdaIAT | https://arxiv.org/abs/2603.04908 (2026) | [ADAPTATION] | V5 | Real object tokens assign *higher attention to generated text* than hallucinated ones (attention to image is not the separator); layer-wise thresholds on generated-text attention cut hallucination 35.8%/37.1% on LLaVA-1.5 | attention stats + per-head rescoring |
| Same Attention, Different Truths | https://arxiv.org/abs/2608.07302 (2026) | [ADAPTATION] — closest direct study | V3,V4,V5,V6 | Real and hallucinated objects get equally strong mid-late visual attention; logit-lens decodability separates them; two mechanisms: visual uncertainty (masking removes the hallucination) vs contextual prior (masking fails, attention drifts) | masking = 1 forward/condition |
| Logit Lens Supervision | https://arxiv.org/abs/2602.01530 (2026) | [ADAPTATION] | V3 | Patch-level logit-lens maps on LLaVA-v1.5-7B: off-the-shelf maps are weakly tied to source regions; an auxiliary objective sharpens them and improves grounding/hallucination → raw maps are hypotheses, not evidence | 1 forward + vocab projection |
| Vision-Default Prior-Override | https://arxiv.org/abs/2606.28273 (2026) | [ADAPTATION] | V4,V6 | Activation patching across residual/heads/MLP: ~2.5–4.8% of heads causally enable prior grounding in the second half; ablating flips 68–96% prior→visual but only 0.8–7.5% visual→prior → head shortlist | 1 forward per patched condition |
| ContextualLens | https://arxiv.org/abs/2411.19187 (2024) | [ADAPTATION] | V2 | Argues logit lens is limited for generalized hallucination; mid-layer contextual embeddings improve detection/grounding | probes per layer |
| VisLens | https://arxiv.org/abs/2608.30705 (2026) | [ADAPTATION] | V2,V3 | Logit+tuned lens on MLLMs, single pass; reported 8.5–22.2× speedup vs prior visual-search pipelines → supports one-forward/many-views budgets | 1 forward + lens heads |
| CAA | https://arxiv.org/abs/2312.06681 (2023) | [PROVEN STANDARD] steering | V6 | Contrast-pair activation addition at inference; direct template for interactive add/ablate | 1 generation/edit |
| RepE | https://arxiv.org/abs/2310.01405 (2023) | [PROVEN STANDARD] steering | V6 | Reading + control vectors; representation-engineering toolchain | linear probe + 1 gen |
| MoReS / LLaVA Steering | https://arxiv.org/abs/2412.12359 (2024) | [ADAPTATION] | V6 | Persistent modality imbalance with text dominating output during visual instruction tuning; re-balance visual representations per layer → the "steer toward vision" preset | 1 generation/edit |
| CMAC | https://arxiv.org/abs/2501.01926 (2025) | [ADAPTATION] | V3,V5 | Position bias + spurious inter-modality correlations in cross-modal attention; masking high-cross-attention value vectors as distortion → calibrate position before ranking patches | 2 forwards/step for the contrastive variant |
| Attention sinks | https://arxiv.org/abs/2309.17453 (2023) | [CAVEAT] | V5 | First-token sink absorbs mass; sink attention is not semantics (in-repo: BOS norm 8.3 → 1568.8 at L16 `[A M14]`) | — |
| Massive activations | https://arxiv.org/abs/2402.17762 (2024) | [CAVEAT] | V2,V5,V6 | A few outlier dimensions dominate residual norms → residual-relative α scaling and norm-aware normalization | — |
| Register tokens | https://arxiv.org/abs/2309.16588 (2023) | [CAVEAT] | V3 | High-norm "register" artifacts in ViT feature maps → background hot spots can be artifacts, not semantics | — |
| MMFM interpretability survey | https://arxiv.org/abs/2502.17516 (2025) | [PROVEN STANDARD] | all | Taxonomy of adapting LLM interpretability to multimodal models; gap list for missing techniques | — |
| VISTA (Hidden Life of Tokens) | https://arxiv.org/abs/2502.03628 (2025) | [ADAPTATION] | V2 | Visual information degrades gradually across decoding; early-excitation effect; a mitigation reports ~40% hallucination reduction | per-step readouts |
| LLaVA / LLaVA-1.5 | https://arxiv.org/abs/2304.08485 (2023) · https://arxiv.org/abs/2310.03744 (2023) | [PROVEN STANDARD] models | — | Architecture + training context for the target checkpoint (`llava-hf/llava-1.5-7b-hf`) | — |

## 2. Signals for captioning hallucination and the UI affordance for each

Each signal below is split into **measured** (replicated result) vs **interpretation** (mechanism
attributed by an author or by me).

1. **Language-prior over-trust / "text inertia".** *Measured:* VCD shows decoding over-relies on
   statistical bias and unimodal priors (https://arxiv.org/abs/2311.16922); the text-inertia study
   reproduces identical outputs with and without the image (https://arxiv.org/abs/2407.21771); IBD
   contrasts image-conditioned vs image-biased branches for the same reason
   (https://arxiv.org/abs/2402.18476); VLMs underperform their own CLIP encoder on classification
   largely because class-frequency exposure pins behavior
   (https://arxiv.org/abs/2405.18415); ~2.5–4.8% of heads are causally necessary for prior grounding
   in the second half of the stack (https://arxiv.org/abs/2606.28273). *UI:* every readout view gets
   a **no-image twin run**; V2 plots image-conditioned and text-only lens distributions on the same
   axes and reports the logit gap as a first-class number; V6 exposes a "prior-suppression" α.
2. **Total visual attention is not the separator.** *Measured:* real and hallucinated objects receive
   equally strong mid-late visual attention; logit-lens decodability of the attended region differs,
   and masking reveals two mechanisms (visual uncertainty vs contextual prior)
   (https://arxiv.org/abs/2608.07302); real object tokens attend *more to generated text* than
   hallucinated ones (https://arxiv.org/abs/2603.04908). *Interpretation:* attention budgets are
   not a hallucination meter; the informative quantity is whether the attended content is decodable
   and causally used. *UI:* V5 must never show a bare attention heatmap as "explanation"; it pairs
   rollout weights with a per-region logit-lens decodability band (V3 scores) and links to a V4
   knockout button for the selected region.
3. **Attention sinks and massive activations.** *Measured:* first-token sinks
   (https://arxiv.org/abs/2309.17453) and outlier activation dimensions with huge norms
   (https://arxiv.org/abs/2402.17762) exist in text LMs; in-repo, the L16 BOS residual norm is
   1568.8 vs 26–48 for image patches and 633.7 at L31 `[A M11/M14]`; image-token sinks behave
   differently by depth (dense shallow, sparse deep: https://arxiv.org/abs/2411.09968). *UI:* V2/V5
   normalize out and label BOS/sink positions; V3 lets the user mask high-norm patches; V6 α is
   residual-relative (`α = k·‖h‖/‖v‖`), not absolute `[A E5]`.
4. **Visual-token redundancy, background/outlier dominance.** *Measured:* 14× token compression with
   comparable accuracy (https://arxiv.org/abs/2403.15388); pruning half the visual tokens after layer
   2 is nearly free (https://arxiv.org/abs/2403.06764); decoder attention tracks the encoder's CLS
   attention and focuses background outlier tokens (https://arxiv.org/abs/2410.04514); ViT register
   artifacts inflate background patches (https://arxiv.org/abs/2309.16588); removing object-specific
   tokens destroys object identification >70% (https://arxiv.org/abs/2410.07149). *UI:* V3's quadrant
   (image-q0..q3) aggregation is only a summary — always expose the raw 24×24 grid plus
   mean/max per quadrant, and offer a "top-k patches only" knockout preset (V4) that spends its
   forward passes where the lens says information lives.
5. **Spatial/positional structure inside the image block.** *Measured:* with LLaVA's causal masking,
   patch *i* attends only to patches ≤ *i*, so patch positions are not exchangeable `[A F8, V2]`;
  patch residual norms grow from 3.0 (first patch, L0) to 79.9 (last patch) and mid-stack patches
  sit at 26–48 `[A M11/M14]`; cross-modal position bias and spurious correlations are large enough to
   need explicit calibration (https://arxiv.org/abs/2501.01926). *UI:* V3 shows patch index and
   raster order alongside the grid; quadrant tables are reported **per ordered quarter** (X2 pending
   `[A X2]`), never as an unordered set average; V5 message-passing is sensitive to position, so
   report attention by (layer, quarter) not just totals.
6. **Depth profile of visual evidence.** *Measured:* re-injecting visual tokens mid-stack mitigates
   hallucination ("amnesia", https://arxiv.org/abs/2410.03577); information flow converges shallow,
   diversifies deep, with a context-dependent cliff (https://arxiv.org/abs/2406.06579); visual
   information degrades gradually across decoding steps (https://arxiv.org/abs/2502.03628);
   in-repo, held-out lens fidelity generalizes at L30 but is non-monotone mid-depth (L10 beat L20 at
   n=10) `[A M12]`. *UI:* V2's prob-vs-layer chart carries a shaded "lens-validated" band (late
   layers, where fidelity was measured) and a vertical marker for the current steering layer; warn
   when the user reads mid-depth rows as belief.
7. **Two hallucination mechanisms, distinguished by intervention.** *Measured:* masking the attended
   region removes visual-uncertainty hallucinations, while contextual-prior hallucinations persist
   and attention drifts to other regions (https://arxiv.org/abs/2608.07302); analogous asymmetry in
   prior-enabling heads (https://arxiv.org/abs/2606.28273). *UI:* V4 classifies each knockout result
   into "removed / persisted / changed" and V6 offers a "visual evidence injection" preset (steer a
   region's decoded object direction) for the persisted class.
8. **Representation-level grounding confidence.** *Measured:* real objects decode from image
   representations with higher confidence than hallucinated ones
   (https://arxiv.org/abs/2410.02762); visual-token representations become increasingly
   vocab-aligned with depth (https://arxiv.org/abs/2410.07149); patch-level logit-lens maps are only
   weakly grounded off the shelf and improve with a dedicated objective
   (https://arxiv.org/abs/2602.01530); a logit-lens consistency check detects hallucination
   (https://arxiv.org/abs/2608.07302). *UI:* V3 colors patches by the lens score of the *selected
   verbalization token* and additionally shows its rank among vocab entries (rank, not just
   probability); V2 offers the same rank toggling for caption tokens.

**Claim hygiene:** V1 image-tag readouts are descriptive only — ~94% of the image-row cotangent mass
lands on later patch positions inside the image block, so image-position lens rows mostly continue
the image block rather than verbalize it `[A V1]`; the scorer drops placeholders and reduces the
image tag to ~1 position per sample `[A V5]`; the final-layer row is the model by construction
`[A E2]`; and item 8's confidence ordering is a property of *readouts*, not of the generator's
beliefs unless a V4/V6 intervention confirms it.

## 3. Reusable code inventory

| Library | License (verified how) | Adapt / why not |
| --- | --- | --- |
| `jacobian-lens` (workspace `third_party/jacobian-lens`) | Apache-2.0 ([R]) | Keep: fit/merge/save/load, `lens_l` readout, interactive slice page (d3 + ISC) with top-1 cell + rank superscript and pinned-token rank charts — the visual grammar for V2/V3 |
| `vlm_lens` + `scripts/` (workspace) | repo-internal | Keep: measured-fit config, `trace_generation`, `generate_with_edits`, quarter masks; fix defect list from `[U §5]` (cache keys, swap conditioning) before exposing in UI |
| TransformerLens https://github.com/TransformerLensOrg/TransformerLens | MIT | Borrow hook/cache patterns; do not port a full VLM (vision tower integration cost) |
| tuned-lens https://github.com/AlignmentResearch/tuned-lens | MIT | Only if V2 needs calibrated early-layer rows; adds training infra for one view |
| CircuitsVis https://github.com/TransformerLensOrg/CircuitsVis | MIT (LICENSE.txt pinned) | Reuse React components (token coloring, attention patterns) in the front-end; Python package optional |
| BertViz https://github.com/jessevig/bertviz | Apache-2.0 (LICENSE pinned) | Design reference for V5 head/layer views; Jupyter-widget-bound, so not embeddable as-is |
| Chefer Transformer-Explainability https://github.com/hila-chefer/Transformer-Explainability | MIT (LICENSE pinned) | Reuse relevance-propagation code to prototype a second attribution metric against rollout |
| Chefer MM-Explainability https://github.com/hila-chefer/Transformer-MM-Explainability | MIT | Only if vision-tower co-attention is added; includes DETR/LXMERT/CLIP generators |
| Captum https://github.com/meta-pytorch/captum | BSD-3-Clause (LICENSE pinned) | Input-space attribution for sanity checks (which pixels matter); not patch attribution — do not put in the six views |
| LVLM-Interpret https://github.com/IntelLabs/lvlm-interpret | Apache-2.0 | Closest existing tool (patch relevancy + grounding + causality for LLaVA); study its Gradio UX and `llava-hf/llava-1.5-7b-hf` integration; small research release |
| nnsight https://github.com/ndif-team/nnsight | MIT | Optional: intervention tracing without touching model code; NDIF remote only if the tool ever moves off local GPUs |
| PruMerge https://github.com/42Shawn/LLaVA-PruMerge | Apache-2.0 | Token-importance code if a "prune then rerun" control is added to V4 |
| attention_flow https://github.com/samiraabnar/attention_flow | **no license detected** (raw LICENSE → 404) | Do not copy; rollout is ~20 lines — reimplement from the paper (https://arxiv.org/abs/2005.00928) |
| FastV / DAMRO / MemVR repos | license unverified | Cite papers only; no code reuse until licenses are read |

## 4. Rejected techniques (and why, mechanistically)

1. **Raw attention heatmaps (single layer/head) as the patch attribution.** Attention weights are
   mixed across layers and are demonstrably replaceable by adversarial distributions that preserve
   behavior (https://arxiv.org/abs/1902.10186); rollout exists precisely because raw weights
   "unreliable as explanation probes" (https://arxiv.org/abs/2005.00928). *Mechanism:* a softmax
   weight is not an information-flow quantity; without marginalizing over layers you report routing
   noise. Kept only as a toggle in V5, never as the default.
2. **Pixel saliency maps (Grad-CAM/Integrated Gradients on input pixels) as the causal story.**
   Saliency maps can be invariant to model and data randomization (https://arxiv.org/abs/1810.03292).
   *Mechanism:* gradient saliency measures local sensitivity in a saturated, path-dependent way and
   cannot express "remove this patch and the caption changes", which is exactly the claim the tool
   needs. Captum is available for sanity checks but stays out of the six views.
3. **t-SNE/UMAP scatter plots of visual-token residuals.** Cluster sizes, distances and apparent
   clusters are artifacts of hyperparameters and are routinely misread
   (https://distill.pub/2016/misread-tsne/). *Mechanism:* the lens story is directional (does token
   *t* decode?), and forcing residuals into 2-D preserves neither cosine ranks nor the unembedding
   geometry the readout uses.
4. **Patch→word "word clouds" from a single lens layer without rank or causal check.** Off-the-shelf
   patch-level lens maps are weakly tied to source regions (https://arxiv.org/abs/2602.01530), and
   top-1 decoding ignores the rank gap that separates real from hallucinated regions
   (https://arxiv.org/abs/2608.07302). *Mechanism:* a top-1 word can be a low-confidence argmax of a
   flat distribution; showing it without rank (and without the V4 knockout) overstates grounding.
5. **Uncertainty/entropy shading of caption tokens as the hallucination signal.** Attention
   magnitude does not separate real from hallucinated objects
   (https://arxiv.org/abs/2608.07302, https://arxiv.org/abs/2603.04908), and logit-lens rows alone
   are limited for generalized hallucination (https://arxiv.org/abs/2411.19187). *Mechanism:* final
   probabilities conflate visual evidence with language prior; the separation appears in
   *consistency* checks (lens vs attention) and interventions, not in scalar confidence.
6. **Full per-head attention galleries (all 32×32 heatmaps).** 1024 panels carry no causal ordering;
   causal evidence shows only ~2.5–4.8% of heads matter for prior grounding
   (https://arxiv.org/abs/2606.28273), and image-token sink behavior is a small head subset
   (https://arxiv.org/abs/2411.09968). *Mechanism:* heads are redundant and often cancel; a gallery
   substitutes visual density for evidence. V5 instead ranks heads by knockout effect and shows the
   top-k.
7. **Vision-tower-only explanations (CLIP attention/rollout alone).** The lens operates on the
   post-fusion residual, and the vision tower/projector is outside it `[A V7]`; decoder attention
   largely mirrors encoder CLS attention including background outliers
   (https://arxiv.org/abs/2410.04514). *Mechanism:* naming the encoder's suspects does not identify
   what the language stack does with them; keep the tower frozen and use it only to explain V3's
   background artifacts (registers, https://arxiv.org/abs/2309.16588).

## 5. Engineering readout — concrete choices per view

**V1 — session + generate.**
- *Data computed:* one greedy generation with a fixed prompt template `USER: <image>\nDescribe this
  image.\nASSISTANT:` (18 text tokens + 576 placeholders = 594 `[A F4]`), capturing per-block
  outputs (pre-final-norm) at all positions, plus the tokenizer ids and the template hash.
- *Defaults:* bf16, greedy, `max_new_tokens=40` (the fit protocol's caption length; at 40 tokens 93.6% of its targets are placeholders `[A F4]`), seed
  fixed; model loaded via the LLaVA fork path (`get_image_features` at layer −2, `masked_scatter` at
  id 32000 `[A F7]`) because upstream `jlens` cannot forward image batches `[U §2]`.
- *Pitfalls:* processor quirks — pass PIL images, never paths `[A M8]`; orientation metadata handling;
  keep the template constant or readouts shift `[A V8]`; record the template so cross-sample
  comparisons are valid; don't regenerate when only the lens layer changes (readout is cheap; the
  forward is not: 288 s/fit-sample scale `[A M13]` is fit-time, inference generation is sub-second to
  seconds `[A M10]`).

**V2 — caption-token trace.**
- *Data computed:* per generated token, per layer ∈ {late stack, e.g. 24–31}: top-k (k=10) lens
  tokens with probabilities and the rank of a pinned token; plus the model-logits row as anchor;
  plus a text-only twin run for the prior-gap.
- *Defaults:* show L30 and L31 side by side (paper default = penultimate vs repo target = final,
  X5 unmeasured `[A X5]`); label the final row "model output" and never present it as lens evidence
  `[A E2]`; color by rank, not probability, when comparing layers; default filter to text positions
  (image positions have no ground truth `[A V4]`).
- *Pitfalls:* mid-depth rows are not monotone in fidelity (L10 > L20 at n=10 held-out `[A M12]`);
  image-tag rows are descriptive only `[A V3/V5]`; don't materialize full-vocab logits per layer×row
  (32·594·32000 ≈ 1.2 GB fp16 [arith]) — keep top-k only; the final-layer score row is synthetic by
  construction `[U §3 #5]`.

**V3 — patch grid + quadrant aggregation.**
- *Data computed:* score(layer, patch p, token t) = lens logit/prob of t at patch p (J_l applied to
  that position's residual, then `unembed`), on a 24×24 grid; aggregation per quadrant q0..q3 as
  mean and max; per-layer normalization; patch index/meta stored for tooltips.
- *Defaults:* layer slider starting at 16 and 24 (informative mid stack per `[A M7]` L16 showing
  `Stadium, Tennis, ...` for a tennis-court sample); per-layer z-score color scale by default with a
  raw-scale toggle; mark high-norm patches (registers/background) and BOS-like positions; show the
  selected token's rank in the tooltip (top-1 alone overstates, see §4.4).
- *Pitfalls:* patch order is causal, not spatial: patch *i* sees only patches ≤ *i* `[A F8/V2]` —
  never average quarters as if exchangeable; quadrant means hide within-quarter gradients (show max);
  a shared cross-layer color scale masks the depth story; the vision side (CLIP patch per se) is not
  what is being visualized `[A V7]`; 25.7%-style erasure results come from a purpose-built
  orthogonalization (https://arxiv.org/abs/2410.02762), not from a generic top-k filter.

**V4 — causal patch knockout.**
- *Data computed:* for each mask condition S (quadrants first, then top-k lens patches): (a) Δ
  logprob of the pre-generated baseline caption under masking, scored with teacher forcing — the
  primary metric, 1 forward, no generation; (b) a regenerated caption (greedy) for the CHAIR/POPE
  readout; (c) the mask spec, seed, and the zero/mean choice. Implementation point: mask at the
  projector output (or the image-feature rows feeding `masked_scatter`); `vlm_lens.interventions`
  exposes residual edits only (verified in `src/vlm_lens/interventions.py`), so this is a small fork
  addition hooked around the fusion step.
- *Defaults:* zero and mean masks both offered; run conditions one at a time or add a
  multi-condition batch path (`MultimodalBatch` is batch-size 1; `expand()` replicates one identical
  sample for the fit `[A F10]`); classify outcomes as removed/persisted/changed to expose the two
  mechanisms (https://arxiv.org/abs/2608.07302); fix the metric and corruption protocol up front
  (https://arxiv.org/abs/2309.16042) and note non-additivity of effects
  (https://arxiv.org/abs/2004.12265).
- *Pitfalls:* regenerated captions are the only valid causal evidence — post-edit lens readouts are
  the linear model's prediction `[A E4]`; each condition costs ~1 gen (0.6–1.0 s/12 tokens `[A M10]`)
  so a 4-quadrant × 2-mode sweep is seconds, but a 576-single-patch sweep is not; masking may leave
  contextual-prior hallucinations intact and attention drifts (https://arxiv.org/abs/2608.07302);
  the four-quarter masks assume equal chunks — verify with the quarter fidelity test `[A X2]`.

**V5 — attention-rollout overlay.**
- *Data computed:* rollout matrix R = Π_l (0.5·A_l + 0.5·I) per head (or head shortlist), evaluated on
  rows for the target token(s); aggregate image columns to 576 patches; alongside it, raw last-layer
  attention share by modality (text/image/BOS) per layer.
- *Defaults:* average heads then rollout, or rollout per shortlisted head (top-k by V4 sensitivity or
  the prior-enable head shortlist from https://arxiv.org/abs/2606.28273); capture only the last-row
  (≈2.4 MB) instead of the full matrix (0.72 GB) [arith]; exclude BOS/sink columns from share
  normalization and say so on the chart; default toggle: rollout overlay + lens decodability, not raw.
- *Pitfalls:* rollout is correlational and can disagree with causal effects — always offer the V4
  button next to it (https://arxiv.org/abs/1902.10186, https://arxiv.org/abs/1908.04626); the
  0.5+I mixing coefficient is a modeling choice, document it; sinks and massive activations inflate
  mass (https://arxiv.org/abs/2309.17453, https://arxiv.org/abs/2402.17762, `[A M14]`); heads are
  not interchangeable — a full-gallery default is rejected in §4.6; vision-tower attention is out of
  the lens's scope `[A V7]`.

**V6 — interactive steering.**
- *Data computed:* direction v_t = W_U[t] @ J_l (exactly the readout gradient `[A F15]`), or the
  §7 token-restricted estimator when only a few tokens are steered; edits: `add` (α·v̂ scaled to
  ‖h‖), `ablate` (project out), `swap` (2-coordinate change via the [v_s, v_t] basis), applied at
  layer L, positions ∈ {last, all}.
- *Defaults:* residual-relative α (α = k·‖h‖/‖v‖, k ∈ {0.5, 1, 2, 3}); positions default `last`
  for minimal edits and `all` for broadcast; per-edit generation 0.6–1.0 s/12 tokens `[A M10]` so the
  UI can stream before/after captions; exposure of the two priors: "ablate prior direction" and
  "inject visual evidence" presets (§2.7); log ‖v‖, residual norms and the swap basis condition
  number on every edit `[A E5, X7]`.
- *Pitfalls:* transposed-Jacobian bug class — v_t must be W_U @ J (the guard test
  `test_lens_vectors_are_readout_gradients` exists `[A F15]`); v_t omits the final RMSNorm Jacobian,
  so "add token t" raises the *linear* readout, not necessarily the real logit `[A E7]`; swap
  pseudo-inverse amplifies nearly-collinear pairs — alert above condition ~1e3 `[A X7]`; one-layer,
  one-position edits understate the paper's band protocol `[U §3 #7]`; caching edits keyed by layer
  alone breaks when a second token reuses the layer (known defect `[U §5]`); causal claims must come
  from actual generations, never from post-edit lens readouts `[A E4]`; α units differ per mode
  `[A E5]`.
