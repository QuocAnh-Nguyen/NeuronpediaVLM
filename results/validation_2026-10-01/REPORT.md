# J-lens on LLaVA-1.5 — validation campaign report (2026-10-01)

Campaign executed on the H100 box against the real `llava-hf/llava-1.5-7b-hf` checkpoint.
Autonomous decisions (unstated CONFIG fields, budget, deviations) are logged in
[`DECISIONS.md`](DECISIONS.md); every experiment identifier (X1…X9, V1…V7, E1…E6, F1…F15)
refers to the assumption register [`docs/jlens-vlm-assumptions.md`](../../docs/jlens-vlm-assumptions.md).

Status legend: ✅ done · ⏳ running · ⛔ not run (with reason).

## 0. Headline

<!-- PENDING for S2, X1, X2, X7, X8, X9 (entries below are final). -->

* **X3 (census)**: the leading positions are not sink-like (`pos_1_16_sink_like=false`) and BOS
  is the massive-activation outlier, so `skip_first=1` stands (M14, `step1/x3_norms.json`).
* **X6 (dtype)**: bf16 and fp32+TF32 Jacobians agree to ≤1.5 % per-layer *medians*, but the
  worst cells are real disagreements - 20.1 % (`all`) and 49.2 % (`text`) at L0, the text/L0
  cell also dropping to cosine 0.88, and a single best-fit scale does not absorb any of it.
  The pre-registered rule fires → production fits are fp32+TF32 (1.13× bf16); S1's finished
  bf16 lens stays as a recorded deviation (D18).
* **S1 gate**: validity checks pass on the real checkpoint - identity exact `0.0`, depth trend
  885.54 → 27.31 mean true-token rank, last-layer rank diff `0.0`, frequency control equal to
  the model at L31 - while the strict per-layer middle criterion fails: the J-lens loses to the
  plain logit lens at L17-24 (+99..+813 ranks) and wins at L2-16 and L26-30. Conditional pass,
  finding recorded (D19); the finite-difference row (ii) lands with the re-run.
* **Not yet**: S2 / X1 / X2 / X7 / X8 / X9 ⏳.

## 1. Setup

| Item | Value |
| --- | --- |
| Checkpoint | `llava-hf/llava-1.5-7b-hf` (local HF cache, `HF_HUB_OFFLINE=1`) |
| Hardware | 1× H100 80 GB, shared with a non-campaign co-tenant (2 + 26.7 + 30.7 GiB observed 03:1x; its footprint OOM-killed the fp32 leg — D11) |
| Python | `~/miniconda3/envs/vlm_truth_py313/bin/python`, torch CUDA, transformers 5.x |
| Code | repo `~/ai4life/phuongnh/vlm-lens`, `PYTHONPATH=$REPO/src` |
| Artifacts | `/data/vlm-lens/validation/…` (outside the repo); logs in `~/.vlm-lens-setup/` |
| Budget | 24 h wall, 2.4 h reserved for this report → 21.6 h usable |

**TF32 flag (added mid-campaign):** `vlm_lens.fitting.configure_tf32` + `scripts/fit_llava.py
--allow-tf32`, recorded in provenance as `extra.allow_tf32`, guarded by
`tests/test_precision.py` (torch's `allow_tf32` → `fp32_precision` migration must not
silently no-op the flag). Reason: true-fp32 cuBLAS runs on CUDA cores (~15× slower; D10).

**Equivalence gate** (pre-fit, real checkpoint, bf16, `EQUIVALENCE PASS` required):
✅ passed before every fit chain (`run_step1.sh` runs it first and aborts the chain on
failure); the gate output is in `~/.vlm-lens-setup/step1_chain.log`.

## 2. Step 0 — corpora, splits, cost model, X3 census

### 2.1 Corpora ✅

| Corpus | Fit | Held out | Notes |
| --- | --- | --- | --- |
| COCO val2014 captions (on-policy, `prompt+caption`) | 100 images | 30 images | `seed=0`; disjoint by construction; 10-question bank cycles (10 samples/question); every sample carries LLaVA's own greedy caption |
| WikiText-103 (text control) | 100 prompts (train) | 30 prompts (validation) | `min_chars=600`, `max_tokens=1536`, 0 dropped; token lengths 139–568 (fit), 124–400 (held-out) |

`step0/corpus-split.json` records the image split and the A/B question halves;
`step0/text-corpus.json` records the WikiText span and token-length stats. Both are
frozen on disk, so every fit is offline and reproducible (D1).

### 2.2 Cost model (measured, `step0/cost.json`) ✅

dim_batch 8, bf16, all 31 layers below the target, 3 masks:

| Corpus | Sample geometry | Seconds/sample | Peak GiB | Projection |
| --- | --- | --- | --- | --- |
| image | 660 tokens (82 text + 576 image) | **288.2** | 36.4 | 8.0 h / 100 samples |
| text | 190 tokens | **85.9** | 19.9 | 2.4 h / 100 samples |

These replace the earlier 355 s/sample estimate (D5); S1 therefore kept all 100 prompts and
X4's conditional 4 h stayed available but untriggered.

### 2.3 X3 residual-norm census ✅ — `skip_first=1` stands

50 real COCO fit samples, layers 0/16/31, groups = BOS, positions 1–16, text before the
image block, first 16 patches, patches 17…end, text after the block
(`step1/x3_norms.json`, `code/x3_census.py`):

| Layer | BOS mean ‖h‖ | pos 1–16 | text_pre | patch 1–16 | patch 17–end | text_post |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 8.3 | 29.8 | 3.1 | 38.6 | 39.4 | 2.2 |
| 16 | **1568.8** | 34.7 | 30.0 | 35.8 | 38.2 | 37.4 |
| 31 | **633.7** | 139.7 | 151.8 | 136.0 | 155.0 | 205.0 |

Script verdict: `pos_1_16_sink_like=false`, `recommended_skip_first=1`. BOS is a
massive-activation outlier at depth (M14; V2/V6) but its dominant dimensions are disjoint from every
other group (Jaccard 0.111), and positions 1–16 track the ordinary text/patch groups.
Consequences: the multimodal fits keep `skip_first=1` (D5 in the register), the text control
keeps the paper's 16, and the conditional **X4 boundary sweep was not run** (trigger absent,
D6). X4's chain is committed (`code/run_step5_x4.sh`) for a future run if the boundary is
ever questioned again.

## 3. Step 1 — X6: bf16 vs fp32 fit ✅

Design: the same first 8 COCO fit samples (`step0/manifest-fit.jsonl`), layers
`0,8,16,24,30`, target 31, masks `text,image,all`, `skip_first=1`, `max_seq_len=1536`,
`seed=0`, `dim_batch=8` for **both** legs — only `--dtype` differs. Comparator
`code/x6_compare.py` (relative Frobenius, cosine, best-fit scale, residual after scale
removal); verdict rule: switch production to fp32/tf32 iff rel Δ ≳ a few %.

* bf16 leg ✅ 8/8 samples, 0 skipped, `wall_seconds=2205.8`
  (`/data/vlm-lens/validation/x6-bf16/artifacts/provenance.json`).
* fp32 leg ✅ TF32 (D10): true fp32 ran 2 h 10 min without finishing 4 samples and was
  killed; the TF32 re-run OOMed once during weight load (co-tenant at ~57 GiB, D11) and then
  completed under the retry guard - 8/8 samples, `wall_seconds=2486.5`, `extra.allow_tf32=true`
  (`/data/vlm-lens/validation/x6-fp32/artifacts/provenance.json`).

**Measured** (`step1/x6_dtype.json`, 5 layers × 3 masks, 8 image samples per leg): per-layer
`rel_frobenius` medians are 1.20 % (`all`), 1.26 % (`image`), 1.48 % (`text`); the worst cells
are L0 - 20.1 % (`all`), 4.4 % (`image`), 49.2 % (`text`, cosine there 0.88, and
`rel_after_scale` 48.1 %, i.e. not a scale artefact). Verdict under the pre-registered rule
(`fp32 if median > 3 % or max > 10 %`): `use_fp32 = true` (max 49.2 %), at 1.13× the bf16
wall time (2486.5 s vs 2205.8 s for the same 8 samples) → S2/X1 fit fp32+TF32, S1's existing
bf16 lens stays (D18). Cosine is 0.98+ everywhere except that one text/L0 cell.

## 4. Step 2 — S1: text-only control fit ✅⏳

WikiText-103 train, `skip_first=16` (register D5), mask `all` (the paper's target-protocol
mask), all 31 layers, bf16, 100 prompts. Scored with `code/s1_score.py` on the 30 held-out
prompts, tag `text`:

* (a) rank / KL / top-1 vs the model ceiling; (b) logit-lens baseline and identity check
  (J = I reproduces the logit lens exactly); (iii) last-layer agreement; (iv) depth trend
  (last layer beats the midpoint; monotonicity kept as a diagnostic, D16); (v) unigram
  frequency control; (ii) finite-difference check in fp32 (`eps=1e-2`, 4 top-norm
  *columns* - source dims - at layers 0/8/16/24/30; `J_l` is stored output-dim-major, so
  perturbing a source dim reproduces a column `J_l[:, i]`, not a row; the first two server
  runs died on an all-dims perturbation `IndexError`, D18) — the FD model runs fp32 because
  bf16 cannot resolve the perturbation.

**Measured** (2026-10-02, `step2/s1_score.json`, 30-sample held-out WikiText shard, scored
mask `all` - the corpus has no image tokens, so `text` ≡ `all` - lens `s1-attempt1`): identity
check exact `0.0`; held-out mean true-token rank 885.54
at L16 against 27.31 at L31 (the model's own 27.31); last-layer top-1-in-top-50 and mean rank
equal the model's (0.450 / 27.3, frequency control). The J-lens beats the plain logit lens at
L2-16 (L14: 1101.2 vs 5178.4) and L26-30 (L30: 46.5 vs 75.2) but loses at L17, 20, 21, 23, 24
(+379, +813, +331, +99, +271 against the 16032.5 chance level) - so the gate verdict is FAIL
on the strict per-layer middle criterion only; the diagnosis and the decision to proceed are
D19. The finite-difference rows (ii) land after the fp32 FD model finds a window (D19).

## 5. Step 3 — S2: on-policy caption pilot (100 images) ✅⏳

Design (D4/D7): the 100 fit images are split by instruction half (A = questions 1–5,
B = 6–10, 50 image-disjoint samples each); both halves are fitted with all 31 layers,
masks `text,image,all`, `skip_first=1`, bf16, then merged into the 100-sample lens
(`--merge` weights by `n_prompts`, so the merge equals a single 100-sample run).
`code/s2_eval.py` then scores the merged lens on the 300 held-out captions:

* tags `text`, `image`, `all`, and `image-q0…q3` (patch block quarters — X2/V2/V5),
  reported with the scorer's placeholder exclusion (F3/V4) and with
  `include_placeholders=True` for description;
* mask composition per tag (mean/min/max positions) and `n` per row;
* A4/X8: lens A and lens B cross-scored on held-out question halves (same-half vs
  cross-half cells, no image leakage);
* A4/E6: cross-corpus transfer both ways (caption lens → held-out WikiText at
  `skip_first=16`; text lens → held-out captions at `skip_first=1`).

Prerequisites pre-verified on the frozen corpora (CPU only, 2026-10-01T18:52Z,
`code/split_halves.py`): the halves are 50/50 samples and 50/50 images, sample keys and
images are disjoint, the union equals the fit manifest, the questions partition the
10-question bank (5+5) with half A equal to `corpus-split.json`'s list, the template hash is
propagated into both half headers, both halves preserve the parent manifest's sample order,
and the payload lines are byte-identical across re-runs (only the header's `created_utc`
changes). Since the checkpoint fingerprint excludes the manifest and resume walks
`next_idx` over that stable order, re-running the split on every attempt cannot disturb a
half-fit.

<!-- PENDING: s2_eval.json — per-tag layer tables, quarter spread, halves grid, transfer rows. -->

## 6. Step 4 — X1 target-mask, X7 conditioning, X9 edits ⛔⏳

* **X1** (`code/run_step4.sh`, `code/x1_compare.py`): 20-image shard fitted with
  `target_mask=all` and `target_mask=text`. Bit-level check on one real sample: with the
  image block before the text, the `text` row block must be `torch.equal`-identical between
  the variants and the `image` rows must differ (register V1/D7). The lens-level table then
  quantifies the `image`-row movement (relative Frobenius, cosine, scale, residual after
  scale removal).
* **X7** (`code/x7_x9_interventions.py`): condition number of the 2-column swap basis
  `[v_s, v_t]`, cosine and norm ratio for 10 COCO concept pairs at layers 8/16/24; pairs
  above `cond=1e3` are excluded from swaps and recorded.
* **X9**: greedy generations on 10 held-out captions with `add`/`ablate`/`swap` at layers
  16/24 and an α grid; add strengths are residual-relative
  (`α = k·‖h_layer‖/‖v_t‖`), swap/ablate α ∈ {0.5, 1}. Reported as baseline vs edited text,
  change rate and first differing token — "directions that never change generation are
  mis-scaled, not necessarily meaningless" (register E5).

<!-- PENDING: x1_targetmask.json, x7_conditioning, x9 table + change rate. -->

## 7. Not run (and why) ⛔

| Experiment | Reason |
| --- | --- |
| X4 (`skip_first` sweep) | Conditional on X3 flagging sink-like leading positions; X3 did not (D6). Chain committed. |
| X5 (`target_layer` 31 vs 30) | Outside this campaign's scope; register A6 stays open. |
| X10 (token-restricted estimator) | Not on the critical path; §7 identity is already guarded by a fixture test. |
| POPE yes/no end-to-end | Out of scope for this pilot (populated by `data/pope.py`, exercised in tests only). |

## 8. Register updates

`docs/jlens-vlm-assumptions.md` is the REGISTER. Flips carry an evidence pointer; items whose
verdict needs the fits stay ⏳ (they gate on S1/S2/X1-X9).

| Item | Status | Evidence |
| --- | --- | --- |
| V6 (`skip_first=1` vs 16) | ✅ D5 stands | [measured] X3 census M14: positions 1-16 track the ordinary text/patch groups, `pos_1_16_sink_like=false` (`step1/x3_norms.json`) |
| V8 (prompt-format dependence) | ✅ hash recorded | [measured] `x6-bf16/artifacts/provenance.json` → `corpus.prompt_template_sha256=1cf033c0…` (mirrored in `extra`) |
| V12 (over-length truncation bias) | ✅ rate recorded | [measured] same provenance: `corpus.n_dropped_over_length=0`, `extra.drop_rate=0.0` |
| V10 / F6 (§0 facts; fail-open guard) | ✅ closed | code + tests (step-0 hygiene): a missing `image_seq_length` is derived from (image_size/patch_size)² and the resolved vision config is in provenance (`model.vision_*`: layer −2, select `default`, 336/14 → 576 tokens) |
| V2 (block positional norm gradient) | ⏳ X2 | census M11/M14 shows the gradient at L0; per-quarter fidelity decides whether it matters |
| V1, V3, V4, V5 | ⏳ S2 / X1 | implementation ready (`include_placeholders`, quarter tags, target-mask variant); verdicts need the fits |
| E5 (α units per mode) | ⏳ X7 | residual-relative α from the X3 norms, once a lens exists |
| E6 (holding out) | ⏳ S2 | S1's held-out shard is disjoint by construction; S2 adds the cross-corpus rows |
| F1 §5 (bf16 gradients) | ✅ flipped - production lens moves to fp32+TF32 | [measured] `step1/x6_dtype.json`: medians 1.2-1.5 % but L0 max 49 % (text) ⇒ `use_fp32: true` under the pre-registered rule; the fp32+TF32 leg cost 2486.5 s vs 2205.8 s bf16 (1.13×), so S2/X1 now fit `--dtype float32 --allow-tf32` (D18); S1's existing bf16 lens stays, recorded as a deviation |
| F1-F7 §0 (upstream-code facts) | ✅ verified | pinned to the vendored commit; re-checked against `jlens/fitting.py` for this campaign |
| X1-X9 | per-section verdicts | X3/X6 have their verdicts (sections 2.3, 3); X1/X2/X4/X5/X7/X9 wait for the fits |

Discrepancy log (the brief requires logging register/code mismatches):

* The REGISTER reuses IDs across sections: `F1`-`F7` appear both as §0 upstream-code facts and
  as §5 engine assumptions. The "F1" flip above is the §5 engine item (bf16 gradients), not
  the §0 fact.
* This report's earlier "(E5)" citation for the BOS massive-activation finding was wrong (E5 is
  the α-units item); it now points at M14/V2/V6.
* `positions.py`'s docstring points at `vlm_lens.evaluate` for image-mask validation, and that
  module is implemented - not a stale pointer any more.

* X6's price tag was mis-stated in the earlier reports: "true fp32 is ~15× slower" is the
  *un-flagged* cuBLAS path (D10). With `--allow-tf32` the fp32-weight leg costs 1.13× bf16
  (2486.5 s vs 2205.8 s for the same 8 image samples), so obeying the pre-registered dtype
  rule is cheap; the "fp32" leg is TF32-backed (a bf16-vs-TF32 comparison, TF32's mantissa
  being 4× finer).
* The S1 finite-difference check had never run: both server attempts died in it with
  `IndexError` (the hook wrote 176 positions × 4 dims through a broadcasting index) and,
  once that was fixed, the comparison was *transposed* — perturbing source dim `i` yields
  the estimator column `J_l[:, i]`, while the check compared against row `J_l[i, :]`
  (the estimator stores output-dim-major, `jlens/fitting.py`). Fixed and re-verified:
  independent autograd through the same hook path matches the estimator to 6.7e-08; the
  repaired check gives worst per-layer mean 4.4e-04 at `eps=1e-5` on the tiny CPU fixture
  (gate bar 0.05). The scoring rows are now written before the FD model loads (D18).

## 9. Budget accounting

Accounting unit: **GPU time** (the task CONFIG asked for a "total GPU budget": 24 h with 2.4 h
reserved for this report → 21.6 h usable). Caveat [inference]: if the 24 h was meant as
wall-clock, the elapsed wall (~22 h since 2026-09-30T18:4x, mostly spent waiting for a free
GPU) already exceeds it; then S2's `n` must shrink to the largest multiple of 10 that fits the
remaining time (autonomy rule 3) instead of skipping a step - the step scripts take `n` from
the environment for exactly this.

Spent so far ([measured] from provenance and the chain logs in `~/.vlm-lens-setup/`):

| Step | GPU time | Evidence |
| --- | --- | --- |
| Step 0 corpus + cost + X3 census | ~0.9 h | `step0_chain.log` 23:36:05 → closed 00:30 (includes the `jlens`-import re-run, D8) |
| Equivalence gate (real checkpoint, bf16) | minutes | `step1_chain.log` 00:00:29, `[PASS]` LM stack max abs diff = 0 |
| X6 bf16 leg (8 image samples) | 0.61 h | `x6-bf16/artifacts/provenance.json`: `wall_seconds=2205.8` (275.7 s/sample) |
| X6 fp32/TF32 leg: 7 failed attempts, then the successful run | ~1.8 h | OOM/guard retries 02:03-02:34, then `x6-fp32/.../provenance.json` `wall_seconds=2486.5` (0.69 h) at 03:21, `extra.allow_tf32=true` |
| S1 bf16 fit, first 20 samples (old chain) | ~0.48 h | 04:45 checkpoint, 85.9 s/sample [derived] |
| S1 bf16 fit, remaining 80 samples + artifacts | 2.04 h | `s1_attempt1.log`: resumed 20/100 at 23:30:48, `fitted 1 lens(es) from 100 samples in 7350.4s`; convergence 4.0e-02 over the last 10 samples |
| S1 scoring attempts ×3 (resume 31.8 s + scoring + FD crash) | ~0.4 h | `s1_attempt{1,2,3}.log`; all three died in the FD (D18), now fixed and verified |
| **Total** | **≈ 6.3 h** | [derived] sum of the rows |

Projection for the remaining campaign at the measured costs ([measured] seconds/sample,
dim_batch 8, checkpoint every 5):

| Step | Projected | Basis |
| --- | --- | --- |
| S1 scoring + FD (fixed check) | ~0.5 h | bf16 scoring model; the fp32 FD model (29 GiB) then runs one estimator pass + 40 forwards on 1 sample |
| S2 two 50-image halves in fp32+TF32 + merge + eval | ~9.1 h + ~0.3 h | [derived] 288.2 s/sample (bf16) × 1.13 (the measured X6 leg ratio) × 100 samples |
| X1 shard pair (20 images, both target masks) in fp32+TF32 | ~1.8 h | [derived] 288.2 s/sample × 1.13 × 20: per-sample cost is ~independent of the recorded layer count (the X6 leg fitted 5 layers at 275.7 s/sample on the same corpus) - the earlier ~0.3 h projection was wrong |
| X7 conditioning + X9 edit sweep | ~0.3 h | [inference] 10 short generations + swap math |
| **Total remaining** | **≈ 13 h** | fits the ~15 h left under the GPU-time reading; the optional S1 TF32 re-fit (D18) would exceed it |

Steps 2-4 actuals are ⏳ (not completed); the S1 and X6 rows above are measured, and every
failed attempt's GPU time is included in its row.

State 2026-10-02T07:50Z [measured]: S1's fit is complete (`fitted 1 lens(es) from 100 samples
in 37.0s` on the 07:44 window) and the L24 census published at 07:45:26; the window was too
tight for both at once (see §11, the 07:45 memory race), so the S1 step is requeued for its
scoring re-run after the transient `cuda-error`, and free VRAM is back to 17.75 GiB. X6's
verdict moved S2/X1 to fp32+TF32. Everything else waits on the next window.

## 10. Reproduction appendix

Every step is a committed shell script under `code/`; all of them are idempotent and
resumable, and each re-runs the equivalence gate before its first fit.

```bash
# 0. corpus + cost + census  (network needed for WikiText streaming; then HF_HUB_OFFLINE=1)
bash results/validation_2026-10-01/code/run_step0.sh

# 0b. supervisor/guard logic regression test (no GPU; run on the server)
bash results/validation_2026-10-01/code/test_supervisors.sh
# 0c. analysis check logic (needs torch, no GPU/model): S1 gate (iv) trend rule
python results/validation_2026-10-01/code/test_analysis_logic.py
# 1. equivalence gate + X6 bf16 leg (8 samples)
bash results/validation_2026-10-01/code/run_step1.sh
# 1b. X6 fp32/TF32 leg - memory-guarded and opportunistic (waits for campaign idle)
bash results/validation_2026-10-01/code/run_x6_fp32.sh    # exits 3 when <55 GiB free
bash results/validation_2026-10-01/code/run_x6_retry.sh   # bounded retry loop + compare
# 2. S1 text control fit + scoring/gate
bash results/validation_2026-10-01/code/run_step2.sh
# 3. S2 caption halves + merge + held-out evaluation
bash results/validation_2026-10-01/code/run_step3.sh
# 4. X1 target-mask shard pair + bit-level check
bash results/validation_2026-10-01/code/run_step4.sh
# 5. (conditional) X4 skip_first sweep - not triggered, X3 found pos_1_16 not sink-like
bash results/validation_2026-10-01/code/run_step5_x4.sh
# 5b. X3 re-census including L24 (feeds the X7/X9 alpha units; the live campaign's in-memory
#     script predates this insertion, so it runs out-of-band: 22 GiB guard, 3 attempts,
#     atomic publish by rename)
bash /tmp/x3_recensus_watch.sh
# 6. X7 conditioning + X9 edit sweep (needs a fitted lens dir + X3 norms)
python results/validation_2026-10-01/code/x7_x9_interventions.py \
    --lens-dir /data/vlm-lens/validation/s2-merged/artifacts \
    --manifest /data/vlm-lens/validation/step0/manifest-heldout.jsonl \
    --norms-json /data/vlm-lens/validation/step1/x3_norms.json \
    --json /data/vlm-lens/validation/step4/x7_x9.json

```

## 11. Operational incidents (mid-campaign)

Environment-driven failures shaped the launch mechanics (details in DECISIONS.md):

1. **S1 OOM under a growing co-tenant** (D12, ~04:50): the former supervisor `run_rest.sh`
   held a 27.9 GiB bf16 fit while the co-tenant grew to ~45 GiB across several jobs; the
   fit OOMed, but the OOM'd process kept ~28 GiB resident instead of exiting.
   `code/reclaim_and_launch.sh` kills campaign leftovers, waits for nvidia-smi to confirm
   the release, and launches `code/run_campaign.sh`, which gates every step on a free-VRAM
   window (polled every 30 s) and retries transient failures (OOM, SIGKILL, CUDA errors).
2. **Checkpoint fingerprints include `dim_batch`** (D13): the first guarded attempt asked to
   resume the 20/100-sample S1 checkpoint at `dim_batch=4` and hard-failed
   (`{'dim_batch': (8, 4)}`), even though dim_batch is mathematically inert. S1 was
   relaunched at `dim_batch=8` to keep the 20 samples; S2's fresh fits use 4.
3. **Neighbours SIGKILL our fits; the X6 guard was too low** (D14, 07:16 + 07:28): S1's
   resumed fit (free window 49.4 GiB) and the X6 fp32 leg (free 81.0 GiB) were both
   `Killed` externally, one minute to 23 minutes in; no traceback, no code fault. The X6
   leg had also OOMed once at 47.65 GiB allocated (222 MiB request), proving its 45 GiB
   guard too low. Fixed: guard 55 GiB, transient-aware retry loops in both supervisors,
   checkpoint every 5 samples (was 20/25), and a latent `Any` import bug in
   `tests/test_data_captions.py` (ruff F821). Everything of ours was left stopped.
4. **External kills identified, supervisors hardened and regression-tested** (D15, recon at
   16:1x): the co-tenant's `VLLM::EngineCore` (pid 3351112) started 07:28:33, 13 s after our
   X6 leg was `Killed` (07:28:21) on a GPU that had been empty when the leg started (07:27:20,
   free=81000 MiB); the shared `nvidia-lab` account hosts several people, so a
   reclaim-then-launch step in the co-tenant's tooling is the [inference] source of both
   SIGKILLs. The 06:52:57 chain ran the pre-fix supervisor (only `s1_attempt1.log` exists), so
   D14's retries had never executed. Fixed: transient retries are no longer capped at 3
   attempts (kill storms keep resuming from the checkpoint), `reclaim_and_launch.sh` refuses
   to kill a live chain without `FORCE=1`, and its undefined `free_mib` is defined.
   `code/test_supervisors.sh` (stub-based, no GPU) asserts all of it: 36/36 checks pass on the
   deployed copies.

Measured GPU-sharing volatility (nvidia-smi compute apps, 2026-10-01): 02:5x ~10 GiB
co-tenant; 04:1x 2.0 + 30.7 + 12.1 GiB; 06:5x 12.1 GiB + a fresh 19.0 GiB job; free VRAM
swung between 8 and 49 GiB within minutes. Every long step therefore starts under a
guarded window rather than assuming a quiet box.

Operational notes learned here: `rsync` the repo **before** launching a chain (D9); scripts
that import `jlens` directly must `import vlm_lens` first (D8); reserve
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` on the shared GPU.

`code/run_campaign.sh` drives steps 2–4 and the intervention sweep unattended: it gates
each step on a free-VRAM window (D12) and retries transient failures (D14/D15), so it coexists
with the shared GPU. The retired `code/run_rest.sh`, `code/run_x6_fp32.sh`'s old guard and
the old X6 leg live on only as history.

A self-inflicted memory race worth recording (07:45Z): the L24 census watcher and the
campaign's S1 step woke on the same 32.9 GiB window (07:44:25). The census (16.3 GiB) plus the
bf16 scoring model (15.5 GiB) left ~0.6 GiB, and the first scoring forward died in
`cublasCreate(handle)` (`CUBLAS_STATUS_ALLOC_FAILED`, `s1_attempt1.log`). Both parts did real
work first - the fit resumed and finished (`fitted 1 lens(es) from 100 samples in 37.0s`,
skipped=0) and the census published its L24 rows at 07:45:26 - and the supervisor classified
the failure as a transient `cuda-error` (`other=0/3`) and requeued the step 60 s later. The
race cannot recur (the census is finished); the transferable rule for shared-GPU runs is that
a co-scheduled 16 GiB watcher must not share one window with a fit step - a 28 GiB guard
leaves no headroom for a second model.

## 12. Recommended production settings (provisional)

Current best answers from the real checkpoint. Inputs marked ⏳ are not measured yet (S2's
held-out fidelity rows and the S1 finite-difference row); everything else is measured in this
run. Re-check when those land.

| Knob | Recommended | Basis |
| --- | --- | --- |
| `dtype` | **fp32 weights + TF32 matmuls for production fits (S2, X1)**; bf16 where the lens already exists | [measured] X6 verdict `use_fp32: true` (per-layer medians 1.2-1.5 %, text/L0 max 49 %); the fp32+TF32 leg took 2486.5 s against 2205.8 s for bf16 (1.13×). S1's finished bf16 lens stays (D18). |
| `dim_batch` | 8 (text), 4-8 (image) | [measured] peak 19.9 GiB (text, 190 tok) and 36.4 GiB (image, 660 tok) at dim_batch 8; pure memory knob, but it is part of the checkpoint fingerprint, so keep it fixed per run (D13). |
| `skip_first` | 1 (multimodal), 16 (text-only control) | [measured] X3 census: positions 1-16 are not sink-like (`pos_1_16_sink_like=false`); BOS is the massive-activation outlier (D5). |
| `target_layer` | 31 (final residual) | [derived] the lens lives on the pre-norm residual stream; 31 is the last fitted source layer. X5 (31 vs 30) was not run. |
| masks | `text` = primary readout; `image`/`all` descriptive only | [derived] mask semantics (V4/E3): a readout at an image position is a first-order disposition to verbalize, not a next-token prediction. |
| prompts | 100 fit / 30 held-out per corpus | [measured] cost model: text 85.9 s/sample (~2.4 h/100); image 288.2 s/sample bf16, ~326 s/sample fp32+TF32 (~9.1 h/100) at dim_batch 4-8; scale n by the budget rule and keep the 10 % report reserve. |
| checkpoint cadence | every 5 samples | [measured] an external SIGKILL then costs at most 5 samples; fits are resumable and the supervisor retries transients up to `MAX_ROUNDS=60` (D14/D15). |
| memory guards | ≥28 GiB free (text fit), ≥32 GiB bf16 / ≥55 GiB fp32+TF32 image fits, ≥32 GiB for the fp32 FD model | [measured] peaks plus the fp32 OOM at 47.65 GiB allocated; `run_campaign.sh` polls every 30 s, and `s1_score.py` now waits for a window before loading its fp32 FD model instead of falling back to a multi-hour CPU run (D19). |
| environment | `HF_HUB_OFFLINE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, greedy decoding, `torch.compile` off | [measured] the box is first-come-first-served; 65.7 GiB observed held by a co-tenant, and our fits have been SIGKILLed twice by external reclaims (D14/D15). |

Claims this setup can and cannot support:

* Can: token-level disposition of the *text* residual after multimodal fusion ("what is this
  state disposed to say"), per layer, on the fit corpus and its held-out shard - S1's
  validity checks pass on the real checkpoint (identity exact 0.0, depth trend 885.5 -> 27.3
  mean rank, last-layer alignment 0.0, frequency control equal to the model at L31) even
  though the strict per-layer middle criterion fails (D19); S2's held-out rows are ⏳.
* Can: comparative statements across layers, masks and corpora (J-lens vs logit lens, text vs
* Cannot: "the J-lens dominates the plain logit lens at every depth" - [measured] S1 shows it
  wins at L2-16 and L26-30 but loses at L17-24 (+99..+813 ranks, D19).
* Cannot (yet): causal claims from lens numbers alone (X7/X9 are the causal probe and they are
  qualitative); "the model sees X in the image" from an image-position readout; absolute rate
  claims ("hallucinates N%") from rank/KL values, which are corpus- and
  instruction-conditional.
