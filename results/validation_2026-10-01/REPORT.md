# J-lens on LLaVA-1.5 — validation campaign report (2026-10-01)

Campaign executed on the H100 box against the real `llava-hf/llava-1.5-7b-hf` checkpoint.
Autonomous decisions (unstated CONFIG fields, budget, deviations) are logged in
[`DECISIONS.md`](DECISIONS.md); every experiment identifier (X1…X9, V1…V7, E1…E6, F1…F15)
refers to the assumption register [`docs/jlens-vlm-assumptions.md`](../../docs/jlens-vlm-assumptions.md).

Status legend: ✅ done · ⏳ running · ⛔ not run (with reason).

## 0. Headline (PENDING — fill as steps complete)

<!-- PENDING: one-paragraph verdict per experiment X1, X3, X6, X7, X8, X9 and the S1 gate. -->

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
massive-activation outlier at depth (E5) but its dominant dimensions are disjoint from every
other group (Jaccard 0.111), and positions 1–16 track the ordinary text/patch groups.
Consequences: the multimodal fits keep `skip_first=1` (D5 in the register), the text control
keeps the paper's 16, and the conditional **X4 boundary sweep was not run** (trigger absent,
D6). X4's chain is committed (`code/run_step5_x4.sh`) for a future run if the boundary is
ever questioned again.

## 3. Step 1 — X6: bf16 vs fp32 fit ✅⏳

Design: the same first 8 COCO fit samples (`step0/manifest-fit.jsonl`), layers
`0,8,16,24,30`, target 31, masks `text,image,all`, `skip_first=1`, `max_seq_len=1536`,
`seed=0`, `dim_batch=8` for **both** legs — only `--dtype` differs. Comparator
`code/x6_compare.py` (relative Frobenius, cosine, best-fit scale, residual after scale
removal); verdict rule: switch production to fp32/tf32 iff rel Δ ≳ a few %.

* bf16 leg ✅ 8/8 samples, 0 skipped, `wall_seconds=2205.8`
  (`/data/vlm-lens/validation/x6-bf16/artifacts/provenance.json`).
* fp32 leg ⏳ TF32 (D10): true fp32 ran 2 h 10 min without finishing 4 samples and was
  killed; the TF32 re-run then OOMed during weight load because the co-tenant held
  ~57 GiB (D11). `code/run_x6_retry.sh` waits for the campaign to go idle and retries
  under a ≥45 GiB free-memory guard (`run_x6_fp32.sh` exits 3 and is retried).

<!-- PENDING: x6_dtype.json numbers + verdict (switch production fit to fp32/tf32 iff rel Δ ≳ few %). -->

## 4. Step 2 — S1: text-only control fit ✅⏳

WikiText-103 train, `skip_first=16` (register D5), mask `all` (the paper's target-protocol
mask), all 31 layers, bf16, 100 prompts. Scored with `code/s1_score.py` on the 30 held-out
prompts, tag `text`:

* (a) rank / KL / top-1 vs the model ceiling; (b) logit-lens baseline and identity check
  (J = I reproduces the logit lens exactly); (iii) last-layer agreement; (iv) depth trend
  (last layer beats the midpoint; monotonicity kept as a diagnostic, D16); (v) unigram
  frequency control; (ii) finite-difference check in fp32
  (`eps=1e-2`, 4 top-norm rows at layers 0/8/16/24/30) — the FD model runs fp32 because
  bf16 cannot resolve the perturbation.

<!-- PENDING: s1_score.json tables + gate dict (i–vi) + the logit-vs-J comparison. -->

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

<!-- PENDING: status flips for V1/V2/V3/V5/V6, E5/E6, F1, and X1–X9 verdicts; each with the
one-line evidence pointer (JSON path + row). -->

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
| X6 bf16 leg (8 samples) | 0.61 h | `x6-bf16/artifacts/provenance.json`: `wall_seconds=2205.8` |
| X6 fp32 attempts (true fp32, then TF32) | ~1.1 h | `step1_chain.log` closed 02:20 (`X6_FP32_FAILED`); the 06:48 retry OOMed; the 07:27 attempt was killed after 1 min |
| S1 bf16 fit attempts (20 samples, then ~23 min) | ~0.85 h | 04:45 checkpoint (20 samples at 85.9 s/sample [derived]); 06:52:57 → 07:16:47 external kill |
| **Total** | **≈ 3.5 h** | [derived] sum of the rows |

Projection for the remaining campaign at the measured costs ([measured] seconds/sample,
dim_batch 8, checkpoint every 5):

| Step | Projected | Basis |
| --- | --- | --- |
| S1 fit 100 prompts + scoring + FD | 2.4 h + ~0.2 h | 85.9 s/sample; FD is minutes (CPU fallback possible, D16) |
| S2 two 50-image halves at dim_batch 4 + merge + eval | ~8 h + ~0.3 h | 288.2 s/sample |
| X1 shard pair (20 images, 6 layers each) | ~0.3 h | [inference] 288.2 s/sample × (6/31 layers) × 20 |
| X7 conditioning + X9 edit sweep | ~0.3 h | [inference] 10 short generations + swap math |
| **Total remaining** | **≈ 12 h** | fits the 21.6 h usable under the GPU-time reading, reserve intact |

Steps 2-4 actuals are ⏳ (not completed); the failed attempts above are all the GPU time they
have consumed so far.

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
# 5. (conditional) X4 skip_first sweep
bash results/validation_2026-10-01/code/run_step5_x4.sh
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

## 12. Recommended production settings (provisional)

Current best answers from the real checkpoint. Inputs marked ⏳ are not measured yet (X6's
fp32 verdict, the S1 gate, S2's held-out fidelity rows); everything else is measured in this
run. Re-check when those land.

| Knob | Recommended | Basis |
| --- | --- | --- |
| `dtype` | **bf16** for all fits | [measured] bf16 X6 leg: 8/8 samples, 2205.8 s, images at dim_batch 8; fp32 weights are ~29 GiB and the TF32 leg OOMed at 47.65 GiB allocated under a co-tenant. ⏳ switch to fp32/TF32 iff `step1/x6_dtype.json` shows median relative Frobenius Δ > 3% (max > 10%). |
| `dim_batch` | 8 (text), 4-8 (image) | [measured] peak 19.9 GiB (text, 190 tok) and 36.4 GiB (image, 660 tok) at dim_batch 8; pure memory knob, but it is part of the checkpoint fingerprint, so keep it fixed per run (D13). |
| `skip_first` | 1 (multimodal), 16 (text-only control) | [measured] X3 census: positions 1-16 are not sink-like (`pos_1_16_sink_like=false`); BOS is the massive-activation outlier (D5). |
| `target_layer` | 31 (final residual) | [derived] the lens lives on the pre-norm residual stream; 31 is the last fitted source layer. X5 (31 vs 30) was not run. |
| masks | `text` = primary readout; `image`/`all` descriptive only | [derived] mask semantics (V4/E3): a readout at an image position is a first-order disposition to verbalize, not a next-token prediction. |
| prompts | 100 fit / 30 held-out per corpus | [measured] cost model: text 85.9 s/sample (~2.4 h/100), image 288.2 s/sample (~8.0 h/100) at dim_batch 8; scale n by the budget rule and keep the 10% report reserve. |
| checkpoint cadence | every 5 samples | [measured] an external SIGKILL then costs at most 5 samples; fits are resumable and the supervisor retries transients up to `MAX_ROUNDS=60` (D14/D15). |
| memory guards | ≥28 GiB free (text fit), ≥32 GiB (image fits), ≥55 GiB (fp32 leg) | [measured] peaks above plus the fp32 OOM at 47.65 GiB allocated; `run_campaign.sh` polls every 30 s. |
| environment | `HF_HUB_OFFLINE=1`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, greedy decoding, `torch.compile` off | [measured] the box is first-come-first-served; 65.7 GiB observed held by a co-tenant, and our fits have been SIGKILLed twice by external reclaims (D14/D15). |

Claims this setup can and cannot support:

* Can: token-level disposition of the *text* residual after multimodal fusion ("what is this
  state disposed to say"), per layer, on the fit corpus and its held-out shard - once the S1
  gate passes and S2's held-out fidelity rows are in.
* Can: comparative statements across layers, masks and corpora (J-lens vs logit lens, text vs
  image rows, WikiText vs captions) - those are the S2 A4/X8 rows.
* Cannot (yet): causal claims from lens numbers alone (X7/X9 are the causal probe and they are
  qualitative); "the model sees X in the image" from an image-position readout; absolute rate
  claims ("hallucinates N%") from rank/KL values, which are corpus- and
  instruction-conditional.
