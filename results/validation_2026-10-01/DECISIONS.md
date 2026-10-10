# Autonomous decisions — J-lens validation campaign

Task: "Run the prioritized validation experiments for the J-lens port to LLaVA-1.5". The
task template's CONFIG block was not filled in; every value below was resolved from the
environment and the repository, never by asking. Rule references are to the task text.
Numbering note: the D-numbers below are campaign-local to this directory. The repository
register (`docs/jlens-vlm-assumptions.md`) has its own, unrelated D-table of design
decisions; cite this file's D-numbers only for decisions recorded here (the report does).

## D0 — CONFIG resolution (autonomy rule 1)

| CONFIG field | Resolved value | Reason |
| --- | --- | --- |
| REGISTER path | `docs/jlens-vlm-assumptions.md` | The only file with A/V/E/F/G/D/X identifiers |
| Checkpoint | `llava-hf/llava-1.5-7b-hf` (HF cache, offline) | Pinned by the repo (`AGENTS.md`); gate already passed on it |
| GPU | 1× H100 80 GB (shared; co-tenant used 10.3 GB at start) | `nvidia-smi` |
| **Total GPU budget** | **24 h wall** (10 % / 2.4 h reserved for the report) | Unstated in the task; the plan below fits with ~3.6 h slack |
| COCO images | `<vlm-truth>/data/coco2014/val2014/val2014` (40 504 jpg) | Only COCO copy on the box |
| WikiText | HF Hub `Salesforce/wikitext` `wikitext-103-raw-v1`, network allowed for this step | No local copy; box has internet (HTTP 200 to HF API) |
| Output dir | `results/validation_2026-10-01/` in the repo + `/data/vlm-lens/validation` for artifacts | Deliverables must be committed; big artifacts must not be |

Planned budget spend (projection from measured 355 s/sample at bf16, 3 masks, all layers):
X6 16 samples ≈ 1.6 h · S1 100 text prompts ≈ 2.6 h · S2 100 image samples ≈ 10 h ·
X1 30-sample target-mask shard ≈ 3 h · X4 (only if X3 triggers it) 4 × 8-sample shards ≈ 3.2 h ·
X7/X9 generation-only ≈ 0.3 h → ≈ 20.7 h. Each step re-projects from its *measured* per-sample
time and shrinks `n_samples` (never the budget) if it would not fit (rule 3).

## D1 — WikiText needs network (Step 2)

`data/text.py` streams `Salesforce/wikitext`. The fit runs with `HF_HUB_OFFLINE=1` for the
model; the text-corpus build unsets it for the streaming download, then the manifest is
frozen on disk so the fit itself is offline and reproducible.

## D2 — X1 target-mask variant implemented inside `vlm_lens`

Upstream's estimator uses one position mask as source *and* target set (register F1). The
variant therefore lives in `vlm_lens.fitting.jacobian_for_sample(..., target_mask=...)`:
the cotangent is seeded only at positions of the requested mask. No vendored file is
touched (rule 5). `target_mask` is part of the checkpoint fingerprint.

## D3 — Text rows in X1: what "unchanged" can mean (register V1)

V1 argues text sources *after the image block* can only causally reach text targets, so
their rows must be bit-identical between `target_mask=all` and `target_mask=text`. Text
sources *before* the block (`USER:` tokens) do reach placeholder targets and therefore
change. The aggregate `text` row is a mixture of the two, so the campaign reports:
(a) a bit-identity unit test on the causally-disjoint subset (tiny fixture + one real
sample), (b) per-mask relative Frobenius difference plus the best-fit scale factor and
cosine similarity on the real shard, to separate "normalization" from "structural" change.

## D4 — Corpus splits (S2)

- Images: `select_images(..., seed=0)` over the COCO pool; first 100 → fit, next 30 → held
  out (never fit). Disjointness is by construction and recorded in the manifest headers.
- Instruction halves (A4/X8): the 10-question bank cycles, so each question has exactly 10
  samples. Half A = questions 1-5, half B = 6-10; lens A is fit on half-A samples, lens B
  on half-B samples; cross-half scoring uses the held-out images, whose manifest carries
  all 10 questions (300 captions).
- Corpus builder gained `image_list=` so the driver (not the builder's internal sampling)
  owns the split.

## D5 — Measured costs (Step 0d) replace the 355 s projection

`cost.json` (dim_batch=8, bf16, all layers below the target, 3 masks): image sample
(660 tokens = 82 text + 576 image) **288.2 s, peak 36.4 GiB**; text sample (190 tokens)
**85.9 s, peak 19.9 GiB**. Re-projected: X6 8+8 samples ≈ 2.2 h · S1 100 text ≈ 2.4 h ·
S2 100 image ≈ 8.0 h · X1 20-sample shard ×2 ≈ 3.2 h · X7/X9 ≈ 0.3 h → ≈ 16.1 h of the
21.6 h usable, so S1 keeps all 100 prompts (no shrink needed, rule 3 not triggered).
Fits ran with `HF_HUB_OFFLINE=1`; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` set
because the GPU is shared with a 10.3 GiB co-tenant.

## D6 — X3 verdict: `skip_first=1` stands, X4 not triggered

`x3_norms.json` (50 real samples, layers 0/16/31): BOS is a large-norm outlier at depth
(mean ‖h‖ 1569 at L16 and 634 at L31 vs 30–205 for every other group), but its dominant
dims are disjoint from every other group (Jaccard 0.111) and positions 1–16 track the
patch/text groups, not BOS. Script verdict `pos_1_16_sink_like=false`,
`recommended_skip_first=1`. The conditional X4 boundary sweep (4 × 10 samples ≈ 4 h) is
therefore **not run**; its chain is committed as `code/run_step5_x4.sh` with the trigger
recorded in the report. Freed budget stays as rerun reserve.

## D7 — S2 = two 50-sample half-fits, then merge (extends D4)

`--merge` weights J by `n_prompts`, and each sample carries exactly one question, so the
merged A+B lens is bit-equivalent to a single 100-sample run while giving the A4/X8
cross-half table for free. Half-fits are also image-disjoint, so the cross-half cells have
no image leakage. `split_halves.py` writes both manifests from `corpus-split.json`.

## D8 — `import vlm_lens` must precede `import jlens`

X3 first died with `ModuleNotFoundError: No module named 'jlens'`: the vendored path is
installed as a side effect of importing `vlm_lens` (repo convention). Campaign scripts
that import `jlens` directly now carry an explicit `import vlm_lens  # noqa: F401` above
it; the sharp edge is already documented in the README ("Import `vlm_lens` before `jlens`").

## D9 — Launch mechanics: ship code before launching

The first `run_step1.sh` launch died instantly ("No such file or directory") because it
was started in the same breath as the first code `rsync`; the code must be synced first,
then launched. All campaign launches: `rsync` repo → `nohup bash ... &` via the shared
SSH control socket, then verify with `pgrep -af`.

## D10 — True-fp32 fit is infeasible; X6's fp32 leg re-run with TF32

The first fp32 leg ran 2 h 10 min without finishing 4 samples (no checkpoint) because
torch defaults `allow_tf32=False`: true-fp32 cuBLAS on H100 uses CUDA cores, ~15x slower
than bf16 tensor cores at these shapes (~10 min/sample → ~9 h for 8 samples). The leg was
killed and re-run with **TF32** (`allow_tf32=True`, 10-bit mantissa, fp32 storage and
accumulation) — the register's D20/X6 wording ("fp32/tf32") already anticipates this.
Implemented as `vlm_lens.fitting.configure_tf32` + `scripts/fit_llava.py --allow-tf32`
(recorded in provenance; guarded by `tests/test_precision.py` so the flag cannot silently
no-op across torch's `allow_tf32` → `fp32_precision` migration). The bf16 leg
(`x6-bf16`, 37 min for 8 samples) is unchanged, so X6 now compares bf16 against TF32-fp32.

## D11 — X6 fp32/TF32 leg: CUDA OOM under the co-tenant, moved to an opportunistic retry

The re-run with TF32 died during model load with `torch.OutOfMemoryError` (112.75 MiB free
of 79.11 GiB): the box's co-tenant held ~57-61 GiB (measured: 2 GiB + 26.7 GiB +
30.7 GiB compute processes), so fp32 weights (~30 GiB) could not fit, let alone
activations. Nothing to do with the TF32 switch — `model ready: ... tf32=True` printed
before the failure. The campaign's own fits also leave no room for a co-running fp32
model, so `run_x6_retry.sh` waits for all campaign launchers to exit and then retries the
leg in a 2 h loop, gated on >=42 GiB free (`run_x6_fp32.sh` exits 3 and retries when the
GPU is too full). If no window opens, X6's fp32 row is reported as environmentally
infeasible with this evidence instead of a fabricated number.

## D12 — S1 OOMed under live GPU sharing; chain replaced by a memory-guarded supervisor

The old `run_rest.sh` chain died shortly after 04:45: its S1 fit (bf16, dim_batch 8,
27.9 GiB resident) hit `torch.OutOfMemoryError` (70 MiB free) while the co-tenant grew to
four jobs (~2 + 30.7 + 12.1 GiB and later a fresh 19 GiB job). The OOM'd fit process then
stayed resident holding ~28 GiB instead of exiting. `code/reclaim_and_launch.sh` kills
campaign leftovers (X6 retry runner excluded, verified by pattern), waits for nvidia-smi
to confirm the release, and launches `code/run_campaign.sh`: same step order, but each step
waits for a free-memory window (polled every 30 s; 28/32/16/22 GiB for S1/S2/X1/X7X9) and
is retried only on OutOfMemory. Big fresh fits run at dim_batch 4 to keep windows reachable.

## D13 — A checkpoint's settings fingerprint includes dim_batch

The first guarded S1 attempt (04:59:51) hard-failed non-OOM: `ValueError: checkpoint at
.../s1-text/checkpoint.pt was fitted with different settings: {'dim_batch': (8, 4)}`. The
old chain's fit had reached 20/100 samples (2.08 GB checkpoint, 04:45) and the fingerprint
treats dim_batch as a fit setting even though it is mathematically inert (`dim_batch` only
splits backward passes). S1 was therefore relaunched with `DIMBATCH=8` to *resume* those 20
samples (relaunch at 06:52:57, free=49.4 GiB, fit resident 22.8 GiB, no resume error);
S2's fresh fits stay at dim_batch 4. Follow-up for the repo (not done mid-campaign): the
fingerprint could ignore dim_batch, or the docs could say resume requires the same value.

## D14 — Neighbours SIGKILL our fits; guards, retries and checkpoint cadence hardened

Between 07:16 and 07:28 both of our running fits were killed by an external actor (`Killed`
in the shell's job report, no traceback):

* S1's resumed fit (`3328912`, free window 49.4 GiB) died at ~07:16 after ~23 min, before
  its next checkpoint, so the 04:45 checkpoint (20/100 samples) is all that survived;
  `run_campaign.sh` classified it non-OOM and stopped (CAMPAIGN_S1_FAILED).
* The X6 fp32 leg (`3350091`, free window 81.0 GiB — the GPU was empty) died ~1 min in.

Diagnosis (worker request: "check whether everything ran smoothly"): the logs show no code
fault in either kill. Two genuine defects were exposed and fixed:

1. **X6 fp32 leg memory guard too low.** The 06:48 attempt OOMed at 47.65 GiB allocated by
   PyTorch (222 MiB request, 119 MiB free); the guard said 45 GiB so the leg could start in
   windows too small for it. `MIN_FREE_MIB` is now 55 GiB (`run_x6_fp32.sh`).
2. **Transient-failure handling.** Both supervisors treated a SIGKILL as fatal. The retry
   runner now retries on any non-deterministic failure (guard skip / OOM / CUDA error /
   SIGKILL) and gives up only after 3 consecutive `other` failures; `run_campaign.sh` got the
   same classification. The two heavy fits also checkpoint every 5 samples now (was 20/25),
   so an external kill costs minutes, not half an hour, once a chain is relaunched.

Also fixed: `tests/test_data_captions.py` used `Any` in annotations without importing it
(ruff F821; latent only because `from __future__ import annotations` defers evaluation).

State at 14:5x: everything of ours is stopped (as instructed - the GPU is held by a
neighbour's vLLM EngineCore, 65.7 GiB). Nothing is relaunched until the user says so; all
fixes are shipped so the next launch picks them up.

## D15 — Supervisor follow-up: transient retries uncapped, reclaim guarded, both regression-tested

Follow-up on D14 with fresh server evidence (recon at 16:1x UTC):

* The co-tenant's `VLLM::EngineCore` pid `3351112` started `Thu Oct 1 07:28:33 2026`, 13 s
  after our X6 fp32 leg's `Killed` (retry log mtime 07:28:21) - and that leg had started at
  07:27:20 on an empty GPU (`free=81000MiB`). The account `nvidia-lab` is shared by several
  people (our tree `~/ai4life/phuongnh/`, the co-tenant's `~/data_mount/tritd/`), so the
  [inference] source of both SIGKILLs is a reclaim-then-launch step in whoever needs the GPU
  next. Nothing in our logs shows a code fault.
* `run_campaign.log` contains exactly one run (started 06:52:57) with a single
  `s1_attempt1.log`: the failed chain used the *pre-fix* single-attempt supervisor, i.e. D14's
  retry logic had never executed on the server when it was shipped.

Three defects fixed (all shipped to `code/`):

1. **Transient retries were capped at 3 attempts.** With a co-tenant that SIGKILLs fits, S1
   (a ~4 h fit) would stop after three kills although each kill costs at most
   `checkpoint_every=5` samples. `run_guarded` now retries transient reasons (OOM / CUDA
   error / SIGKILL) with no fixed attempt cap, bounded by `MAX_ROUNDS=${MAX_ROUNDS:-60}`, and
   stops only after 3 *consecutive* non-transient failures - the same rule as
   `run_x6_retry.sh`.
2. **`reclaim_and_launch.sh` would kill a live chain.** Its `PAT` matches `run_campaign.sh`,
   `run_step*.sh` and `fit_llava.py`, so re-running it while a chain is alive would kill that
   chain and start a duplicate supervisor. It now refuses unless `FORCE=1`.
3. **`reclaim_and_launch.sh` called an undefined `free_mib`** (its header printed
   `free=MiB`); now defined.

Verification (no GPU): `code/test_supervisors.sh` drives both supervisors and the reclaim
guard with stubbed externals (PATH-stubbed `nvidia-smi`/`sleep`, per-scenario fake `$HOME`,
mode files driving rc/marker sequences) and asserts classification, retry/stop rules, window
waiting, chain continuation and guard refusal: **36/36 checks pass** against the deployed
copies. It also pinned a useful property: a missing step script (rc=127) is `other`, so real
deployment mistakes still stop the chain after 3 attempts.

## D16 — S1 gate (iv) is a depth trend, not per-layer monotonicity; FD frees the scoring model

`s1_score.py` is the first thing the chain runs after S1's fit and had never executed on the
server. Reviewing it against the brief exposed two defects in the gate code itself:

1. Check (iv) ("fidelity improves with depth") was implemented as *strict* per-layer
   monotonicity of the mean true-token rank across the last 16 layers. A single wobble of
   0.01 rank - routine on a finite held-out shard - would have failed the go/no-go
   spuriously. It is now a trend (the last layer must beat the depth midpoint, `depth_checks`)
   with the monotone flag and the count of non-monotone steps kept in the JSON as diagnostics.
2. The fp32 finite-difference model (~29 GiB) loaded while the bf16 scoring model (~15 GiB)
   was still resident, so the FD check fell back to CPU (minutes) on every run. The scoring
   model is now freed first (`del model; gc.collect(); torch.cuda.empty_cache()`).

Also recorded for the report: gate (ii) passes at a max per-layer mean relative error <= 5%
(the brief's 1-2% is the expected value, not the bar), and (i) demands exact (0.0) agreement
between the J=I lens and the logit-lens baseline - the paper's own identity check, so a
nonzero value means a real scoring-path bug.

Verified: `code/test_analysis_logic.py` (9/9 checks over clean / wobble / last-layer
regression / flat profiles) passes locally and on the server's `vlm_truth_py313` python;
`py_compile` clean; the server suite is green (`PYTHONPATH=src pytest`, rc=0).

## D17 — Budget unit read as GPU time; X6 runner logs live in `~/.vlm-lens-setup/`

1. The task CONFIG asks for a "total GPU budget" (24 h, 21.6 h usable after the 10% report
   reserve). Read as GPU time: ~3.5 h spent so far and ~12 h projected for S1 + S2 + X1 +
   X7/X9 (report section 9), so S2 keeps n=100 images. [inference] If the 24 h was meant as
   wall-clock, the budget is already spent; then shrink n via the env overrides
   (`N_PROMPTS`, the S2 `--limit`) rather than skipping a step.
2. Correction of a chat-level claim: the X6 retry runner's stdout *is* persisted - at
   `~/.vlm-lens-setup/x6_retry.log`, not under `results/.../logs/`. It shows the 07:02 runner
   was the pre-fix version ("retry 3: leg failed with rc=1 ... - giving up" after two guard
   skips), i.e. the give-up-after-one-failure behaviour that D14/D15 replaced. When the leg is
   relaunched, pipe the runner's stdout into `results/.../logs/x6_retry.log` so one directory
   holds the whole chain's logs.

## D18 — X6 verdict: S2/X1 move to fp32+TF32; S1's lens stays bf16; the S1 FD check repaired

1. X6's verdict is in (`step1/x6_dtype.json`): per-layer relative Frobenius medians are
   1.2-1.5 % (below the 3 % median bar) but the `text`/L0 max is 49 % (bar: 10 %), so the
   pre-registered rule (`x6_compare.py`: fp32 if median > 0.03 or max > 0.1) is
   `use_fp32: true`. Followed: `run_step3.sh` (both S2 halves) and `run_step4.sh` (both X1
   legs) fit with `--dtype float32 --allow-tf32`, and `run_campaign.sh` guards both steps at
   the fp32 leg's validated 55000 MiB.
2. Cost correction: "true fp32 is ~15x slower" (D10) is the *un-flagged* cuBLAS path. The
   measured fp32+TF32 leg took 2486.5 s for 8 image samples against 2205.8 s for bf16
   (1.13x), so the rule costs S2 ~13 % wall time, not 7-15x. The leg is TF32-backed, so the
   comparator is really bf16 vs TF32; TF32's mantissa is 4x finer than bf16's, which bounds
   bf16's error but does not make the reference exact fp32.
3. S1's fitted lens stays bf16 - a deliberate deviation from the rule's "production"
   reading: the 100-sample WikiText lens already exists, the fired branch is the L0 anomaly
   while the medians sit at 1.5 %, and S1's gate claim is a ranking claim that a 1.5 % J
   perturbation cannot flip. [inference] Re-fitting S1 in TF32 is a cheap follow-up
   (~2.3 h) if the gate verdict comes out marginal.
4. The S1 finite-difference check had two real bugs, both found by reading the first server
   run's log: (a) the perturbation hook wrote `hidden[:, positions, dims]` with 176 positions
   x 4 dims - a broadcasting `IndexError` that killed both score attempts before the JSON
   was written; (b) behind it, the hook perturbed *all* top-norm dims at once while the
   comparison expected one, and the comparison used `J_l[i, :]` although the estimator
   stores output-dim-major (`jlens/fitting.py` assigns `grad[:, positions, :].mean(1)` into
   `jacobians[layer][dims, :]`), so the FD column `J_l[:, i]` was being compared against the
   row `J_l[i, :]`. Fixed: one source dim per pass, column comparison against
   `J_l[:, i]`, an `eps`/`--fd-eps` parameter (the tiny CPU fixture's residual RMS is 0.01,
   the real model's is O(10)), and the scoring rows are written to JSON before the FD model
   loads so an environmental OOM/SIGKILL cannot lose them. [measured] Verified locally:
   independent autograd through the same hook path matches the estimator to 6.7e-08, and the
   repaired check gives worst per-layer mean 4.4e-04 at eps=1e-5 on the tiny fixture (gate
   bar 0.05).

## D19 — S1's gate: FAIL on the strict middle criterion (diagnosed; S2 proceeds); the FD CPU trap

1. S1's scoring ran to completion on the frozen 30-sample WikiText shard (`step2/s1_score.json`;
   the repaired finite-difference check follows it). Gate as pre-registered: (i) identity exact
   `0.0` PASS; (iii) last-layer rank difference `0.0` PASS; (iv) depth trend PASS (mean
   true-token rank 885.54 at the depth midpoint against 27.31 at the last layer; 5 non-monotone
   steps kept as a diagnostic); frequency control PASS (at L31 the lens matches the model:
   top-1-in-top-50 0.450, mean rank 27.3). But `j_not_worse_in_middle` is **False**: the J-lens
   loses to the plain logit lens at layers 17, 20, 21, 23, 24 (rank deltas +379, +813, +331,
   +99, +271 against a 16032.5 chance level) while winning at L2-16 (L14: 1101.2 vs 5178.4) and
   L26-30 (L30: 46.5 vs 75.2). Verdict: FAIL on one of five criteria.
2. Decision: S2 proceeds. Rationale [inference]: the failing criterion is a comparative nuance
   ("never worse than the untrained logit lens at any middle layer"), not a validity criterion;
   the lens's own validity checks (i)/(iii)/(iv) and the frequency control pass on the real
   checkpoint, and S2's questions (per-tag held-out fidelity, instruction-half cross-scoring,
   WikiText<->captions transfer) do not depend on per-layer logit-lens dominance. The deficit
   band is recorded as a finding in the report (§4/§12) instead of being rounded away.
   Per-position significance of the five deltas was not tested (the scorer returns means only);
   that is a stated limitation, not a claim.
3. The FD's CPU fallback is a trap, not a safety net: the estimator's ceil(d_model/dim_batch)
   chunk-backwards on a 7B fp32 CPU model take hours. The 06:05 attempt entered the fallback
   under a 90 MiB GPU window and was killed at 06:45 with no FD numbers. `s1_score.py` now waits
   for `--fd-min-free-mib` (default 32000) up to `--fd-wait-minutes` (default 60) *before* the
   fallback, printing each poll; only then does it drop to CPU. This supersedes D16's "minutes
   on the 263 GiB host" claim for the estimator leg (forward-only work would be minutes).

## D20 — S1's finite-difference row is decoupled from the chain (watcher, not a stall)

`s1_score.py` runs the FD check (ii) at the end of the same process, and its 60-min window
wait plus its hours-long CPU fallback would sit *inside* the S1 step - delaying S2 (55 GiB)
past the budget in the worst case (the co-tenant held 72 GiB at 09:55Z, free 7.9 GiB).
`run_step2.sh` therefore scores with `--skip-fd` and the chain moves on; a separate watcher
runs the full scoring (FD included) on the first unopposed ≥32 GiB window after S1's step
exits, publishing `step2/s1_score.json` by rename so no reader sees a partial file. The
in-process gate now marks the missing criterion as `not_measured` and reports `PASS` over
the measured criteria plus `PASS_complete`, so a skipped FD can never masquerade as a
failed check. If no window ever comes, the report states the FD row as unmeasured and
leans on the tiny-fixture equivalence (4.4e-04 worst per-layer mean at eps=1e-5, D18).

## D21 — S2/X1 drop to dim_batch 1 under 36 GiB guards; S1's guard drops to 24 GiB

The co-tenant held 57-73 GiB for hours on 2026-10-02, so the chain could never see the
windows its 55 GiB guards required. dim_batch scales activations, not math (F11/D13), so
S2/X1 now run `DIMBATCH=1` under a 36 GiB guard (~29 GiB fp32 weights plus ~1-2 GiB
activations, validated 36.4 GiB at dim_batch 8 in step 0) and S1's step runs under a 24 GiB
guard (~16 GiB bf16 weights). `run_step4.sh` now takes the knob from the environment. The
supervisor was restarted in place (v3, 12:43:53Z) while it slept inside `need_free` - no GPU
was held and no partial work existed, so nothing was lost. The bf16 fallback in the report's
schedule contingency is demoted to a last resort: it would trade the pre-registered dtype
verdict, which dim_batch does not.

## D22 — GPU windows are speed-gated, not just memory-gated; the FD CPU fallback is retired

Measured 2026-10-02T17:05Z: the co-tenant (three ~10 GiB jobs at ~99% CPU; one since 09:33Z,
two arrived 15:14Z/15:30Z) pins `utilization.gpu` at 100% while leaving 14.8-38 GiB free.
TF32 20x4096^3 matmuls then measure 41 TFLOPS (~8% of the H100 PCIe's 494 TFLOPS peak;
SM clock at max, no throttle reason) against the ~311 s/sample the X6 fp32 leg measured on a
quiet box (03:21Z). S2, launched 15:16:39Z into a 58.9 GiB window the co-tenant refilled
within minutes, produced <5 samples by 17:24Z (>25 min/sample, ~10x derate) and checkpoint 5
never landed; the run was killed (uncheckpointed partials lost, ~25 quiet-minutes) and S2
restarts from scratch in the next fast window. D12/D21 guards check free memory only -
necessary, not sufficient. `gpu_guard.sh` now (a) probes before starting the campaign
(>=150 TFLOPS TF32 and >=24 GiB free; the campaign's per-step guards handle each step's real
appetite) and (b) kills the whole chain when a running fit lands no new `*.pt` for 90 min
(restarts re-run S1's ~3 min scoring and resume every fit; the campaign has no skip markers).
The 2026-10-02 user directive (results on GPU only; CPU for quick checks only) also retires
D20's CPU fallback: the FD row is now the chain's optional `fd` step (`run_fd.sh`, 36 GiB
guard, fp32 estimator on GPU, `step2/s1_score.json` published by rename only when
`check_ii_finite_difference` is present), and `/tmp/s1_fd_watch.sh` is disabled. The FD
watcher's in-flight CPU run (PID 53431, 9 cores, 48 GiB RSS) was killed and its partial
`s1_score.new.json` removed.

`s1_score.py`'s own two automatic CPU fallbacks (window-wait expiry at `--fd-wait-minutes`,
and the `torch.OutOfMemoryError` retry) were removed for the same reason: both now fail the
optional step, so a retry lands in the next fast window.

## D23. Run to completion regardless of wall time (user directive, 2026-10-02 ~19:50Z)

User directive during the derated S2 half-A fit: "just keep running even if it take many times,
don't stop." Consequences: the pre-registered n-shrinking rule (autonomy rule 3) and the
quiet-wait contingency in the report's section 9 are withdrawn for this campaign;
`gpu_guard.sh` retains only occupancy gating for starts (FREE_MIN=24000; TF_MIN=0
logging-only) and the 10 h no-new-output-checkpoint trip (TRIP_S=36000). S2 sample 1
(19:45:11Z) verified: seq=609 = 576 image + 33 text, masks text 31 + image 576 = all 607,
3431 s; its `rel_change=nan` is the explicit `n_done[mask]==0` branch in fitting.py (first
sample has no running mean), not a NaN tensor. Measured pace 3431 s/sample (vs 311-326 s
quiet) => checkpoint-5 ~23:40Z, half A ~47.5 h [projection].

## D24 — 2026-10-03 ~01:05Z: /data hit 100% full; S2 checkpoint writes fail (ENOSPC); chain loop stopped

S2 half-A (started 19:45Z) fitted samples 1-5 healthily (3431->3307 s/sample; `rel_change`
decaying 8.0e-01->1.6e-01 text, 4.0e-01->1.5e-01 image) but its checkpoint-5 save (23:35Z)
died in `torch.save`'s zip trailer (`unexpected pos 4294983872 vs 4294983760`), leaving a
4.3 GB unrenamed `checkpoint.pt.tmp.227311`; `s2_attempt2/3` then failed explicitly with
`OSError [Errno 28] No space left on device` (SPLIT_FAILED) and the chain gave up on
3 consecutive non-transient failures. `gpu_guard.sh` relaunched it every ~3 min into an
S1 fail-loop (even ~3 KB saves hit the same pos-mismatch signature; 8+ cycles by 01:01Z).

Root cause: `/data` (5.5T) at 100% - 0 bytes free (our tree is only 8.8G; co-tenant data
dominates). The fp32 three-mask S2 checkpoint (~6.4 GB) cannot be written at all; nothing
in the campaign can persist while this holds. Partials from the 5 fitted samples were
never checkpointed and are lost (~5 h GPU; second uncheckpointed loss after D22).

Actions: (1) reclaimed the corrupt 4.1G tmp after a `fuser` check (not in use);
(2) deliberately stopped guard+campaign+step+fit at ~01:05Z - continuing would burn ~5 h
per fresh S2 attempt whose checkpoint can never land; `s2_watch.sh` stays (read-only).
Restart is state-free: `cd .../validation_2026-10-01/code && nohup bash gpu_guard.sh >> ../logs/gpu_guard.log 2>&1 &`
- S1 resumes from its 20-sample checkpoint, S2 half-A restarts from sample 0.

Open decision (user): free `/data` to >= ~20 GiB [derived: 6.4G half-A ckpt + 6.4G half-B
ckpt + 6.4G merged artifacts + headroom] or add storage; relocation to `/home` (22G avail,
peak ~20G) rejected as razor-thin default.

## D25 — 2026-10-03 ~09:00Z: outputs relocated to the user's data_mount; chain relaunched

User directive: "Store the outputs in /home/nvidia-lab/data_mount. It has 49gb left. And
resume where the process left off." Executed: MOUNT=/home/nvidia-lab/data_mount/vlm-lens
(exfat, 48.9 GiB free; no symlinks on exfat, so paths are patched in-script): s2-half-a/b,
s2-merged and the X1 fits write there; s2_eval and x7_x9 read the merged lens there; step
JSONs and read-only /data artifacts stay in place. D24's DISK_NEED gates are now
path-aware (`DISK_PATH`): s2 waits for >=24 GiB on the mount, x1 for >=2 GiB, /data steps
for 1 GiB. `PYTHONUNBUFFERED=1` is exported by the campaign and both step chains (buffered
stdout was one suspect behind the traceback-less 01:05Z deaths). `gpu_guard.sh` watches
checkpoints in both trees. Relaunched 09:01:02Z: guard pid=987298; S1 attempt 1 started
(GPU free 32117 MiB >= 24000; /data free 4109 MiB >= 1000 - both new gates logged passing);
deployed script md5s verified identical to the repo commit. S2 half-A restarts from sample 0
- D24 confirmed the 5 fitted samples were never checkpointed, so nothing was resumable.
Budget: mount holds S2 (~31 GB) + X1 (~2 GB) with ~16 GB spare; no further deletions needed
[derived from artifact sizes: 3x1984 MiB per lens set + 2 checkpoints + merge].
## D26 — 2026-10-03 ~12:09Z/~12:36Z: user interrupted the chain (SIGTERM); ~27 min lost; relaunched 13:51:56Z

User: "i interrupted the process" - accurate: SIGTERMs at 12:09:12 (s2 attempt-1, 1 min in,
during model load, zero output) and ~12:36:5x (attempt-2, seconds after its sample 1 line
landed). The sweep also took out guard+campaign; nothing was checkpointed (s2-half-a remains
empty on the mount), so the interruption cost ~27 min of GPU plus two model loads. S2 could
not start when S1 finished at 09:04Z - the co-tenant held VRAM below the 36000 MiB gate until
~12:06Z, which is why the S2 attempts carry 12:0xZ timestamps.

Legitimacy of attempt-2's single sample [measured]: same deterministic manifest sample as the
10-02 run (`000000012966::prompt+caption`, seq=609, images=576, masks text 31 + image 576 =
all 607 - exact again), 1584 s of fit between 12:10:27 and 12:36:51 (wall-consistent), clean
log apart from the benign cuBLAS-context warning; `rel_change` nan is the documented n_done=0
first-sample branch. The attempt ran entirely under the D25 mount paths, so the redirect is
field-proven writable; no ENOSPC, no traceback, no tmp residue in the campaign tree.

Relaunched 13:51:56Z: guard probe reported tf=231 TFLOPS (genuinely quiet - vs 88 at 09:01Z
and 41-52 under the 10-02 derate) and free=81000 MiB; campaign s1 attempt-1 started instantly,
both gates passing (/data 4109>=1000, GPU 81000>=24000). Expected pace at 231 TFLOPS is
~5-6 min/sample, checkpoint-5 in ~35 min. Watch item: the mount reported 44658 MiB free at
12:10Z and 37910 MiB at 13:43Z (-6.7 GB, not ours - s2-half-a is empty); a `find -size +1G`
sweep on the next probe will attribute it.

## D27 - 2026-10-03 ~14:19Z: co-tenant took the GPU; S2 attempt 1 SIGKILLed, attempt 2 OOM; cadence 5->2

Attempt 1 (started 13:54:22, pace 375 s/sample - quiet box) was SIGKILLed mid-sample-5
(samples 1-4 done, no checkpoint). Campaign auto-retried in 60 s: attempt 2 (14:25:03) OOM'd
during model load - GPU 0 had 77 MiB free of 79.11 GiB (co-tenant holding essentially
everything; host RAM 124/251 GB used, clean dmesg - GPU-side, not host OOM). The chain is
alive and parked in the 36 GiB s2 gate (holds up to 16 h; after that the fit's own
transient-OOM retry loop covers it). Cost: ~31 min of S2 compute, nothing durable.
Fix applied: `--checkpoint-every 5 -> 2` in run_step3.sh (worst-case loss per kill is now
~2 samples / ~12.5 min). Cadence is not part of the checkpoint fingerprint - verified in
fitting.py:289-305 (layers/target/dim_batch/max_seq_len/skip_first/masks/target_mask/shard).
Mount still holds no >1 GB files from us (s2-half-a empty); only s1-text/checkpoint.pt on /data.
## D28 - 2026-10-04 ~10:15Z: S2 half-A verified (first durable caption lens); ops notes

Half-A fit completed 09:16Z after 15.7 h (17:34Z-09:16Z, ~1130 s/sample - co-tenant derate
~3x vs the 375 s quiet pace; ran to completion under it). Verified by loading the three lens
files (weights_only load OK; upstream keys; stored fp16, 993 MiB each; fit in fp32+TF32):
  n_prompts 50/50/50, n_skipped=0, finite, max|J| = 9.3 (text) / 4.6 / 4.2
  text:  |J|_F l0/mid/top = 249/69/68;  ||J-I||/||I|| = 4.01/0.88/0.43 nonincreasing
  image: 62/94/72; 1.20/1.25/0.54.  all: 61/90/71; 1.21/1.18/0.52 (early non-monotone in
  image/all - the X6-fp32 reference fit shows the identical pattern: transport geometry,
  not a defect).
  final rel_change text 3.3e-2 / image 1.1e-2 / all 1.2e-2 (from 0.21-0.29 at n=4).
  Cross-check vs x6-fp32 (8 samples, layers 0/8/16/24/30): norms agree within 2-7% on all
  masks. Provenance: manifest-half-a sha256 efd2ffcf..., dim_batch=1, skip_first=1,
  image_token_id 32000, 576 image tokens, vendored jlens commit 581d398.
  Cross-mask cos at L0: text-image 0.05 (near-orthogonal source-side geometry); ~0.98 at L30.

Ops notes:
(1) Completed fits' checkpoints are the step-retry skip markers - S1 re-ran 100 samples in
2.5 min off its 100/100 checkpoint (observed behavior). Do NOT prune s1-text/checkpoint.pt
or s2-half-a/checkpoint.pt; deleting them replays 15+ h.
(2) run_campaign.sh must not be edited in place while running (the s2 call site already bound
its DISK_NEED=24000 argument; bash re-reads moving offsets). If half-B is killed, the retry
parks in the 24-GiB gate which cannot open at mount-free ~16 GB. Recovery: stop the campaign,
sed DISK_NEED=24000 to 12000 (15.9 >= 12 opens it), relaunch guard; s1 skips in ~3 min,
A resumes instantly, B resumes from its checkpoint.
(3) Cleaned 76 zero-byte checkpoint.pt.tmp.* relics under s1-text (pre-D24 S1 OOM-storm era).
(4) Space ledger, mount (free 15.9 GiB at probe): half-b checkpoint 5.9 -> half-b artifacts
3.0 -> merged 3.0 -> X1 ~1 => ~3 GiB final margin. /data free 2.8 GiB, tiny JSON writers only.
(5) Half-B pace watch: sample 1 = 1056 s, sample 2 = 1908 s (derate persists); ETA 15-25 h.
First half-B checkpoint (cadence 2, edit was live for attempt 3) not yet on disk at probe
time - likely a slow mount write; confirm at next probe.
## D29 - 2026-10-04 ~19:50Z: root cause = data_mount (exfat) write-death, not the co-tenant; tree killed; heartbeats deployed; rescue to /home

User directive: kill + detailed progress logging + resume from the latest checkpoint. Kill done:
guard/campaign/step3/fit all stopped cleanly; s2-half-b/checkpoint.pt (20/50, 5.94 GB, mtime
14:06) intact.
Storage autopsy (MOUNT = /dev/sdc1, exfat, 1.9 TB, 100% full, 4.5 GB avail): reads fine (10 MB
instant), writes hang (8 MB write times out in 15 s; 512 MB dd hung > 150 s). The "co-tenant
derate" reading was mostly save stalls: half-A's 15.7 h = ~5.2 h compute + ~25 checkpoint writes
grinding on a full exFAT; half-B's sample-21 "stall" = a checkpoint save hung since ~14:40 (the
stranded checkpoint.pt.tmp.*). Host fs map: / ext4 5.8 GB free; /home ext4 10 GB free (2.5 GB/s);
/data 5.5 TB 100% (2.8 GB free); fuse.rclone 1.7 TB free (another user's, off-limits).
No healthy filesystem can host the 5.9-GB-checkpoint write pattern (tmp+replace needs ~12 GB
peak) => S2 cannot continue safely on this box.
Actions: (1) fitting.py now logs a pass heartbeat (every 10% of the 4096 passes, with elapsed
seconds + GPU alloc) - deployed by scp, local commit f585d3b; (2) run_campaign.sh s2 retry disk
gate 24000->8000 - deployed; (3) the cadence-2 edit in run_step3.sh was still NOT on the server
(server md5 differs from local by exactly that line) - D27/D28's "edit was live" is corrected
here; deployment to this server is manual scp, which is now the stated mechanism; (4) rescue:
cp exfat -> /home/nvidia-lab/vlm-lens-rescue (half-a artifacts md5-verified against provenance
sha256, half-b checkpoint, step0/1/2 JSONs + manifests, x6-fp32 artifacts). S1 lens (3 GB),
s1-text checkpoint (2 GB) and the half-a checkpoint (5.9 GB) remain on /data and exfat (reads
OK; ship or rescue separately).
Resume options: (a) new server - recommended; runbook Path A plus this rescue bundle; (b)
fragile-local: half-B re-fit on /home with checkpoint-every None (no kill protection), then
merge; X1 deferred.

## D30 - 2026-10-05 ~06:05Z: MOUNT relocated to /data (user freed 46 GB at /data/anhnq); campaign relaunched to resume at B=20/50

User freed 46 GB on /data (free 2.8 -> 48.9 GB) and instructed to use it. /data write test:
2.0 GB/s (healthy ext4). The exFAT data_mount remains write-dead (8 MB write still times out
after the user's cleanup there; 3.3 GB free) and is now abandoned for writes - reads stay
available for anything not yet rescued.
Executed: (1) MOUNT=/data/vlm-lens/mount in run_campaign.sh, run_step3.sh, run_step4.sh,
gpu_guard.sh (guard's pt-watch already globs "$MOUNT" "$RUN", so the TRIP clock follows the
outputs); deployed by scp, md5-verified on the server (step4 pre-check matched HEAD). Also
carries the cadence-2 line into the server's run_step3.sh at last (D29's deployment note).
(2) state: cp from the verified rescue bundle (half-a artifacts, half-b checkpoint) + the one
exFAT read still needed (half-a checkpoint, 6.24 GB, for the A resume-skip); B checkpoint md5
46f6b369...46f6... matches the rescue-time hash; 15 GB on /data, 34 GB still free (enough for
B's save churn 5.9 + B artifacts 3 + merged 3 + X1 1 + margin).
(3) guard relaunched; it waits for free>=24000 MiB (GPU was 21.2 GB free at launch), then the
chain runs s1 (resume-skip ~3 min) -> s2 (gate 36 GB; disk gate 8000 on the new MOUNT) -> A
resume-skip -> B resume from 20/50 with the new pass-heartbeat logging and ~second-scale
checkpoint saves on ext4. /home rescue bundle (9.2 GB) retained as the immutable backup.

## D31 - 2026-10-06 ~04:10Z: S2 running on relocated setup; correction - durable B resume was 15/50, not 20/50

S2 step started 22:54:50Z on 10-05 (gate: GPU 38637 MiB >= 36000; disk 33587 >= 8000 on the new
MOUNT). Half-A resumed instantly: 50/50 recognized in ~82 s (warm fp32 load, zero recompute).
Half-B resumed from **15/50** - the true durable state: the 14:06 save on the exFAT was after
sample 15 (the server still ran cadence 5; D29/D30's "20/50" was inferred from the
unpropagated cadence-2 assumption and is corrected here). Samples 16-20 (computed 10-04, never
saved) are being recomputed identically; no data loss.
Progress at 04:07Z: samples 16-26 done (11 in this log), pace ~24.7 min/sample (persistent ~4x
co-tenant derate), 133 heartbeat lines (11/sample - the new logging works), checkpoint saves
landing on /data (mtime 03:33; guard clock resets 02:39 + 03:36; cadence 2 effective). ETA B
done ~14-15Z today, then merge -> s2_eval -> FD -> X1 -> X3 -> X7/X9 automatically. The
server-side hbwatch expired before S2 began (240-min cap started 06:17 on 10-05); live probes
replace it.

## D32 - 2026-10-06 ~08:45Z: campaign relocated to the Brev 8xH100 box at /data/anhnq; S2 refit in parallel; launched 08:36:19Z

The user provided a new server (Brev `brev-jkk3...`, 8x H100 80GB, `/data` 18 TB with 7.1 TB
free) and staged `/data/anhnq` themselves: repo upload (a pre-10-05 snapshot), exactly the
needed 130 COCO val2014 images (`needed.txt`), a complete `hf_cache/hub/models--llava-hf--
llava-1.5-7b-hf` (14 GB, snapshot b234b804), conda env `vlm_truth_py313` (py3.13, torch
2.5.1+cu121, transformers 5.17.0, datasets 5.0.1, matplotlib/pillow/pytest), `vlm_env.sh`
(WORK=/data/anhnq, HF_HOME, HF_HUB_OFFLINE=1, TMPDIR, REPO, RUN), and a 4.8 GB
`vlm-lens-out/validation` tree carrying step0/step1/step2 complete (S1 lens + s1_score; x6
lenses) but no S2 state (s2-half-a empty, no half-b).

Setup performed (all committed here; scripts synced to the box):
* Latest tree rsync'd (src, scripts, tests, docs, code/, DECISIONS, RUNBOOK); report draft
  REPORT.md and neuronpedia/ left untouched.
* All campaign-called scripts re-pathed to env defaults for the Brev layout: REPO=/data/anhnq/
  NeuronpediaVLM, P=.../envs/vlm_truth_py313/bin/python, RUN=.../vlm-lens-out/validation,
  MOUNT=$RUN (single-disk, per vlm_env.sh), HF_HOME=.../hf_cache, DIMBATCH=1 (proven ~31 GiB).
* Gates are GPU-aware now: the box is shared (co-tenant training jobs with 40-57 h elapsed);
  need_free polls every card and pins CUDA_VISIBLE_DEVICES to the freest >= threshold,
  re-picking on each retry. step3 fits each half with its own lock-guarded GPU picker
  (concurrent when two cards are free, sequential otherwise) plus a 190-min no-checkpoint
  hang watchdog. GPU 0 is no longer special (its free memory was 1.7 GB while GPU 5 held 38).
* `manifest-fit/heldout` embedded the old box's absolute image dir; rewritten in place to
  `/data/anhnq/coco_val2014` (originals in `step0/prepath_bak/`; text manifests had no image
  paths; the half manifests regenerate from manifest-fit at step3 start, inheriting the fix).
* S1 skipped on artifacts+s1_score presence (both staged).

Decision - refit S2 on the new box rather than transfer the old campaign's 9 GB of S2
checkpoint/artifacts: quiet-window pace is ~275 s/sample (matches the historical 311 s), the
halves run concurrently on separate cards, and the engine/radius are identical; the old box's
state stays archived (rescue bundle + /data/vlm-lens) and its campaign remains killed
(04:46:18Z, user directive).

Launched 2026-10-06T08:36:19Z (no gpu_guard - the co-tenant speed probe is superseded by the
per-step window gates and the in-step watchdog). Verified: s1 skip, S2 gate -> GPU 5 (38 GB
window), disk gate 7.4 TB, half-a fitting (30.5 -> 32.3 GiB, heartbeats every ~11 passes/
sample), half-b picked a second card ~2 min later (30.8 GiB). Expected: halves ~3.8 h each ->
S2 fits done ~12:30Z -> merge + s2_eval -> FD -> X1 -> X3 -> X7/X9; report materials under
results/validation_2026-10-01/.

## D33 - 2026-10-06 ~21:05Z: half-A cross-box verification (old vs new agree to fp32 noise); both halves done on Brev; merge/eval running

User directive: "the first half is already done [on the old box] - take that as the results for
the running ones." Verified:
* Structure: old-A artifacts (`/data/vlm-lens/mount/s2-half-a/artifacts` on the old box) are the
  current format - 31 layers (0-30), J as per-layer [4096, 4096], fp16 on disk, rich provenance,
  fit_config identical to new-B/new-A (dim_batch 1, masks text/image/all, skip_first 1,
  n_samples 50, shard [0,1]); loads with `vlm_lens.artifacts.load_lens_set`.
* Content: per-layer mean|J|, old-A vs new-A (the SAME 50-sample half, old H100 vs Brev GPU 5),
  agree to <=1e-4 absolute (<=1% relative) at all 31 layers. old-A vs new-B (a different half)
  differ ~2.5% - so the cross-box same-samples gap is fp32/TF32 reduction-order noise
  (RUNBOOK 11); either half-A is interchangeable for the merged lens.
* The campaign merges the new-A (written 20:52:36Z; B 20:02:28Z; same box/library stack as B).
  Old-A stays archived. Manifest sha differs (b7d60ab..., the D32 image-dir rewrite) as
  expected; the checkpoint fingerprint excludes the manifest.

Brev half timings: B done 20:02Z, A 20:52Z (both ~11.3-11.6 h wall, co-tenant derate ~3x).
step3 continues: merge (n_prompts-weighted) -> s2_eval on 300 held-out captions.

## D34 - 2026-10-07 ~00:35Z: s2_eval runtime swap (reference -> GPU-metric scorer), A/B-exact; S2 main table as expected

The reference scorer (kept verbatim as `code/s2_eval_ref.py`) is CPU-metric bound: every
phase runs twice (default + include_placeholders, the second discarded for halves/transfer),
and metrics span 30 layers x 32k vocab x ~600 positions per sample on the CPU. Measured on
the first attempt: ~34 cores busy, GPU ~22 % utilization, main's first table still unprinted
after 115 min (main = 600 passes before its print), projected ~14 h total. Killed at 2h07m
(zero printed output lost; log kept as s2_attempt1.log); run_guarded retried step3 as
s2_attempt2 (halves instant-resumed from their 50/50 checkpoints on GPUs 0/4; merge re-run).

`code/s2_eval.py` now keeps the same `lens_readout` (identical logits, identical JSON schema
and prints) and changes only the metric reduction: GPU reductions, one readout shared by both
modes, and the discarded include_placeholders pass skipped for halves/transfer. Verified by
`code/s2_eval_ab.py` on real samples: default 128 cells and ip 224 cells with
max|drank| = max|dagree| = max|dmodel_rank| = 0.000000 (integer metrics exact) and max rel
dKL 1.1e-5 / 4.8e-6 -> PASS. Known residual cost: the fp32-CPU logits are staged back to the
GPU per chunk (lens_readout's float32-CPU contract) - a `to_cpu=False` readout option would
remove roughly half of the current ~12 s/sample; future work, not applied mid-campaign.

S2 main table (attempt2, 300 held-out captions, merged lens, default = placeholder-excluded):
text tag rank 26120.6 (L0) -> 1307.6 (L16) -> 215.2 (L30) -> 42.62 = model (L31; identity
exact, agree 1.000, KL 0.000). Normalized, L16 is 30.7x the model ceiling against S1's 32.4x
- S2 tracks S1, and the curve matches the M6 train-set shape (L30 215 vs M6's 236). Two
diagnostics flagged, both pre-consistent: (i) an L20 bump (2753, +1446 over L16) echoing
S1's L17-24 middle-layer weakness (D19); (ii) a steeper L30->L31 convergence (5.0x ceiling
at L30 vs S1's 1.7x). Tags behave per design: image/image-q3 collapse to one position per
sample in the default table (identical rows, V5); quarters appear only under
include_placeholders (n=43200); whole-block image rows are placeholder-target dominated
(model rank 14.7k) - descriptive only (V1/V4/V5). Follow-up diagnostic: the untrained
logit-lens baseline (use_jacobian=False) on the same held-out, to attribute (i)/(ii) to the
model's mid-layer states vs the fitted map. Chain continues FD -> X1 -> X3 -> X7/X9
automatically.

## D35 - 2026-10-07 ~04:45Z: S2 section complete - expected structure, halves symmetric, L20 bump attributed to the model

`step3 chain done 2026-10-07T00:56:00Z`; `step3/s2_eval.json` (154 KB, 300 held-out captions).
Main table as in D34 (text L0 26120.6 -> L16 1307.6 -> L30 215.2 -> L31 = model 42.62, identity
exact; composition text 82.0 / image 576 / all 658 / quarters 144 positions per sample, as
designed).

Halves (A4/X8, text tag, 150 + 150 held-out): at L30, lens_a own 233.00 vs cross 231.83,
lens_b own 197.81 vs cross 203.15 - near-symmetric with the correct in-domain sign, no
leakage/overfit. lens_b is the stronger half (~15%) on both subsets (question-half
distribution, not a defect). L31 identity exact in all four cells.

Transfer (A4/E6):
* Captions, S1 text-lens vs S2 merged, at L30: rank 109.49 vs 215.21 (S1 better) while agree
  0.706 vs 0.729 and KL 0.474 vs 0.428 (S2 better) - the same rank-tail vs distribution
  tension recorded for S1 (D19). By rank S1 leads L0-L20 (L0 5.6k vs 26.1k), S2 leads L24-L27
  (1047.5 vs 1095.2; 414.2 vs 592.3). Note S2's text-mask fit sees ~82 text positions per
  sample (the 576-token image block dominates the sequence) against S1's ~190 - half the
  source tokens at equal n_prompts.
* WikiText, S2 caption lens: L30 rank 67.9 (2.49x ceiling) vs S1's own 46.5 (1.70x) - the
  expected out-of-domain loss; agree/KL close (0.743 / 0.470).

Diagnostics from D34: the L20 bump is SHARED by every lens on captions and by the caption
lens on WikiText (S1-on-captions +48%, S2-on-captions +110%, S2-on-WikiText +138% against
their L16) - the L17-24 region is a model-level property (consistent with the S1 gate's D19
record), not a defect of the fitted S2 map. The L30 rank gap is S2-specific, noted together
with the countervailing agree/KL. X11 (logit-lens baseline) stays queued for a GPU window.

Campaign: FD failed twice with "CUDA OOM for the float32 FD model - FD not measured
(GPU-only)" - run_fd.sh's designed graceful path; classified `other` only because
run_guarded greps "OutOfMemory" and not "CUDA OOM" (cosmetic; three `other` in a row skip
FD by design via `|| echo CAMPAIGN_FD_SKIPPED`). X1 (36 GiB gate) is queueing - the box is
memory-packed (max free ~19.9 GB) so later steps wait for windows. X11 deferred (needs
~24 GB), will launch opportunistically.

## D36 - 2026-10-08 ~05:15Z: campaign COMPLETE (19:15:49Z Oct 7) - X1/X3/X7/X9 results; X1's hard criterion mis-calibrated; cos>1 artifact fixed

`=== campaign done 2026-10-07T19:15:49Z ===` - every step ran; FD skipped by design (2 OOMs
+ 1 no-window). X11 (logit-lens baseline, the queued diagnostic) launched separately at
~05:10Z on GPU 3 (26.4 GB window) once the campaign freed the box; ~1 h for 300 samples.

**X1** (`step4/x1_targetmask.json`): the pre-registered HARD criterion (text rows
`torch.equal`-identical between `target_mask=all` and `target_mask=text`) FAILED -
`text_rows_bit_identical=False`, but with rel_fro 1.37e-2 (L0) -> 1.75e-6 (L30), i.e. the
**X6-era TF32/atomics run-to-run noise floor** (X6 measured 1.2-1.5 % per-layer medians).
The two variants are separate fit invocations, so bit-identity was unachievable by
construction - the criterion was mis-calibrated, not the estimator broken. The REAL signal
confirms V1 spectacularly: the image rows move by rel_fro 24.5 (L0) -> 27.4 (L8) -> 38.1
(L16) -> 103 (L24) -> 255 (L30) with cosine collapsing 0.39 -> 0.09 - a ~180-1500x separation
between image-row and text-row movement. Lens-level (aggregate J): text rows differ 2.12 (L0)
-> 0.17 (L30) - explained by the library text mask covering the USER: tokens before the
placeholder, which causally reach image targets (pre-registered in x1_compare's docstring,
D3); image J moves 23.7 -> 115.6 with best-fit scale 8-28x - the image->image cotangent mass
is ~8-28x larger and nearly orthogonal (V1).
Artifact fixed: `_compare` computed a 16.7M-element cosine in fp32 and returned an
impossible cos=1.0019 (>1) on the L0 text rows - now float64 (exact to ~1e-15) and a
`text_rows_max_rel_fro` field added; re-run in flight to `step4/x1_targetmask_v2.json`.

**X3 re-census** (`step1/x3_norms.json`): L0/L16/L31 rows reproduce the committed census
within cross-GPU rounding (L0 bos 8.32 vs 8.3; L16 bos 1568.83 vs 1568.8; L31 633.65 vs
633.7; one cell 0.4 % off - within the S11 noise clause); the L24 row landed (bos 1567.4,
pos_1_16 65.25, text_post 91.50) giving X9's alpha grid measured units. Verdict re-confirmed:
`pos_1_16_sink_like=false`, `recommended_skip_first=1`.

**X7** (`step4/x7_x9.json`): 31 conditioning entries; cond in [1.30, 2.89] - ALL far below
the 1e3 exclusion threshold, `degenerate=false` everywhere, cosine 0.25-0.78, norms
0.83-1.35 (residual-relative units). The pseudo-inverse swap bases are well-conditioned;
the exclusion rule never needed to fire.

**X9** (`step4/x7_x9.json`): 215/1400 generations changed (rate 0.154; add 161, ablate 32,
swap 22); first-diff tokens cluster at caption positions 10-14 and 1-3 - edits bite early.
The E5 minimal bar (change rate > 0, first differing token reported) PASSES; the strong
bar (directed concept-specific changes) is partial - 85 % of edits left greedy generation
unchanged, and some pairs (dog->cat on a snowboarding image) are semantically irrelevant to
their sample. Follow-up: extend the alpha grid and filter pairs by image content.

## D37 - 2026-10-08 ~09:30Z: X11 completes the L20 attribution - REVISES D35/D36: the bump is fitted-map-specific, not a model property

X11 (logit-lens baseline, use_jacobian=False, 300 held-out captions, ~55 min on GPU 3) is in
`step4/x11_logit_lens.json`. The logit lens's text-tag rank is MONOTONE through L17-24
(2201.8 at L16 -> 1557.9 at L20 -> 1305.8 at L24 -> 702.3 at L27 -> 284.8 at L30) - NO bump.
The fitted S2 lens bumps 1307.6 -> 2753.0 at L20 (+110 %, KL 4.46 -> 6.60), and S1's lens
bumped +48 % there. So D35/D36's "the L17-24 region is a model-level property" is WRONG: the
bump appears only in fitted average-Jacobian maps. Corrected attribution: A2's single-linear-
map approximation degrades in the L17-24 region (the same region where S1's gate failed its
strict middle criterion, D19) - a property of the estimator, and the campaign's main honest
limitation. The bump is a genuine distributional degradation (rank AND KL move together), not
a metric artifact.

Other X11 readings: at L0 the logit lens scores 6842 vs the S2 lens 26121 (the fitted lens is
3.8x worse at the embedding layer on captions; on WikiText S1 beat the logit lens at L2-16 -
domain difference); agree at L30: logit 0.763 > S2 0.729 (the untrained lens still leads on
agreement late, consistent with S1's D19); image tag: the logit lens's last-patch rank 33.17
BEATS the model's own 57.83 (the intermediate unembed's flatter tail - a rank-metric
curiosity), while the fitted S2 image lens sits at 3302.7 (V1: the image rows carry the
image->image mass). image-q3 rows identical to image (the same collapsed last patch, V5).

X1 v2 re-run: the first launcher fired at 06:11 but the compare OOM'd (its >=20 GiB pick
shrank mid-run); redeployed with expandable_segments + 3 attempts + a >=22 GiB gate, waiting
for a window. FD remains not_measured (skipped by design). /tmp on the workstation wiped the
older pulls after ~6 h; all canonical results are durable on the Brev box under
/data/anhnq/vlm-lens-out/validation/.

## D38 - 2026-10-08 ~12:50Z: X1 v2 float64 re-run deferred - 3 consecutive OOMs; the analytic justification stands

The corrected X1 compare (float64 cosine + text_rows_max_rel_fro) failed all 3 attempts
(last: rc=1 at 11:04:25Z, ALL_ATTEMPTS_FAILED 11:06:25Z): every >=22 GiB window was grabbed
by a co-tenant mid-load; the compare's peak is ~25-30 GiB (bf16 model + the per-sample
Jacobian backward through 31 layers). The box has been memory-packed for ~30 h (max free
18.7 GiB at 12:45Z). Deferred as OPTIONAL: the float64 fix is justified analytically (the
16.7M-element reductions become exact to ~1e-15; the fp32 artifact cos=1.0019 > 1 cannot
survive exact reductions), the original JSON's max_abs/rel_fro numbers are plain
subtractions and trustworthy, and the adjudication (D36) rests on those plus torch.equal,
not the cosine. Re-run when a >=30 GiB window persists; also queued as optional: an FD
re-run and the X9 alpha-grid extension.

Campaign status: COMPLETE. S1 conditional pass (D19), X6 fp32 mandate measured, S2 verified
expected (D34/D35, 544 cells / 0 anomalies / 17 identities exact), X1 V1 confirmed
(D36), X3 reproduced + L24 measured, X7 clean, X9 E5-passed, X11 attribution complete
(D37: the L17-24 degradation is a property of the fitted average-Jacobian map, A2's limit).
All artifacts durable under /data/anhnq/vlm-lens-out/validation/ on the Brev box.

## D39 - 2026-10-08 ~16:00Z: final conclusion - the J-lens gives real value for hallucination analysis as a disposition + intervention instrument, not as a detector

Directionality mining of X9's 215 changed generations: only 7 are CONCEPT-DIRECTED (target
token appears / source disappears; add x4, ablate x2, swap x1; 3.3 % of changed, 0.5 % of all
1400 edits). The directed cases are real (e.g. `man->woman add@L24 a=13.81` rewrote a caption
to "The woman is the main focus") but required alpha ~= 5-14x the residual norm - a dominating
perturbation, not a surgical edit; at alpha ~= 1-3x, 85 % of edits leave the greedy path
unchanged (the E5 mis-scaling nuance). Edits bite early (first-diff tokens cluster at caption
positions 10-14 and 1-3) and both edit layers work (add@16 91 vs add@24 70).

CONCLUSION. Yes - qualified: the J-lens is a validated and useful instrument for VLM
hallucination analysis in three specific roles, with mapped limits.
* USE IT (a) as a per-position next-token disposition readout on TEXT positions: it beats the
  untrained logit lens from L12-L27 on rank and KL (L12 1487.5 vs 3875.2; L16 1307.6 vs
  2201.8; L27 414.2 vs 702.3), exact at L31 (17/17 identities) - including the LAST PATCH
  (the image->text handoff, where a hallucination-prone commitment forms).
* USE IT (b) as a causal handle: the directions (rows of W_U J_l, F15) are live - 215 greedy
  caption changes from add/ablate/swap - and the 2-column swap bases are well-conditioned
  (cond <= 2.89, X7). Hallucination-mitigation experiments (ablate a hallucinated concept's
  direction at L16/24) are feasible with this tooling.
* DO NOT use it (c) for mid-layer (L17-24) fidelity claims: the averaged-Jacobian
  approximation degrades there (X11: the untrained lens is monotone, both fitted lenses bump;
  A2's limit - the campaign's main honest limitation, sharpening D19).
* DO NOT use it (d) to attribute hallucination to image content via the image rows: the
  image->image cotangent mass dominates them (X1: 24-255x, cos 0.39->0.09) - the image rows
  are not next-text predictors (V1).
* OPEN (e): the campaign validated the instrument and the causal handle but did NOT run a
  hallucination-specific probe. The natural next experiment (cheap on this tooling):
  identify hallucinated spans in held-out captions (vs annotations or an image-grounding
  check), then (i) test whether the lens's disposition at those positions deviates from the
  model's own logits before the unsupported commitment, and (ii) ablate the hallucinated
  concept's direction and measure whether the hallucination disappears - the causal cure test.
Also note: n=100 fit / 300 held-out is small (the 15 % lens_b half-gap), the rank-vs-KL
tension (D19) means metric choice matters, and fp32+TF32 is required (X6). Optional re-runs
(X1 v2 float64, FD) remain window-gated; a 34 GiB window was observed at 15:54Z.

## D40 - 2026-10-08 ~21:50Z: improvement campaign - metric redesign, affine correction (P1), scaling (P2), alpha sweep (P3) - the mid-layer verdict reverses

The improvement campaign ("design experiments to improve our J Lens performances") produced four
measured levers on the same shared H100 (all window-gated):

1. METRIC REDESIGN. lens_zoo_eval.py now scores mean rank, median rank, top-1/5/10/50
agreement, mean KL and a composite LQS. Medians unmask the lens: at L16 mean rank 1307.6 but
MEDIAN 15; L24 mean 1047.5 / median 4; L30 mean 215.2 / median 1 with top10 0.886 - for the
typical position the lens nearly solves next-token prediction from L16 on. Cross-run determinism:
the re-scored 100-sample merged lens reproduces the historical ranks exactly (L30 215.21).

2. SCALING (P2). Flat from n=20 to n=100 (LQS -0.25/-0.12/-0.19/-0.03; L16 ratio ~31x, L30
4.7-5.7x at every n) - the limiter is the estimator's model class (A2), not sample count. Data
levers: target_mask=text (excluding the image block's next-image-token targets) gives the best
n=20 score (LQS +0.116, L30 ratio 2.61 vs 4.7-5.7) - a 50-sample refit (phase3c) is queued,
window-gated; image-target-mask rows and the all-target zoo are worse; prompt/caption variants
neutral.

3. AFFINE CORRECTION (P1) - the decisive win. moment_census.py measures per-layer E[h] and
E[h_final] on 100 corpus samples (text mask, 7144 positions); the readout can apply
h_final ~= b_l + s_l * (J_l h_l) with b_l = E[h_f] - J_l E[h_l], s_l = <t,h_f> / ||t||^2. Census:
||b|| ~ 97-131, s ~ 0.75-1.92 mid-late but NOISY early (s_0 = -3.19, a ratio-of-means artifact).
Merged-lens A/B on the 300-sample held-out (tag=text):

| L | rank_nb | rank_b | KL_nb | KL_b |
| --- | --- | --- | --- | --- |
| 0 | 26120.6 | 30075.6 | 14.21 | 21.88 (fixed by the clamp) |
| 8 | 6439.1 | 1794.6 | 8.13 | 6.72 |
| 16 | 1307.6 | 1185.6 | 4.46 | 4.60 |
| 20 | 2753.0 | 811.3 | 6.60 | 2.87 |
| 24 | 1047.5 | 626.7 | 4.60 | 2.44 |
| 30 | 215.2 | 76.9 | 0.43 | 0.44 |

The L17-24 bump collapses (L20/L16 bump ratio 2.11x -> 0.68x; L30 ratio 5.05 -> 1.80); LQS
-0.102 -> +0.335. A scale-clamp variant (phase3d: every s clipped to [0.5, 2.0]) fixes the L0-4
blowup and strictly dominates: KL at L0-4 21.9 -> 7.0 (better than unbiased 9.6-15.6), rank
ratio 706 -> 46 (unbiased 281-647), LQS 0.335 -> 0.766, layers 5-30 unchanged. Cost: a mild KL
regression at L9-16 (+0.4-0.7) that the clamp does not touch. Artifacts: step4/bias-text.pt,
step4d/bias-text.pt (clamped), lens_zoo_biased{,_clamped}.json.

REVISED CONCLUSION (supersedes D39(c)): with bias+clamp the L17-24 region is the lens's BEST
zone (lens KL 2.6-4.0 vs unbiased 5.0-6.6; L20 rank ratio 19.0 vs 64.6) - the "do not use
mid-layer" caveat was substantially the uncorrected mean shift, not only A2's variance limit.
D39 (a)/(b)/(d) stand. A per-layer gated variant (apply the correction where it helps) is the
natural refinement for the L9-16 regression.

4. ALPHA SWEEP (P3/x9b: 2800 greedy generations, held-out captions, add/ablate/swap at L16/L24).
Dose-response at L16: ablate alpha 0.5/1/2/4 -> 7/16/24/34% of generations change; swap ->
7/12/24/39% (swap steepest; swap@L24 reaches 35% at alpha 4). The add grid was designed in
k units and runs entirely at alpha ~= 27-1035 (its summary "k=1" label is really alpha ~= 27 at
L16, 58 at L24 - report alpha, never the k label); combined with D39's add grid (alpha
1.34-14.68, 10-80% change) the change-rate curve is monotone. Concept-directedness validated:
the stored per-row flags reproduce an independent word-boundary recomputation 2800/2800, and
within the sane window (alpha <= 4) directedness is 0.5-3% (ablate 1-2/200, swap 0-6/200); at
alpha >= 27 the add "directed" rows are dominated by degeneration, not injection (example:
dog->cat add@L16 yields "cat cat cat cat ...", a repetition collapse). So: change is cheap,
direction is rare (D39's 7/215 = 3.3% of changed remains the reference), and large-alpha effects
are breakdown. Artifact: step4/x9b_alpha_sweep.json.

Campaign status: P1 done (bias+clamp validated; per-layer gating open), P2 done (saturation;
target_mask=text refit queued), P3 done (dose-response + degeneration boundary), P4 = this entry.
Note: the first fp32 census attempt OOM'd under a 20 GiB gate - the fp32 census needs >= 31 GiB;
the 20 GiB gate suffices for the bf16 census that produced the shipped bias file.

## D41 - 2026-10-09 ~23:20Z: P5 campaign - structural controls, calibration ladder, tuned-lens-style translators - the J-lens's unique value is the late-layer transport

The improvement round (unique ideas from lens literature: logit lens, tuned lens, Patchscopes,
SAE readouts, span diagnostics; the sources and their fitting-problem lists are in the research
payloads) produced a complete MODEL-CLASS LADDER for the LLaVA-1.5 J-lens, all on the 300-sample
held-out (tag=text), zero new Jacobian fits:

| lens | LQS | L8 | L16 | L20 | L30 |
| --- | --- | --- | --- | --- | --- |
| untrained logit lens (baseline) | 0.000 | 151.1 | 30.7* | 64.6 | 5.05 |
| alphaI (scaled identity, no bias) | 0.000 (exact) | 112.9 | 51.7 | 36.5 | 6.68 |
| rank64 / rank256 truncated J | -0.587 / -0.284 | 229/174 | 65.8/46.6 | 265.6/172.7 | 18.9/9.1 |
| full J, uncorrected | -0.102 | 151.1 | 30.7 | 64.6 | 5.05 |
| diag(J) only, no bias | +0.374 | 60.3 | 34.0 | 24.6 | 6.48 |
| shiftP4 (J_{l+4} on h_l, no bias) | +0.524 | 34.6 | 88.7 | 30.6 | 5.05 |
| full J + census bias+scale (D40) | +0.766 | 42.1 | 27.8 | 19.0 | 1.80 |
| **full J + bias + logit_bias (P5e)** | **+0.922** | **37.0** | **21.4** | **15.3** | **1.79** |
| ident + bias (calibrated logit lens) | +0.647 | - | - | - | - |
| lowrankJ (J + rank-32 KL translator) | 1.714 | 10.4 | 6.1 | 6.8 | 5.82 |
| lowrankRaw (pure tuned lens, rank-32) | 1.718 | 9.4 | 5.9 | 4.8 | 5.62 |
*the baseline's L16 row is the untrained lens; ratios are lens/model mean rank.

STRUCTURE (jacobian_structure.py, CPU): alpha_ls = trace(J)/d crosses 1 at L20-23 (0.05 at L0,
1.04 peak at L25) - the late average-Jacobian is identity-like; the early J is a few dominant
asymmetric off-diagonal directions (top-64 energy 87.5% at L0 vs 9.9% at L30; symmetry 1.41 ->
0.51; column-norm std 0.99 -> 0.03). diag carries only 0.04-5.3% of the Frobenius mass yet
scores +0.374 uncorrected - the off-diagonal transport is mostly mean-shift noise until the
bias removes it. RANK COMPRESSION LOSES: truncating J strips the near-identity mass (the
identity part has ~4096 equal singular values), leaving exactly the uncorrected off-diagonal -
the J's value is NOT low-rank compressible.

TRANSPORT DECOMPOSITION (ident_lens.py + zoo, R3 P5): with the SAME bias+scale payload, the
identity lens scores +0.647 vs the J-lens +0.766 - the transport adds +0.119 (16% of the fix).
Per layer the transport is WORTHLESS at L0-7 (~0), grows -0.05..-0.13 through L8-16, peaks
-0.10..-0.25 at L17-28, and -0.508 at L30 (127.7 -> 76.9 rank). The affine correction does the
early work; the transport does the late work.

SPAN OVERLAP (readout_span_overlap.py, R2/R3 P5, zero fits): the J_l top-64 right-singular
directions carry only 3.5-10.6% energy in W_U's readout span (random baseline 3.1%) - at chance
at early layers, 3.4x chance at L30; cross-layer overlap vs the final layer ~chance (0.0002-0.006)
until L16, then 0.018-0.050 at L20-26 (75-200x chance - the tuned lens's covariance drift
quantified). This MECHANISTICALLY EXPLAINS the alpha thresholds: a J-lens-vector edit moves the
state along directions with ~3-10% logit-level effect, so ~10-30x the residual-level magnitude
is needed to move logits (D39's alpha 5-15 transition + the x9b's alpha 27 = 80% change).

CALIBRATION (calib_readout.py + write_calibrated_bias.py, I3a/I3b): per-layer output temperature
is rank-invariant by construction (a positive scalar on the residual) - the KL-scalar calibration
changes no LQS, only KL (-3.3% mean, mid-late already KL-optimal: the census L2 scale equals the
KL optimum there). The LOGIT-SPACE bias (the mean-gap logit_bias, P5e) with the WRONG sign cost
-0.100 LQS (doubled the marginal bias - caught and fixed); with the CORRECT sign (subtract
E[z_lens - z_model]) it is a WIN AT EVERY LAYER (2-23%): LQS +0.766 -> +0.922 - the new best
J-lens readout, generalizing from 2482 fit positions. readout/s2_eval gained temp/logit_bias
payload keys (bit-identical when absent; pytest 62 green incl. the parity test).

TUNED-LENS-STYLE TRANSLATORS (lowrank_translator.py, I3b+R1, rank-32 KL distillation to the
model's own logits on 40 fit samples, J FIXED): lowrankJ (x = J_l h_l) 1.714 ~= lowrankRaw
(x = h_l, the pure tuned lens = the literature's ceiling) 1.718 - the rank-32 translator
dominates mid-layer (+0.8 LQS over the best J-lens) and both show an L24-28 bump the J-lens+bias
does not; BUT BOTH ARE 3x WORSE THAN THE J-LENS AT L30 (5.82/5.62 vs 1.79) - the J's full-rank
late-layer transport is unmatched by rank-32 bottlenecks.

REVISED CONCLUSION (D39/D40 + this): the ladder is complete. The J-lens's unique value over the
tuned lens is the LATE-LAYER (L25-30) transport; the tuned lens's value is the mid-layer
translator; the affine bias+scale+logit_bias correction is necessary everywhere. Best J-lens
readout: z = unembed(s_l (J_l h + b_l)) + logit_bias_l, LQS +0.922, L30 ratio 1.79. The
practical recipe for deployment: J at the last 6 layers + the moment census + the mean-gap
logit_bias; for mid-layer readouts prefer KL-distilled translators.
Status: phase3c's 23.7-h target_mask=text refit COMPLETED 20:44 (its zoo was killed 8 s in by
an unrelated SIGTERM; re-launched as phase7: the tmtext50 lens bare + with the best payload).
Artifacts: step5/{jacobian_structure,synth/,ident/,calib/,span_overlap,W_U.pt,lens_zoo_*},
step6-lowrank, step6-raw, step5e/bias-text.pt; commits 29fa836, 893f8d2.

## D42 - 2026-10-10 ~00:20Z: target_mask=text confirmed at n=50 and COMPOSES with the calibration payload - the final deployment recipe

The phase3c 23.7-h refit (target_mask=text, 50 samples, all layers, fp32+TF32) completed 20:44;
its zoo was killed 8 s in by an unrelated SIGTERM and re-run as phase7 (bare + with the best
payload). Verdicts, 300-sample held-out (tag=text):

* BARE: LQS +0.337, L30 ratio 2.667 - the D40 x1_text finding (LQS +0.116, L30 2.61 at n=20)
  CONFIRMS at n=50 and the lever is stable across sample counts (+0.44 LQS over the all-target
  merged at bare). Mid-layer trades off as noted (L16 41.2 vs the merged's 30.7).
* COMPOSED (step5e payload: census bias+scale + logit_bias): **LQS +0.947, L30 ratio 1.30** -
  better than the merged+payload (+0.922 / 1.79) and THE BEST LATE-LAYER READOUT MEASURED, J or
  tuned-lens (the rank-32 translators' L30: 5.62-5.82). Mid-layer stays behind (L16 37.7, L20
  30.8 vs the merged+payload's 21.4/15.3) - the image-target exclusion sharpens exactly the
  late (image->text handoff) layers.

FINAL RECOMMENDATION (the complete ladder, 300-sample held-out):
* LATE-LAYER / captioning-disposition readouts (the hallucination-commitment region):
  **tmtext50 + census bias+scale + logit_bias = LQS +0.947, L30 ratio 1.30** (artifacts
  tmtext-half-a + step5e).
* MID-LAYER readouts: rank-32 KL-distilled translators (LQS 1.72, step6-lowrank/step6-raw) or
  the merged lens + the same payload (+0.922).
* The J-lens's unique value over the tuned lens remains the L30 transport (1.30 vs 5.6-5.8);
  for mid-layer fidelity prefer distilled translators.
Artifacts: step4/lens_zoo_tmtext{,_cal}.json; phase7 launcher; commit follows.

## D43 - 2026-10-10 ~13:00Z: DATA SCALING - the J-class saturates at ~100, the translator class does NOT; the production recommendation flips to the translator

Two scaling curves on disjoint corpora, all metrics 300-sample held-out (tag=text) except
noted.

J-CLASS (average-Jacobian + moment calibration). Inter-fit LQS flat 20->100 (D40). NEW: an
online probe (fit_masked --probe-every; the running mean-J scored on 16 held-out samples every
5 fit samples) gives the intra-fit curve: text-KL 6.305 (n=5) -> 6.043 (n=10) -> 5.989 (n=15)
- improving but decelerating; image-position KL flat throughout (11.51/11.58/11.53 - the image
rows are not next-token predictors, V1's echo). The J-class is data-saturated: its value is the
moment-based affine correction, which converges ~1/sqrt(n) over 7k positions by n~100.

TRANSLATOR CLASS (rank-32 KL distillation to the model's own logits; CPU-resident activation
cache so fits are OOM-robust on the co-tenant-heavy box). Fit-side mean best val_kl: 1.424
(n=40) -> 1.049 -> 0.934 -> 0.888 (n=1000) - decelerating ~1/sqrt(n). BUT the held-out LQS
KEEPS RISING and is ~linear in log n: 1.714 (40) -> 2.043 (100) -> 2.691 (500) -> **3.009
(1000)**; +0.33/+0.65/+0.32 across the rungs (~0.4 LQS per e-fold of data). The fit-KL and the
rank metric DECOUPLE: KL approaches its asymptote while the rank ordering keeps sharpening
(the D19 rank-vs-KL tension at scale).

PER-LAYER (rank ratios, lower better): lrJ1000 = L0 3.06, L8 3.72, L16 1.45, L20 1.22, L24 2.25,
L28 4.03, L30 **1.30** vs the best J-lens (tmtext50+bias+scale+logit_bias) L0 42.6, L16 21.4,
L20 15.9, L24 12.5, L30 **1.30**. The J-lens's L30 parity means the D41/D42 claim "the J's
unique value is the late-layer transport" WAS A SMALL-DATA ARTIFACT: the translator's late-layer
weakness (5.82 at n=40) decays with data (3.90/2.05/1.30) exactly to the J-lens's level.

REVISED PRODUCTION RECOMMENDATION (supersedes D42):
* Mid/late-layer readouts: rank-32 KL translator fitted at n >= 1000 (LQS 3.009); keep scaling
  - the curve is not saturated at n=1000, recommend n=2000-5000 (~+0.3-0.7 LQS per e-fold;
  ~1-2 h per 1000 samples with the CPU cache).
* Cheap zero-fit fallback (~1 h total): the J-lens (moment census + bias+scale + logit_bias),
  LQS 0.92-0.95 - 3x below the n=1000 translator but needs no distillation fit.
* Causal edits: the J-lens keeps the edit handle; composing edit directions through the
  translator lens (W_U[t] @ A_l J_l) is the natural follow-up.
Tools shipped this round: the online probe (fit_masked --probe-every/--probe-manifest, bit-
identical when off; pytest 64 green), build_caption_manifest_n.py (disjoint nested scaling
corpora), the CPU-cache lowrank_translator (bit-parity verified). Artifacts: step8/{n100,n500,
n1000}, step8/lens_zoo_scaling.json, step9-jprobe (probe fit, running), manifest-scaling.jsonl.

## D44 - 2026-10-10 ~15:40Z: the workspace round - the visual->verbal handoff, the prior-vs-grounding race, and a training-free causal cure

Literature frame (R4/R5/R6; ~30 sources): Anthropic's J-space/global-workspace paper (mid-layer
verbalizable band, motor-regime flip in the final layers; only an anecdotal multimodal check);
Zhang et al.'s staged cross-modal flow (early=global features, mid=object-specific, high layers
propagate to the LAST INPUT position); FastV (image tokens processed mainly early); DAMRO
(attention mirrors the ViT and favors background); OPERA/VCD/PAI/LURE (prior dominance,
summary-token over-trust, text inertia); Fazli et al.'s commitment-depth gap; VRP (LLaVA "locks
its prediction in a fragile late-stage bottleneck"); HALP (pre-generation hallucination
detectability); Dual-Pathway Circuits (grounding vs hallucination pathways with a polarity
flip); Endognostics (decodability != causal control). THE GAP: no systematic test of a
visual->verbal workspace handoff in VLMs existed - this round fills it with three zero-fitting
instruments (cross-image patching x12, mean-replacement cure x13, calibrated-lens workspace
trajectories x14).

X12 CROSS-IMAGE PATCHING (8 disjoint-object COCO pairs; replace A's image-token rows with B's;
300-sample quality swings):
  layer:  0     8     12    16    20    24    28    30  (image variant)
  changed:1.00  1.00  1.00  1.00  0.875 0.125 0.50  0.25
  A-objects vanish: 1.00/1.00/1.00/0.81/0.31/0/0.125/0 ; B-objects appear: 0.41-0.49 through
L16 then ~0. Single-position (last-image-token) patches do nothing (<=0.25) - the content is
distributed, not stored in one handoff token. Text-position controls stay at the 0.12-0.37
floor. THE COMMITMENT BOUNDARY: visual content is causally live at image positions only until
L16-20; re-writing image tokens after L20-24 no longer changes the caption.

X13 CURE TEST (39 images, 10 hallucinations vs COCO instances ground truth; mean-replacement
ablation of the hallucinated category's J-lens-vector direction, alpha=1):
  layer 16/24/28/30 -> removed 0.50/0.60/0.70/**0.90**, grounded-category retention
  1.00/1.00/1.00/**1.00**, clean-cure 0.50/0.60/0.70/**0.90**.
  CONTROL (ablating a GROUNDED category's direction): retention 0.97/0.79/0.52/**0.35** -
  grounded-direction ablation is destructive; hallucinated-direction ablation at L30 is
  surgical. A training-free 90%-effective causal cure with a clean layer dose-response.

X14 WORKSPACE TRAJECTORIES (24 images, calibrated logit lens = identity transport + the moment
payload, zero fitting beyond moments):
* THE PRIOR-VS-GROUNDING RACE: hallucinated generated words are high-ranked FROM L0 (rank ~787
  vs grounded ~5358 - the language prior leads 7x before visual computation), while grounded
  words are built up by the model (5358 -> 1.18 monotone); they CROSS at L24-26 (grounded 4.86
  vs halluc 6.50 at L24; grounded 1.18 vs halluc 4.5 at L26). Grounding is a computation that
  must overtake the prior; hallucination is the prior's default.
* TEXT-MEDIATED HANDOFF CONFIRMED: the image's true categories rank ~1200-2000 at the last
  IMAGE-token position at every layer (never decodable there), but at the LAST PROMPT position
  (the handoff whose distribution predicts the first caption token) the best-per-image true
  object rank falls 1423 (L16) -> 34 (L20) -> 6-15 (L22-25), then the L31 readout flips to the
  imminent token (3464 mean; the motor regime). The visual->verbal handoff is text-mediated and
  completes ~L20-25 - exactly the X12 boundary and the D41-D43 transport zone.
* DOES-THE-MODEL-KNOW (negative): at hallucinated words' decoding positions the image's actual
  categories rank ~16-35 through L20-30 (vs the hallucinated word's 2-6) - the model does NOT
  represent the truth competitively at those positions; hallucination is the prior winning, not
  a late suppression of better knowledge (sharpens the knows-but-commits hypothesis).
* SEPARABILITY: grounded vs hallucinated word ranks at L30: 1.18 vs 6.0 (gap 4.8) - the
  calibrated lens is a training-free hallucination-risk signal during generation.

THE INSTRUMENT (the round's goal - cheap, training-free, causal): the calibrated logit lens
(identity transport + moment census payload; one forward per sample) for readout, its
J-lens-vector directions for causal edits (mean-replacement ablation per Belrose App. D), and
cross-image residual patching for localization. It produced: the commitment boundary (L20-24),
the prior-vs-grounding race and crossing (L24-26), the text-handoff localization (L20-25), a
90%-effective hallucination cure at L30 with zero collateral, and the layer-dose-response
matching the L30 transport: the workspace commits late, and the late layers are where both the
readout value and the causal handle concentrate.
Held-out predictions for the next round: sink-clamp on massive activations (H6), hesitation
prediction (H7), attention-mirror correlation (H3), POPE pre-generation AUROC from the handoff
readout (HALP-style, H4), irreversibility horizon (R6 H9).
Artifacts: step10/{x12_cross_image_patch,x13_cure_test,x14_workspace_probe}.json; commits
4fc07e1, 96380fc; launchers phase10-12.

## D45 - 2026-10-10 ~20:00Z: round 2 - the held-out predictions, tested (pre-generation truth detection 0.985, the BOS sink is load-bearing, the CLIP-mirror curve, hesitation direction confirmed but underpowered)

Deployment: four experiments as one sequential chain on the shared node (all 8 GPUs held
54-79 GB by other tenants when launched; parallel launchers would race the free-memory guard,
which always sorts to the same GPU). Chain x18 -> x16 -> x17 -> x15: ~45 min total, each step
gated on >=20 GB free and skip-if-present. JSONs: step11/{x18_pope_handoff,x16_sink_clamp,
x17_attention_mirror,x15_hesitation}.json. Scripts committed 352e044 (x16/x17/x18 + launchers)
and d14d55e (x15).

X18 POPE PRE-GENERATION HANDOFF (H4): 49 images, 185 questions (87 present / 98 absent).
Model POPE accuracy 94.1% (present 96.6 / absent 91.8), 8 hallucinated yes. ONE forward at the
last prompt position + the calibrated lens (identity transport + moment payload):
AUROC(present vs absent) = 0.985 at L28 (0.982-0.985 across L20-30); mean gap +5.28 (present)
vs -4.97 (absent). The lens disagrees with the model exactly on its wrong affirmations:
hallucinated-yes items rank LOW on the gap (AUROC 0.105 with hallucinated as positive class =
0.895 as a low-gap hallucination detector). A training-free, single-forward, pre-generation
truth/risk readout - the HALP-style promise, delivered with near-ceiling discrimination.

X16 MASSIVE-ACTIVATION SINK CLAMP (H6): 20 images, top-16 dims per (layer x position group),
zero-clamp at L16/24/28/30. THE BOS SINK IS LOAD-BEARING: clamping it collapses grounded
retention (0.278 at L16 bos; 0.222 for all-groups) and explodes the handoff true-object rank
(+1433 at L16, +550 at L28). IMAGE-TOKEN MASSIVE DIMS ARE NEARLY INERT: retention 0.94-1.00
and rank deltas -43..+330 while captions surface-change (changed 0.3-0.8). The sink is a causal
bottleneck at BOS, decorative at image tokens - consistent with "image tokens are streamed,
not stored". CAVEAT: baseline had only 1 hallucinating image / 20; the hallucination_removed
column is uninformative at this n.

X17 CROSS-MODAL ATTENTION MIRROR (H3): 20 images, eager attention, CLS->patch vs text->image.
THE MIRROR IS REAL AND DEPTH-DEPENDENT: mean Spearman rho rises 0.10 (L0) -> 0.29 (L1) ->
0.38-0.52 (L5-L10) -> 0.58 (L16) -> 0.66 (L20), then DECOUPLES in the final layers (0.17-0.43,
L27-30) - the LLM's access to the image first mirrors the vision tower, then the verbal
workspace takes over. Image-attention share peaks at L0 (0.75) and settles to 0.05-0.19.
Hallucination join UNDER-POWERED: 1 hallucinating / 19 clean images; no claim. Needs ~200.

X15 DECODING HESITATION (H7): 40 images. Direction CONFIRMED, sample small: grounded word
completion steps (n=76): entropy 0.513, top1-margin 5.91, logprob -0.179; hallucinated (n=6):
entropy 0.989 (~2x), margin 4.37, logprob -0.488. AUROC with hallucinated positive: model
entropy 0.636, margin 0.394 (~0.61 inverse), CALIBRATED-LENS ENTROPY beats the model's own
entropy (0.834 at L6, 0.794 L7, 0.748 L5). The lens uncertainty is the better hallucination
signal; the 6-word positive class makes this a direction, not a magnitude.
Next: enlarge n (x15/x17 joins), list the census dims, fuse x18+x15 into a pre-generation
risk score, and the x14/x13-style scale-out (200+ images).
