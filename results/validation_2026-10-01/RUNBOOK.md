# RUNBOOK — J-lens validation campaign: setup & run on a new server

Target reader: whoever operates the new (bigger) box. Everything below is copy-pasteable.
Line references are to the scripts in `code/` as shipped.

**What runs**: 7 sequential steps on `llava-hf/llava-1.5-7b-hf`, driven by `run_campaign.sh`
under the `gpu_guard.sh` supervisor. Outputs: fitted lenses (fp32 Jacobians) + per-experiment
JSONs for the validation report. The whole campaign is resumable per step (checkpoints every
5 samples).

---

## 0. TL;DR — the two paths

**Path A — migrate the completed work (recommended).** Copy 4.8 GB of artifacts from the old
box (§4), set 4 path variables (§3), prewarm caches (§5), run the pre-flight gate (§6), launch
(§7). The campaign then resumes: S1 re-verifies in ~3 min, and real work starts at **S2**.
No COCO-image processing, no WikiText download needed if you also symlink the image dir (§4.3).

**Path B — from scratch.** Same, but also run step 0/1 (§8.2) to rebuild manifests, which needs
the COCO val2014 images (~6 GB) and one network pause for the WikiText stream.

---

## 1. Hardware / OS assumptions

- 1× CUDA GPU, **≥40 GB VRAM** comfortable (the shipped guards assume 80 GB; a 48 GB card works
  with the same settings — fp32 S2 peak is ~31 GiB, §7.2).
- ~200 GB free disk, of which **≥40 GB** for campaign outputs. Scripts split outputs across two
  roots (`RUN`, `MOUNT`); a single disk is fine (§3.1).
- Linux, bash, `nvidia-smi`, `rsync`. No other GPU jobs during the run (the guard exists because
  the old box was shared; on a dedicated box it mostly idles).

## 2. Known-good environment (old box, full campaign validated on it)

| Component | Version |
|---|---|
| Python | 3.13 (conda env `vlm_truth_py313`; ≥3.10 required) |
| torch | 2.5.1+cu121 |
| transformers | 5.17.0 (≥5.5 required by `pyproject.toml`) |
| datasets | 3.5.0 (only needed for step 0/1 rebuilds) |

Setup on the new box:

```bash
conda create -n vlm_truth_py313 python=3.13 -y && conda activate vlm_truth_py313
pip install torch --index-url https://download.pytorch.org/whl/cu121   # or any torch ≥2.4 that sees your GPU
cd <REPO>            # the repo tree copied in §3
pip install -e ".[dev]"        # transformers, huggingface_hub, numpy, pillow, pytest, matplotlib, datasets
```

The vendored Anthropic estimator lives at `third_party/jacobian-lens/` and is loaded
automatically (`src/vlm_lens/_vendor.py`); imports work either via `pip install -e .` or via
`export PYTHONPATH=<REPO>/src` (what the campaign scripts do). Do not touch `third_party/`.

## 3. Repository layout & the path variables you must set

### 3.1 The variables

Every script starts with a small config block. Set these consistently:

| Variable | Meaning | Old-box value |
|---|---|---|
| `REPO` | repo root (this tree) | `$HOME/ai4life/phuongnh/vlm-lens` |
| `P` | python binary | `$HOME/miniconda3/envs/vlm_truth_py313/bin/python` |
| `RUN` | durable outputs (manifests, JSONs, S1 lens, X1) | `/data/vlm-lens/validation` |
| `MOUNT` | S2/X1 scratch+outputs (largest traffic) | `/home/nvidia-lab/data_mount/vlm-lens` |

Files to edit (config block near the top of each):

```
code/run_campaign.sh   (lines ~27-30; also the two `env DIMBATCH=` values, §7.2)
code/gpu_guard.sh      (lines ~22-30; also FREE_MIN / TRIP_S, §7.2)
code/run_step0.sh      (lines ~5-8; IMAGES= COCO val2014 dir)
code/run_step2.sh      (lines ~5-7)
code/run_step3.sh      (same block: REPO/P/RUN/MOUNT)
code/run_step4.sh      (same block)
code/run_fd.sh         (lines ~12-13: R=, OUT=)
code/collect_results.sh / collect_results.py  (check the same names)
```

**Single-disk box**: set `MOUNT="$RUN"` — everything then lands under one root and the
`DISK_PATH` gates all check that disk.

### 3.2 The tree to copy

```bash
rsync -a --info=progress2 OLD:<REPO>/ NEW:<REPO>/
```

Copy the whole tree (includes `src/`, `scripts/`, `third_party/`, `tests/`, and this campaign's
`results/validation_2026-10-01/{code,logs,*.md}`). `.git` optional. Exclude nothing else.

## 4. Data & artifacts to migrate

### 4.1 Completed work from the old box (4.8 GB — Path A)

```bash
rsync -a --info=progress2 OLD:/data/vlm-lens/validation/ NEW:/data/vlm-lens/validation/
```

Contains: `step0/` frozen manifests (+ hashes), `step1/x3_norms.json`, `step2/s1_score*.json`,
`s1-text/` (S1 lens + checkpoint), `x6*/` (dtype verdict data), `x3/`, `fd/` if present.
**Do not regenerate the manifests** if you migrate them — the report cites their hashes, and a
rebuild changes the header (`created_utc`, paths) and therefore every downstream provenance file.

### 4.2 Model download (new box needs its own HF cache)

```bash
export HF_HUB_OFFLINE=0
python - <<'PY'
from transformers import AutoProcessor, LlavaForConditionalGeneration
LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
PY
```

~15 GB. Afterwards all fits run with `HF_HUB_OFFLINE=1` (the scripts set it themselves).

### 4.3 COCO images (needed by every caption-lens step: S2, X1, X7/X9)

Frozen manifests reference the images by name under the old absolute dir
`$HOME/ai4life/phuongnh/vlm-truth/data/coco2014/val2014/val2014`. Two hash-preserving options:

- **Symlink the same path** to wherever the images live on the new box:

  ```bash
  mkdir -p $HOME/ai4life/phuongnh/vlm-truth/data/coco2014/val2014
  ln -s <NEWBOX_COCO_VAL2014_DIR> $HOME/ai4life/phuongnh/vlm-truth/data/coco2014/val2014/val2014
  ```

- Or copy just the 130 used files (25 MB) — names are frozen in the manifest headers:

  ```bash
  python - <<'PY' > needed.txt
  import json, pathlib
  for f in ["manifest-fit.jsonl", "manifest-heldout.jsonl"]:
      h = json.loads((pathlib.Path("/data/vlm-lens/validation/step0")/f).read_text().splitlines()[0])
      print("\n".join(h["meta"]["image_names"]))
  PY
  # then copy those files from the old box into the same dir
  ```

(Full `val2014` is ~6 GB if you prefer to copy it whole.) If you skip this, S2 will fail at
`encode_mm`/image open — the log names the missing file.

## 5. Pre-flight verification (do not fit before these pass)

```bash
cd <REPO> && export PYTHONPATH=$PWD/src
$P scripts/check_equivalence.py --backend tiny                      # exact-equality gate, CPU, seconds
$P -m pytest -q                                                     # full suite, ~15 s CPU
$P scripts/check_equivalence.py --backend hf-llava \
    --model llava-hf/llava-1.5-7b-hf --device cuda --dtype bfloat16 \
    --image <any .jpg from the corpus>                              # must print: EQUIVALENCE PASS
```

The third command is the pre-fit gate: it proves the lens-side residual stream equals the stock
HF forward on *this* GPU. Exit code 0 is required. (`scripts/dry_run.py` is optional.)

## 6. Launching

```bash
cd <REPO>
nohup setsid bash results/validation_2026-10-01/code/gpu_guard.sh \
    >> results/validation_2026-10-01/logs/gpu_guard.log 2>&1 < /dev/null &
```

`gpu_guard.sh` probes the box (a 20×4096³ TF32 matmul) and starts `run_campaign.sh` when
`FREE_MIN` MiB is free; it also kills the chain if a running step produces no new `*.pt` for
`TRIP_S`. On a dedicated box you can also skip the supervisor and run `run_campaign.sh` directly.

### 6.1 Dedicated-box tuning (optional)

| Knob | Where | Shipped | For a dedicated box |
|---|---|---|---|
| `FREE_MIN` | `gpu_guard.sh:29` | 24000 | `8000` (or less) |
| `TF_MIN` | `gpu_guard.sh:28` | 0 (log-only) | leave 0 |
| `TRIP_S` | `gpu_guard.sh:30` | 36000 | 36000 (hang kills) |
| S2 `DIMBATCH` | `run_campaign.sh:131` (`env DIMBATCH=1`) | 1 | `4`–`8` on ≥80 GB — pure activation-memory knob, identical math |
| X1 `DIMBATCH` | `run_campaign.sh:139` | 1 | same |

**Fingerprint rule**: `dim_batch` is part of a checkpoint's settings fingerprint. If you copy a
mid-run S2 checkpoint from the old box (cadence is every 5 samples), you must keep
`DIMBATCH=1` for the half it belongs to, or the fit hard-fails/moves aside the checkpoint.
Fresh halves (no checkpoint) accept any `dim_batch`.

## 7. What the campaign runs, and how it should look

Step order inside `run_campaign.sh` (each behind a free-memory + disk gate, auto-retry of
transient failures — OOM, SIGKILL, disk-full — with no attempt cap):

| # | Step | Script | Guard | Expected (quiet H100, from X6-measured 311 s/img-sample) |
|---|---|---|---|---|
| 1 | S1 text control fit (bf16, skip_first=16) + held-out score | `run_step2.sh` | 24 GiB | ~2.5 h from scratch; ~3 min if migrated |
| 2 | S2 caption halves ×2 (fp32+TF32, 50 img samples each) + merge + held-out eval | `run_step3.sh` | 36 GiB | ~8.5–9.5 h (two halves ≈ 4.5–5 h each) |
| 3 | FD: S1 finite-difference row (optional; chain continues on failure) | `run_fd.sh` | 36 GiB | ~0.5–1 h |
| 4 | X1: 20-image shard, 6 layers, fp32 | `run_step4.sh` | 36 GiB | ~1 h |
| 5 | X3 re-census with L24 (for X9 units) | inline `x3_census.py` | 22 GiB | ~20 min |
| 6 | X7 conditioning + X9 edit sweep vs merged caption lens | inline `x7_x9_interventions.py` | 22 GiB | ~0.5–1 h |

**Total ≈ 14–16 h** on an unshared box; roughly 24–36 h if the GPU is derated by co-tenants.

### 7.1 Healthy-run signatures

```bash
tail -f results/validation_2026-10-01/logs/run_campaign.log     # gates + step starts
tail -f results/validation_2026-10-01/logs/s2_attempt1.log      # per-sample lines
```

Per-sample line (the thing to eyeball):

```
sample 12/50 COCO_val2014_000000xxxxxx::prompt+caption::<hash> seq=6xx images=576
            pos={'text': NN, 'image': 576, 'all': NN+576} <secs>s rel_change=(a,b,c)
```

- `text + image = all`, `all = seq − 2` (skip_first=1 + exclude_last) — arithmetic must hold.
- `rel_change` decays toward small values after the first couple of samples (`nan` on sample 1
  of any fresh run is by construction — the running mean has no history yet).
- A new `<out>/checkpoint.pt` (~6.4 GB) appears every 5 samples; watcher one-liner:

  ```bash
  ls -lh $(grep -o '/[^ ]*s2-half[^ ]*' /dev/null 2>/dev/null); \
  find /data/vlm-lens/validation <MOUNT> -name 'checkpoint.pt' -newermt '-20 min' 2>/dev/null
  ```

### 7.2 Creativity you should not apply

- Do not change the dtype policy: S1 = bf16; S2/X1/FD = **fp32 + TF32** (the X6 experiment
  *measured* this — fp32 quality at TF32 speed; bf16 for S2 was rejected there).
- Do not change `--skip-first` (1 for multimodal, 16 for the text control) or mask sets.
- Do not run fits concurrently: one GPU process, one step. The guard enforces ordering.
- Do not edit `third_party/jacobian-lens/` (vendored, Apache-2.0; upstream interop depends on it).

## 8. Per-step details

### 8.1 Commands (if you need to run one step manually)

Everything is wrapped by `run_campaign.sh`; for surgical reruns:

```bash
env DIMBATCH=1 bash code/run_step3.sh        # S2 halves + merge + eval (resumes from checkpoints)
bash code/run_fd.sh                          # FD row (publishes step2/s1_score.json atomically)
env DIMBATCH=1 bash code/run_step4.sh        # X1
$P code/x3_census.py --manifest $RUN/step0/manifest-fit.jsonl --n-samples 50 --out $RUN/step1/x3_norms.json
$P code/x7_x9_interventions.py --lens-dir $MOUNT/s2-merged/artifacts \
    --manifest $RUN/step0/manifest-heldout.jsonl --n-samples 10 \
    --norms-json $RUN/step1/x3_norms.json --json $RUN/step4/x7_x9.json
```

All fits: `scripts/fit_llava.py --backend hf-llava --manifest … --layers/--masks/--dim-batch/--dtype/--skip-first/--checkpoint-every …`
(see the step scripts for exact argument sets; `--limit` caps samples, `--out` is the lens dir).

### 8.2 Path B — rebuilding step 0/1 from scratch (only if not migrating)

```bash
# needs: COCO val2014 dir + network (WikiText stream) + the HF model
bash code/run_step0.sh   # manifests (100 fit/30 held-out images; 100 fit/30 held-out text) +
                         # cost measurement + X3 census
```

Then run the X6 dtype experiment once on this hardware before trusting S2's dtype choice:
`code/run_step1.sh` (x6 fp32/bf16 legs + X3/x3_census). Keep its JSON — the report cites it as
hardware-local evidence. After that, launch §6 normally.

## 9. Outputs & collection

| Path | Content |
|---|---|
| `$RUN/step0/` | frozen manifests + `cost.json` |
| `$RUN/step1/` | `x3_norms.json` (+ X6 JSONs if rebuilt) |
| `$RUN/step2/` | `s1_score.json` (lens fidelity, FD row) |
| `$RUN/s1-text/artifacts/` | S1 lenses (`lens-{text,image,all}.pt`) + `provenance.json` |
| `$MOUNT/s2-half-a|b/`, `$MOUNT/s2-merged/artifacts/` | S2 lenses + checkpoints |
| `$MOUNT/x1-*/` | X1 lenses (6 layers) |
| `$RUN/step4/x7_x9.json` | X7/X9 experiment results |

Collect + package for the report:

```bash
bash code/collect_results.sh     # (check its RUN/MOUNT block) → copies JSONs + manifest hashes
```

## 10. Troubleshooting

| Symptom | Meaning / action |
|---|---|
| `Terminated` in a step log | External SIGTERM (someone stopped it). Just relaunch §6 — fits resume at the last 5-sample checkpoint. |
| `CUDA out of memory` | Transient by design: the chain retries. If persistent, raise `FREE_MIN` / lower co-tenant pressure (not applicable on a dedicated box). |
| `No space left on device` / `free=X MiB at save time` | Disk full at checkpoint write. Free space; failure classifies as transient and auto-retries in the gates. Hardened saver leaves no `*.tmp.*` residue. |
| `unexpected pos` when loading a checkpoint | Truncated legacy checkpoint from the disk-full era. Move `checkpoint.pt` aside; the fit restarts that half from 0. |
| Hard-fail naming fingerprint/settings | You changed `dim_batch`/`--dtype`/`--skip-first` vs the checkpoint. Restore the original setting or delete the checkpoint. |
| Guard kills a running step (no new `*.pt` in 10 h) | Genuine hang. Inspect the fit log tail + `py-spy`-style stack, then relaunch. |
| `EQUIVALENCE PASS` absent | Environment/model mismatch — fix before any fit. |
| Campaign exits `CAMPAIGN_*_FAILED` | 3 consecutive non-transient failures in that step. Read the step log tail; the chain does not silently continue. |

## 11. Reference: expected determinism across boxes

Same manifests + same code ⇒ same sample order and same mask arithmetic; the fitted numbers are
**not bit-identical across GPUs** (different reduction orders), but reproduce within fp32 noise.
Lens artifacts keep the upstream keys (`J`, `n_prompts`, `source_layers`, `d_model`); provenance
embeds the manifest hashes so cross-box lineage stays auditable.
