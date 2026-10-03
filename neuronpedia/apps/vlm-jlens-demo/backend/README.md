# vlm-jlens-demo backend

A local HTTP server that exposes a **Jacobian lens (J-Lens) workbench for LLaVA**: lens
readouts over image patches and generated tokens, per-patch attribution maps, causal
patch knockout and interactive residual-stream steering. It is built for
captioning-hallucination research: the unit of analysis is *one image + one prompt +
one caption*, and every intervention reports the caption it produced.

The server is deliberately single-process and CPU-friendly by default: a tiny random
LLaVA runs offline (no downloads, no GPU) so the UI can be developed and demoed
anywhere, and the same code path serves `llava-hf/llava-1.5-7b-hf` with a trained lens
on a single 24 GB GPU.

```
apps/vlm-jlens-demo/backend/
├── vlmj/                 # the server package (app.py = FastAPI, engine.py = model+lens)
├── scripts/make_mock_lens.py
├── scripts/e2e_smoke.py            # live end-to-end smoke against a running server
├── tests/test_api_tiny.py
└── README.md
```

## Quick start (tiny, CPU, offline)

```bash
cd apps/vlm-jlens-demo/backend
~/miniforge3/envs/vlm-lens/bin/python -m vlmj.app          # http://127.0.0.1:8000
# options: --host 0.0.0.0 --port 8000 --reload --log-level debug
```

The first request builds a 3-layer random LLaVA and **auto-fits a J-Lens on three
synthetic dummy images** (≈0.3 s) into `vlmj/.cache/tiny-lens`. Nothing is downloaded;
`VLMJ_CACHE` moves the cache.

```bash
curl -s localhost:8000/api/meta | jq '.mode, .device, .lens'
curl -s localhost:8000/api/samples | jq '.samples[].label'
```

Smoke the whole pipeline:

```bash
B64=$(curl -s localhost:8000/api/samples | jq -r '.samples[0].image_b64')
SID=$(curl -s -X POST localhost:8000/api/session -H 'content-type: application/json' \
        -d "{\"image_b64\":\"$B64\"}" | jq -r .session_id)
JOB=$(curl -s -X POST localhost:8000/api/generate -H 'content-type: application/json' \
        -d "{\"session_id\":\"$SID\",\"max_new_tokens\":24}" | jq -r .job_id)
curl -s localhost:8000/api/jobs/$JOB | jq '.status, .result.caption'
```

## Three ways to get a lens

| Path | How | What it is |
| --- | --- | --- |
| Auto-fit (default, tiny) | `VLMJ_BACKEND=tiny` + `VLMJ_AUTOFIT=1` | Real fit on synthetic dummy images with random weights. Structurally valid, **meaningless numerically** (`/api/meta` → `lens.notes`). |
| Mock | `python scripts/make_mock_lens.py --preset tiny --out /tmp/mock-lens`, then `VLMJ_LENS_DIR=/tmp/mock-lens` | Random matrices, `lens.mock = true`. For UI development without fitting. |
| Real artifacts | `VLMJ_LENS_DIR=/path/to/lens-set` (holds `lens-{text,image,all}.pt` + `provenance.json`, e.g. from `vlm_lens` fitting) | A trained lens; the only mode whose ranks/probabilities are research-grade. |

Rules the server enforces:

* The lens `d_model` must match the model and its `source_layers` must be `< n_layers`.
* The demo loads **one** mask per process — preference `text` → `all` → `image` — because
  a 7B lens is ≈0.7 GiB per mask in fp16 and materializing all three would triple that
  for no benefit. `/api/meta.lens.masks` lists the masks present in the set.
* An explicit `VLMJ_LENS_DIR` **always wins** over auto-fit, even when it is empty or
  broken: the lens then reports `available: false` with the error and every lens endpoint
  answers `409`. `/api/generate` and `/api/session` keep working without a lens.

## Running against LLaVA-1.5-7B

```bash
VLMJ_BACKEND=hf-llava \
VLMJ_MODEL=llava-hf/llava-1.5-7b-hf \
VLMJ_LENS_DIR=/data/lenses/llava-1.5-7b/text \
~/miniforge3/envs/vlm-lens/bin/python -m vlmj.app
```

VRAM budget on one 24 GB card: bf16 weights ≈14.5 GB + activations ≈1–3 GB for a
576-token image prompt and ≤64 new tokens ⇒ fits, but **one job at a time** — the server
serializes all model work behind a single worker thread and a global lock, and you must
never run two server processes on the same GPU. `device` defaults to `cuda` for
`hf-llava` and `cpu` for `tiny`; `dtype` defaults to `bf16` and `fp32` respectively.
If the card is too small, `VLMJ_DTYPE=float16` saves nothing on weights but does lower
activation memory slightly; loading in 4-bit is out of scope (`/api/meta.gpu` reports
free/total VRAM so the UI can warn). The lens itself stays in host RAM (≈2 GiB per 7B
mask after the fp32 cast) and each readout copies one `d_model × d_model` block to the
GPU, so the lens does not compete with the weights for VRAM.

## API

All bodies are JSON (`Content-Type: application/json`), all field names are
`snake_case`, and **errors are `{"detail": "..."}`** with status `404` (unknown
session/job), `409` (lens unavailable, or a patch-only endpoint on a `no_image`
session), `422` (validation) or `503` (model failed to load). `max_new_tokens` is
capped at `64` (`/api/meta.capabilities.max_new_tokens`).

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| GET | `/api/meta` | – | mode/device/dtype, model geometry, lens state, capabilities, job counts, free VRAM |
| GET | `/api/samples` | – | 3 deterministic synthetic PNGs as data URLs |
| POST | `/api/session` | `{image_b64}` or `{image_path}`, optional `prompt`, optional `variant` (`image`\|`no_image`) | encoded session (synchronous); `image` is `null` for the twin |
| GET | `/api/session/{id}` | – | `variant`, `has_baseline`, cached `baseline_caption` |
| POST | `/api/generate` | `{session_id, max_new_tokens}` | `{job_id}` → caption |
| POST | `/api/lens` | `{session_id, layers?, targets, topk?, track?}` | `{job_id}` → per-layer top-k + `model_row` |
| POST | `/api/attribution` | `{session_id, layer, token, metric?, rollout?}` | `{job_id}` → 24×24 `grid` + `grid_rank` |
| POST | `/api/knockout` | `{session_id, patches, mode?, target_tokens?, max_new_tokens?}` | `{job_id}` → caption + logprob deltas + `outcomes` |
| POST | `/api/steer` | `{session_id, layer, mode, token?, source_token?, alpha, positions?, max_new_tokens?}` | `{job_id}` → before/after captions + `diagnostics` |
| GET | `/api/jobs/{id}` | – | `{status, stage, progress, result, error}` |

Long work is always a job: `POST` returns `{job_id}` immediately and the client polls
`/api/jobs/{job_id}` (`status`: `queued` → `running` → `done`/`error`, with `stage` and
`progress` for a progress bar). One job runs at a time; the last 128 finished jobs are
kept. If a request body is invalid the `POST` answers `422` **before** a job is created.

Conventions the UI can rely on:

* `image: {start, end, count}` — `start`/`end` are absolute `input_ids` positions with
  an **exclusive** `end`; `quarter_of_patch[576]` gives each patch's image quarter
  (0 = top-left, 1 = top-right, 2 = bottom-left, 3 = bottom-right, row-major).
* `variant`: `"image"` (default; exactly one image source) or `"no_image"` (text-only
  twin). The twin's `image` is `null` and the literal `<image>` is stripped from its
  prompt; `/api/generate` and `/api/steer` work on it, while `/api/attribution` and
  `/api/knockout` answer `409` because there are no patches.
  `GET /api/session/{id}` echoes `variant`.
* `attribution.grid[24][24]` is row-major over patch indices; `quarters` are the means of
  those four regions; `vmin`/`vmax` bound the grid for colour scaling.
* `attribution.grid_rank[24][24]` is the 1-based rank of the requested token in each
  patch's lens distribution (rank 1 = the patch's argmax; the same competition ranks as
  `tracked`), and `null` for `attn_rollout`, which has no token distribution.
* `lens` rank is **1-based** (rank 1 = the token the lens scores highest) and
  `tracked` echoes the exact strings you passed in `track`.
* every `lens` target also carries `model_row` (`{topk, tracked}`): the model's own
  next-token distribution at that position. It is model output **by construction, not
  lens evidence**, so the UI labels it "model output" beside the lens rows.
* `knockout.deltas` covers the *baseline* caption (filtered by `target_tokens`), while
  `caption_after`/`tokens_after` are regenerated **with** the patches ablated.
* `knockout.outcomes` classifies each selected baseline token against the regenerated
  caption (same filter and order as `deltas`): `removed` if `delta <= -1.0` and the token
  is absent from `caption_after`, `persisted` if `|delta| < 0.2` and the token is present
  there, else `changed`. `classification_note` spells the heuristic out.
* `steer` accepts `positions: "last" | "all"`; `alpha: 0` reproduces the baseline exactly,
  which is the cheapest way to verify the plumbing end-to-end.
* `steer.diagnostics` has one entry per edit: `v_norm` = ‖v_t‖ of the **target** token
  (for `swap` that is the target direction, not the source), `h_norm` = baseline residual
  norm at that layer's last position (`null` if unavailable), and `cond` =
  `torch.linalg.cond` of the stacked `[v_s, v_t]` basis for `swap`, else `null` (also
  `null` when the basis is numerically singular). A `cond` above `1e3` means the
  pseudo-inverse edit is unstable (research note X7): treat such swaps as exploratory.
* `metric`: `lens_prob`, `lens_logit` (J-lens transported logits of one token) or
  `attn_rollout` (attention rollout from the last prompt position to each patch — an
  approximation that uses attention weights rather than ablation; `rollout: true`
  mixes each per-layer mean-head row with the identity (`0.5·A + 0.5·I`, Abnar &
  Zuidema) before multiplying across layers).

### Examples

```bash
# J-lens readout: first generated token + patch 300, two source layers
curl -s -X POST localhost:8000/api/lens -H 'content-type: application/json' -d "{
  \"session_id\": \"$SID\", \"layers\": [0, 1], \"topk\": 8, \"track\": [\"a\"],
  \"targets\": [{\"kind\": \"gen\", \"i\": 0}, {\"kind\": \"patch\", \"i\": 300}]
}" | jq

# Patch attribution for one token (+ the attention-rollout variant)
curl -s -X POST localhost:8000/api/attribution -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"layer\":0,\"token\":\"a\",\"metric\":\"lens_prob\"}" | jq

# Knock out the top-left 4×4 patches and caption again
curl -s -X POST localhost:8000/api/knockout -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"patches\":[0,1,2,3,24,25,26,27],\"mode\":\"zero\"}" | jq

# Steering: add a token direction at layer 0, then check the caption shift
curl -s -X POST localhost:8000/api/steer -H 'content-type: application/json' \
  -d "{\"session_id\":\"$SID\",\"layer\":0,\"mode\":\"add\",\"token\":\"a\",\"alpha\":4.0,\"positions\":\"last\"}" | jq

# Text-only twin: no image, no patches (attribution/knockout answer 409)
curl -s -X POST localhost:8000/api/session -H 'content-type: application/json' \
  -d '{"variant":"no_image","prompt":"Describe the scene"}' | jq '.variant, .image'
```

`targets` kinds: `gen` (0-based index into the generated caption), `prompt` (absolute
`input_ids` position) and `patch` (0…575). `steer` layers are limited to the lens's
fitted `source_layers` — an edit needs the lens direction — while `attribution` may also
read the vanilla final layer.

## Environment variables

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLMJ_BACKEND` | `tiny` | `tiny` (random 3-layer LLaVA on CPU) or `hf-llava` (HF checkpoint) |
| `VLMJ_MODEL` | `llava-hf/llava-1.5-7b-hf` | HF model id for `hf-llava` |
| `VLMJ_LENS_DIR` | unset | Lens artifacts dir (or a single `lens-*.pt`); wins over auto-fit |
| `VLMJ_AUTOFIT` | `1` | `0` disables the tiny auto-fit (lens then unavailable → 409) |
| `VLMJ_CACHE` | `vlmj/.cache` | Cache for the auto-fitted lens and dummy manifest |
| `VLMJ_DEVICE` | `cpu`/`cuda` | `cpu`, `cuda`, `cuda:1`, … |
| `VLMJ_DTYPE` | `fp32`/`bf16` | `float32`, `float16`/`fp16`, `bfloat16`/`bf16` |
| `VLMJ_LENS_SRC` | `/home/vacpls/Workspace/NeuronpediaVLM/src` | Fallback `PYTHONPATH` for the `vlm_lens` source tree |

## Tests

```bash
cd apps/vlm-jlens-demo/backend
~/miniforge3/envs/vlm-lens/bin/python -m pytest tests/test_api_tiny.py -v
```

The suite is network-free, CPU-only and self-contained: every app is built from an
explicit `Config` with a `tmp_path` cache, so it never reads `VLMJ_*` from your shell.
It covers `/api/meta`, samples, session encoding (576 image tokens, quarters), greedy
generation, lens readouts with a tracked token and the `model_row` anchor, both
attribution metrics (including the quarter-mean invariant and `grid_rank`), knockout
deltas + `outcomes`, steering diagnostics, alpha-0 steering, the `no_image` twin, the
validation matrix, the `409` no-lens behaviour, and the mock-lens script round-trip.

A live end-to-end smoke against a running server lives in `scripts/e2e_smoke.py`
(`python scripts/e2e_smoke.py [BASE_URL]`; prints `E2E_PASS` and exercises every
endpoint, including the `alpha: 0` steering identity and the twin's `409`).
