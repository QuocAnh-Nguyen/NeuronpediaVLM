# vlm-jlens-demo — interactive J-Lens workbench for LLaVA

A local research tool for **captioning hallucination** in
`llava-hf/llava-1.5-7b-hf`: Jacobian-lens (J-Lens) readouts over image patches and generated
tokens, per-patch attribution, causal patch knockout through the multimodal projector, and
interactive residual-stream steering. One session = one image + one prompt + one greedy caption;
every intervention reports the caption it produced. Standalone and additive: nothing outside this
directory is modified, and it is the MVP1 tool described in
[`../../docs/vlm-jlens-mvp1.md`](../../docs/vlm-jlens-mvp1.md) (frozen HTTP contract, VRAM budget,
upstream integration path). Technique choices and citations:
[`../../docs/vlm-viz-research.md`](../../docs/vlm-viz-research.md).

## Layout

- `backend/` — FastAPI service (`vlmj/{app,engine,jobs,schemas}.py`); `backend/README.md` is the
  canonical API reference, `vlmj/engine.py` wraps the workspace `vlm_lens` library.
- `frontend/` — Vite + React 18 + TypeScript UI (`src/views/{Session,Caption,Patch,Steer}View.tsx`,
  `src/lib/types.ts` mirrors the contract); `frontend/README.md` covers the views.
- `backend/scripts/make_mock_lens.py` — random-but-valid `lens-*.pt` sets for UI bring-up.
- `backend/tests/test_api_tiny.py` — network-free, CPU-only end-to-end contract tests.

## 1. Local tiny stack (CPU, offline, default)

The first request builds a random 3-layer LLaVA and auto-fits a demo lens on synthetic images
(≈0.3 s, cached under `vlmj/.cache`). No downloads, no GPU.

```bash
cd apps/vlm-jlens-demo/backend
~/miniforge3/envs/vlm-lens/bin/python -m vlmj.app --port 8787   # matches the Vite proxy

# second shell
cd apps/vlm-jlens-demo/frontend
npm install
npm run dev                                                     # http://localhost:5173
```

The Vite dev server proxies `/api/*` to `http://127.0.0.1:8787`. Alternatively build the UI once
and let the backend serve it same-origin (no proxy): `npm run build` in `frontend/`, then open
`http://127.0.0.1:8787`.

## 2. Fixtures mode (UI only, no backend)

```bash
cd apps/vlm-jlens-demo/frontend && npm run dev
# open http://localhost:5173/?fixtures=1     (or VITE_FIXTURES=1 npm run dev)
```

Every view is explorable offline against deterministic fixtures; jobs still run through
`queued → running → done` and every control changes the output.

## 3. LLaVA-1.5-7B on a GPU

```bash
cd apps/vlm-jlens-demo/backend
VLMJ_BACKEND=hf-llava VLMJ_MODEL=llava-hf/llava-1.5-7b-hf \
VLMJ_LENS_DIR=/data/lenses/llava-1.5-7b/text \
VLMJ_DEVICE=cuda VLMJ_DTYPE=bf16 \
~/miniforge3/envs/vlm-lens/bin/python -m vlmj.app --port 8787
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLMJ_BACKEND` | `tiny` | `tiny` (random 3-layer LLaVA on CPU) or `hf-llava` (HF checkpoint) |
| `VLMJ_MODEL` | `llava-hf/llava-1.5-7b-hf` | HF model id for `hf-llava` |
| `VLMJ_LENS_DIR` | unset | Lens artifacts dir or single `lens-*.pt`; wins over auto-fit. Partial sets work (one mask is loaded, preference `text` → `all` → `image`); a broken set gives `409`, never a crash |
| `VLMJ_DEVICE` | `cpu` (tiny) / `cuda` (hf-llava) | `cpu`, `cuda`, `cuda:1`, … |
| `VLMJ_DTYPE` | `fp32` (tiny) / `bf16` (hf-llava) | `float32`, `float16`, `bfloat16` |
| `VLMJ_AUTOFIT` / `VLMJ_CACHE` / `VLMJ_LENS_SRC` | `1` / `vlmj/.cache` / workspace `src/` | Tiny auto-fit on/off, cache dir, fallback `PYTHONPATH` for `vlm_lens` |

One GPU process at a time: model work is serialized behind a single worker and a global lock, and
the tool must not co-run with a lens-fitting campaign (deployment decision D8 in the MVP1 doc).
`/api/meta` reports free VRAM; the CPU test suite runs with
`~/miniforge3/envs/vlm-lens/bin/python -m pytest tests/test_api_tiny.py -v` from `backend/`.
