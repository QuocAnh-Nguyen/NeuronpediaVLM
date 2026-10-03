# VLM J-Lens demo — frontend

Local research UI for **Jacobian-lens readouts, patch attribution, causal knockout and lens-vector
steering** on a VLM (LLaVA-1.5-7B, or the `tiny` CPU mode for development). Built with
Vite + React 18 + TypeScript; no UI kit, no chart library.

## Prerequisites

- Node.js ≥ 20.19 (developed on Node 24) and npm.
- For real readouts: the sibling backend listening on `http://127.0.0.1:8787`
  (FastAPI, frozen HTTP contract — mirrored field-for-field in `src/lib/types.ts`).
  Without it, use fixtures mode (below).

## Quick start

```bash
cd apps/vlm-jlens-demo/frontend
npm install
npm run dev          # http://localhost:5173, /api proxied to http://127.0.0.1:8787
```

The dev server proxies every `/api/*` request to `http://127.0.0.1:8787`
(see `vite.config.ts`). The API client uses a same-origin base URL (`''`), so nothing else needs
configuring. If `/api/meta` is unreachable the header shows an offline banner with a retry button.

### Fixtures mode (no backend)

Everything is explorable offline against the deterministic fixtures in `src/fixtures/`:

```bash
npm run dev
# open http://localhost:5173/?fixtures=1
# or export the flag instead:
VITE_FIXTURES=1 npm run dev
```

`?fixtures=1` (checked once at load) or `VITE_FIXTURES=1|true|yes|on` routes every API call to
`src/lib/fixtures.ts`. Fixture jobs still go `queued → running → done` on the normal ~700 ms poll
interval, and every control changes the output (layer/metric/rollout change the attribution field,
the tracked token changes the lens curve, the painted patches change the knockout deltas, the steer
mode/alpha/positions change the edited caption). A `fixtures` chip appears in the status bar.

The **Caption & lens** tab can create a **no-image twin**: a second session with `variant: "no_image"`
(same prompt, no `<image>` block) whose caption and last-layer lens probabilities are compared side
by side with the image session, labelled `[ADAPTATION] prior-gap proxy`. Attribution always reports a
24×24 `grid_rank` field (1-based; `null` for `attn_rollout`), and every lens result carries the
model's own `model_row` logits next to the per-layer lens readouts.

## Production build

```bash
npm run build        # tsc --noEmit && vite build  →  dist/
npm run preview      # serve dist/ locally (no proxy: use fixtures mode for a backend-free check)
```

Serve `dist/` as the static root of the backend process so `/api` stays same-origin in production —
no proxy is required there. `dist/` is the configured `build.outDir`.

## Views

| Tab | What it does | Endpoints |
| --- | --- | --- |
| **Session** | Pick one of the three fixture images or upload your own, edit the prompt (`<image>` marks the image block), create a session, run greedy generation. | `GET /api/samples`, `POST /api/session`, `GET /api/session/{id}`, `POST /api/generate` |
| **Caption & lens** | Click any generated token to read out the lens for that `gen` position: a layers × top-k matrix with probability bars (`prob`/`rank` shading toggle — rank is the default when the matrix compares layers), the tracked token's rank/probability per layer and a probability-vs-layer chart. The model's own logits (`model_row`) render as a distinct **model output** row, and the final lens layer is labelled the same way because it reproduces them. Flag suspicious tokens with notes and export the list as TSV, or compare against the **no-image twin**. | `POST /api/lens`, `GET /api/jobs/{id}` |
| **Patch attribution** | Canvas heatmap (viridis) over the session image on the 24×24 patch grid: layer slider, metric (`lens_prob` / `lens_logit` / `attn_rollout`), rollout toggle, target token, quarter outlines and per-quarter means. Click/drag to paint patches, `select top-64 by heat`, click a patch to read out its lens distribution. Hover shows the patch index, score, 1-based `grid_rank` (hidden for `attn_rollout`) and quarter. The color scale defaults to the per-grid z-score (mean 0, symmetric) with a `raw` toggle. | `POST /api/attribution`, `POST /api/lens` |
| **Knockout & steering** | Apply a lens-vector edit (`add` / `ablate` / `swap`, with residual-relative alpha — preset buttons 0.5 / 1 / 2 / 3 — and `last`/`all` positions) and diff the caption before/after; the per-edit diagnostics table lists `v_norm`, `h_norm` and `cond`, and a warning badge appears when `cond > 1e3` (*nearly collinear pair - swap unstable*). Or knock out the patches painted in the Patch tab (`zero` / `mean`) and inspect per-token log-probability deltas as signed bars, with the classified `outcomes` (removed / persisted / changed row colors, `classification_note` as tooltip). | `POST /api/steer`, `POST /api/knockout` |

Shared state: the session/image/caption, flagged tokens and painted patch selection live in
`src/App.tsx`, so flags seed the token inputs of the intervention views and the painted patches feed
the knockout tool.

## Implementation notes

- `src/lib/types.ts` mirrors the frozen HTTP contract; `src/lib/api.ts` is the typed client
  (`ApiError` normalizes `{detail}` bodies, `pollJob` polls `GET /api/jobs/{id}` every ~700 ms).
- The heatmap is a `<canvas>` overlay sized to the displayed image box (`src/components/HeatmapGrid.tsx`);
  charts are plain SVG (`src/components/LayerChart.tsx`) and tables are plain HTML.
- Theming lives in a single `src/styles.css` (background `#0f1117`, panels `#171a23`, accent `#6d7cff`,
  tabular monospace for numbers).
- `npm run typecheck` runs `tsc --noEmit`; the build script already includes it.
- `src/lib/es2024.d.ts` declaration-merges `PromiseConstructor.withResolvers` so the code can use the
  ES2024 helper while `tsconfig.json` targets ES2022 (the pinned TypeScript 5.6 does not ship an
  `ES2024` lib). It is a global script on purpose — never add `import`/`export` to it.
- The extended contract fields (`model_row`, `grid_rank`, `outcomes` / `classification_note`,
  `diagnostics`, session `variant`) are additive and mirrored field-for-field in
  `src/fixtures/*.json`; fixtures mode covers the no-image twin (`session_no_image.json`) and
  deliberately emits `cond > 1e3` swap diagnostics so the collinearity warning is exercisable offline.
