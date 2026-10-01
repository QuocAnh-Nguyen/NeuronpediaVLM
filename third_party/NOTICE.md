# Third-party code

## `third_party/jacobian-lens` — Anthropic Jacobian Lens reference implementation

- Source: <https://github.com/anthropics/jacobian-lens>
- Vendored at commit `581d398613e5602a5af361e1c34d3a92ea82ba8e` (branch `main`), unmodified.
- License: Apache-2.0 (see `third_party/jacobian-lens/LICENSE`).
- Companion code for *"Verbalizable Representations Form a Global Workspace in Language
  Models"*, <https://transformer-circuits.pub/2026/workspace/index.html>.

Why vendored rather than installed: the fitting estimator must stay byte-for-byte the
published one so that `vlm_lens` extensions (multimodal inputs, modality-aware position
masks) can be diffed against it, and so that artifacts stay loadable by upstream
`jlens.JacobianLens` and by TransformerLens's `JacobianLens` without a network install.

Files under `third_party/jacobian-lens` are third-party code and follow their original
license; no modifications are made to them. All extensions live under `src/vlm_lens/`.
