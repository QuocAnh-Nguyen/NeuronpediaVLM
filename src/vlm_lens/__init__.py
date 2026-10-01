# SPDX-License-Identifier: Apache-2.0
"""``vlm_lens`` — Jacobian lenses for vision-language models (LLaVA-1.5-7B first).

Fit: average input->output Jacobian of the LLM residual stream, estimated with the
published estimator from ``third_party/jacobian-lens`` (Anthropic, Apache-2.0), extended
with modality-aware source-position masks for image tokens. Apply: linear transport of a
residual into the final-layer basis, decoded with the model's own unembedding.

Public surface:

* :class:`vlm_lens.models.llava.LlavaLensModel` — multimodal ``LensModel`` over HF LLaVA.
* :func:`vlm_lens.fitting.jacobian_for_sample`, :func:`vlm_lens.fitting.fit_masked`.
* :mod:`vlm_lens.artifacts` — save/load/merge + provenance, upstream-compatible.
* :mod:`vlm_lens.readout`, :mod:`vlm_lens.interventions` — analysis and causal edits.
"""

from vlm_lens._vendor import ensure_jlens

JLENS_SOURCE = ensure_jlens()

__all__ = ["JLENS_SOURCE", "ensure_jlens"]
