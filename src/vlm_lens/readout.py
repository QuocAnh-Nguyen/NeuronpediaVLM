# SPDX-License-Identifier: Apache-2.0
"""Reading a fitted lens: transport + unembed, position/layer selection, generation traces.

Mirrors the reference's ``JacobianLens.apply`` but on the multimodal path: residuals are
captured with the same forward hooks the fit used (pre-norm block outputs), transported
into the final-layer basis with ``J_l``, and decoded with the model's own unembedding —
``unembed(J_l @ h_l)``, the lens readout. ``lens_readout`` is verified against HF's own
logits by ``scripts/check_equivalence.py`` (the final-layer readout is exact).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch


@dataclass
class LensReadout:
    """Lens and model readouts at selected positions."""

    lens_logits: dict[int, torch.Tensor]  # layer -> [n_positions, vocab] fp32 CPU
    model_logits: torch.Tensor  # [n_positions, vocab] fp32 CPU
    positions: list[int]
    tokens: list[str]  # decoded input tokens at each position
    vocab: list[str]  # id -> display string for the full output vocabulary
    tags: list[str]  # "image" | "text" per position
    input_ids: torch.Tensor  # [1, seq_len] CPU
    seq_len: int

    def top_k(
        self, *, k: int = 5, layers: Sequence[int] | None = None
    ) -> dict[int, list[list[tuple[str, float]]]]:
        """``{layer: [[(token, prob), ...] per position]}`` — the classic lens table."""
        layers = list(layers) if layers is not None else sorted(self.lens_logits)
        out: dict[int, list[list[tuple[str, float]]]] = {}
        for layer in layers:
            probs = self.lens_logits[layer].softmax(dim=-1)
            values, indices = probs.topk(k, dim=-1)
            out[layer] = [
                [(self.vocab[i], float(p)) for p, i in zip(row_values, row_indices, strict=True)]
                for row_values, row_indices in zip(values, indices, strict=True)
            ]
        return out

    def rank_of(self, layer: int, position_index: int, token_id: int) -> int:
        """Full-vocab rank (1-based) of ``token_id`` in the lens distribution."""
        row = self.lens_logits[layer][position_index]
        return int((row > row[token_id]).sum().item()) + 1

    def logit_of(self, layer: int, position_index: int, token_id: int) -> float:
        return float(self.lens_logits[layer][position_index, token_id])


def _decode_tokens(model: LlavaLensModel, ids: torch.Tensor) -> list[str]:
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or not hasattr(tokenizer, "convert_ids_to_tokens"):
        return [str(int(i)) for i in ids]
    return list(tokenizer.convert_ids_to_tokens(ids.tolist()))


def _vocab_strings(model: LlavaLensModel, vocab_size: int) -> list[str]:
    """Id -> display string for the full output vocabulary (padding unmappable ids)."""
    strings: list[str] = []
    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
        try:
            strings = list(tokenizer.convert_ids_to_tokens(list(range(vocab_size))))
        except Exception:  # pragma: no cover - exotic tokenizers
            strings = []
    strings = [str(s) if s is not None else f"id:{i}" for i, s in enumerate(strings)]
    if len(strings) < vocab_size:
        strings.extend(f"id:{i}" for i in range(len(strings), vocab_size))
    return strings[:vocab_size]


@torch.no_grad()
def lens_readout(
    model: LlavaLensModel,
    lens: JacobianLens,
    sample: FitSample | MultimodalBatch | str,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    use_jacobian: bool = True,
    max_seq_len: int = 1536,
    bias: dict[int, torch.Tensor] | None = None,
    scale: dict[int, float] | None = None,
    temp: dict[int, float] | None = None,
    logit_bias: dict[int, torch.Tensor] | None = None,
) -> LensReadout:
    """Lens logits at ``positions`` for every requested layer, plus the model's logits.

    Args:
        layers: Layers to read out; defaults to all fitted ``source_layers``. Must be
            fitted layers when ``use_jacobian`` (the final layer is always available).
        positions: Sequence positions (negative indices allowed). ``None`` reads every
            position — fine on the tiny fixture, potentially large on LLaVA-7B.
        use_jacobian: ``False`` gives the vanilla logit-lens baseline (``unembed`` of the
            raw residual).
        bias: Per-layer additive corrections in residual (d_model) space, applied to the
            transported vector - ``unembed(s_l * (J_l @ h + b_l))``, the affine fix for
            the fitted lens (see the moment census that estimates it). Only layers
            present in the dict are corrected.
        scale: Per-layer scalar multipliers applied to the residual just before
            ``unembed``.
        temp: Per-layer output temperatures applied to the lens logits right after
            ``unembed`` - ``z / temp``. Only layers present in the dict are tempered.
        logit_bias: Per-layer additive corrections in logit (vocab) space, applied after
            the temperature - ``z / temp + logit_bias``. Only layers present in the dict
            are corrected. The final-layer readout (the model's own logits) gets none of
            these corrections.
    """
    batch = as_batch(model, sample, max_seq_len)
    fitted = set(lens.source_layers)
    final_layer = model.n_layers - 1
    if layers is None:
        layers = lens.source_layers
    layers = sorted(set(layers))
    unknown = [
        layer for layer in layers if use_jacobian and layer not in fitted and layer != final_layer
    ]
    if unknown:
        raise ValueError(f"layers {unknown} are not fitted; fitted layers are {lens.source_layers}")
    if any(not 0 <= layer < model.n_layers for layer in layers):
        raise ValueError(f"layers {layers} out of range for a {model.n_layers}-layer model")

    record_at = sorted(set(layers) | {final_layer})
    with ActivationRecorder(model.layers, at=record_at) as recorder:
        model.forward_mm(batch)
        activations = {layer: recorder.activations[layer].detach() for layer in record_at}

    seq_len = batch.seq_len
    if positions is None:
        index_list = list(range(seq_len))
    else:
        index_list = [p if p >= 0 else seq_len + p for p in positions]

    def select(layer: int) -> torch.Tensor:
        full = activations[layer][0]  # [seq_len, d_model]
        return full[index_list].float()

    lens_logits: dict[int, torch.Tensor] = {}
    for layer in layers:
        residual = select(layer)
        if use_jacobian and layer in lens.jacobians:
            residual = lens.transport(residual, layer)
            if bias is not None and layer in bias:
                residual = residual + bias[layer].to(residual.device)
        if scale is not None and layer in scale:
            residual = residual * scale[layer]
        logits = model.unembed(residual).float().cpu()
        if temp is not None and layer in temp:
            logits = logits / temp[layer]
        if logit_bias is not None and layer in logit_bias:
            logits = logits + logit_bias[layer].to(logits.device)
        lens_logits[layer] = logits

    model_logits = model.unembed(select(final_layer)).float().cpu()
    input_ids_cpu = batch.input_ids[0].detach().cpu()
    return LensReadout(
        lens_logits=lens_logits,
        model_logits=model_logits,
        positions=index_list,
        tokens=_decode_tokens(model, input_ids_cpu[index_list]),
        vocab=_vocab_strings(model, model.unembed_weight().shape[0]),
        tags=[
            "image" if bool(batch.image_token_mask[0, index].item()) else "text"
            for index in index_list
        ],
        input_ids=input_ids_cpu,
        seq_len=seq_len,
    )


@torch.no_grad()
def trace_generation(
    model: LlavaLensModel,
    lens: JacobianLens,
    sample: FitSample | MultimodalBatch | str,
    *,
    layers: Sequence[int] | None = None,
    max_new_tokens: int = 64,
    top_k: int = 5,
    use_jacobian: bool = True,
    store_full_logits: bool = False,
    max_seq_len: int = 1536,
) -> dict[str, Any]:
    """Greedy generation with a per-step lens readout at the current last position.

    Captioning analysis entry point: for every generated token, the lens distribution at
    each fitted layer is recorded, so hallucinated objects can be traced back to the layer
    where they enter the J-space. The hook runs inside ``generate``'s forward passes; call
    ``i`` (0-based) predicts generated token ``i``, a convention enforced by asserting the
    number of hook records equals the number of new tokens.
    """
    batch = as_batch(model, sample, max_seq_len)
    layers = sorted(set(layers if layers is not None else lens.source_layers))
    final_layer = model.n_layers - 1
    record_at = sorted(set(layers) | {final_layer})
    first_layer = record_at[0]

    step_records: list[dict[int, dict[str, Any]]] = []

    def make_hook(layer: int):
        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            tensor = output if torch.is_tensor(output) else output[0]
            hidden = tensor[0, -1].float()
            residual = (
                lens.transport(hidden, layer) if (use_jacobian and layer in lens.jacobians) else hidden
            )
            logits = model.unembed(residual).float().cpu()
            probs = logits.softmax(dim=-1)
            values, indices = probs.topk(top_k)
            entry: dict[str, Any] = {
                "top_tokens": [
                    (int(i), float(v))
                    for i, v in zip(indices.tolist(), values.tolist(), strict=True)
                ]
            }
            if store_full_logits:
                entry["logits"] = logits.half()
            # Blocks fire in call order within one forward pass, so the first recorded
            # layer opening a new call is exactly where a new step record starts.
            if layer == first_layer:
                step_records.append({})
            step_records[-1][layer] = entry

        return hook

    handles = [
        model.layers[layer].register_forward_hook(make_hook(layer)) for layer in record_at
    ]
    try:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
        )
    finally:
        for handle in handles:
            handle.remove()

    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    token_strings = _decode_tokens(model, new_tokens)
    if len(step_records) != len(new_tokens):
        raise RuntimeError(
            f"hook recorded {len(step_records)} forwards but {len(new_tokens)} tokens were generated; "
            "the step alignment convention does not hold for this generate() configuration"
        )
    return {
        "text": model._decode(new_tokens),
        "tokens": token_strings,
        "token_ids": [int(t) for t in new_tokens],
        "steps": step_records,
        "layers": layers,
        "top_k": top_k,
        "seq_len": batch.seq_len,
    }


__all__ = ["LensReadout", "lens_readout", "trace_generation"]
