# SPDX-License-Identifier: Apache-2.0
"""Causal interventions on the residual stream via J-lens vectors.

The paper's interventions operate on *J-lens vectors* — the rows of ``W_U J_l``, one per
vocabulary token — because ``lens(h) = softmax(W_U norm(J_l h))`` makes
``v_t = (W_U J_l)[t]`` the direction whose projection drives token ``t``'s readout. Three
edits are applied to the residual stream at a chosen block (that block's output, i.e. the
same tensor the lens reads):

``add``
    ``h + alpha * v_t`` (steering toward a concept).
``ablate``
    ``h - alpha * (h . v̂_t) v̂_t`` (removing token ``t``'s direction).
``swap``
    replace the coordinates of ``h`` in the subspace spanned by ``[v_s, v_t]`` with their
    swapped values, using the pseudoinverse basis — the paper's "coordinate swap", the
    causal cousin of "clamp this token's lens coordinate onto that token's".

Edits are installed as forward hooks, so they work for a single readout forward and for
every step of ``generate`` (hallucination experiments: does steering or ablating the
hallucinated object's direction change what the caption says?).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch
from jlens.lens import JacobianLens

from vlm_lens._batch import as_batch
from vlm_lens.data.manifest import FitSample
from vlm_lens.models.llava import LlavaLensModel, MultimodalBatch

EDIT_MODES = ("add", "ablate", "swap")


def _token_id(model: LlavaLensModel, token: str) -> int:
    ids = model.tokenizer(token, add_special_tokens=False)["input_ids"]
    if isinstance(ids, torch.Tensor):
        ids = ids.reshape(-1).tolist()
    ids = list(ids)
    if not ids:
        raise ValueError(f"token string {token!r} tokenizes to nothing")
    return int(ids[-1])


def lens_vectors(
    model: LlavaLensModel,
    lens: JacobianLens,
    tokens: Sequence[str],
    *,
    layers: Sequence[int] | None = None,
    normalize: bool = False,
) -> dict[int, torch.Tensor]:
    """J-lens vectors ``v_t = (W_U J_l)[t]`` for the given token *strings*.

    ``v_t`` is the gradient of the lens's linear readout ``h -> W_U[t] . (J_l h)`` with
    respect to ``h`` — a *row* of ``W_U J_l``, hence ``weight @ J`` (a transposed ``J``
    yields a different vector of similar scale, so the orientation is load-bearing).

    Returns ``{layer: [n_tokens, d_model] fp32 CPU}``. Each string is tokenized with the
    model's tokenizer and its last token id is used (the token that would be emitted at
    that position). ``normalize=True`` unit-normalises each row.
    """
    token_ids = [_token_id(model, token) for token in tokens]
    weight = model.unembed_weight().detach().float()  # [vocab, d_model]
    layers = list(layers) if layers is not None else lens.source_layers
    out: dict[int, torch.Tensor] = {}
    for layer in layers:
        J = lens.jacobians[layer].to(weight.device).float()
        vectors = (weight[token_ids] @ J).cpu()
        if normalize:
            vectors = vectors / vectors.norm(dim=-1, keepdim=True).clamp_min(1e-8)
        out[layer] = vectors
    return out


def _as_direction(vector: torch.Tensor) -> torch.Tensor:
    return vector.float() / vector.float().norm().clamp_min(1e-8)


def _aligned(vector: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
    """``vector`` flat, as ``like``'s dtype and on ``like``'s device.

    ``lens_vectors`` deliberately returns CPU fp32 (the fitted ``J`` lives on CPU), while
    a cluster forward runs bf16 on CUDA — applying a vector without this alignment fails
    with "Expected all tensors to be on the same device".
    """
    return vector.to(device=like.device, dtype=like.dtype).reshape(-1)


def apply_edit(
    hidden: torch.Tensor,
    edit: ResidualEdit,
    vectors: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply one edit to a residual tensor ``[..., d_model]`` (dtype preserved)."""
    dtype = hidden.dtype
    h = hidden.float()
    if edit.mode == "add":
        out = h + edit.alpha * _aligned(vectors["target"], h)
    elif edit.mode == "ablate":
        direction = _as_direction(_aligned(vectors["target"], h))
        coeff = (h * direction).sum(dim=-1, keepdim=True)
        out = h - edit.alpha * coeff * direction
    elif edit.mode == "swap":
        basis = torch.stack(
            [_aligned(vectors["source"], h), _aligned(vectors["target"], h)], dim=1
        )  # [d_model, 2]
        pinv = torch.linalg.pinv(basis)  # [2, d_model]
        coords = h @ pinv.T  # [..., 2]
        delta = (coords.flip(-1) - coords) @ basis.T  # [..., d_model]
        out = h + edit.alpha * delta
    else:  # pragma: no cover - validated in ResidualEdit.__post_init__
        raise ValueError(f"unknown edit mode {edit.mode!r}")
    return out.to(dtype)


@dataclass(frozen=True)
class ResidualEdit:
    """One edit spec: what to do, at which layer, with which vector(s)."""

    layer: int
    mode: str
    alpha: float = 1.0
    token: str | None = None  # target token (all modes)
    source_token: str | None = None  # source token for "swap"
    positions: str = "last"  # "last" (current token) | "all"

    def __post_init__(self) -> None:
        if self.mode not in EDIT_MODES:
            raise ValueError(f"mode must be one of {EDIT_MODES}, got {self.mode!r}")
        if self.mode in {"add", "ablate"} and self.token is None:
            raise ValueError(f"mode {self.mode!r} needs token=")
        if self.mode == "swap" and (self.token is None or self.source_token is None):
            raise ValueError("mode 'swap' needs token= (target) and source_token=")
        if self.positions not in {"last", "all"}:
            raise ValueError(f"positions must be 'last' or 'all', got {self.positions!r}")

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def _vectors_for(
    model: LlavaLensModel,
    lens: JacobianLens,
    edits: Sequence[ResidualEdit],
) -> dict[int, list[tuple[ResidualEdit, dict[str, torch.Tensor]]]]:
    """Resolve every edit's vectors, grouped by layer (hooks need them pre-computed)."""
    resolved: dict[int, dict[str, torch.Tensor]] = {}
    grouped: dict[int, list[tuple[ResidualEdit, dict[str, torch.Tensor]]]] = {}
    for edit in edits:
        if edit.layer not in resolved:
            tokens: list[str] = []
            for role_token in (edit.source_token, edit.token):
                if role_token is not None and role_token not in tokens:
                    tokens.append(role_token)
            rows = lens_vectors(model, lens, tokens, layers=[edit.layer])[edit.layer]
            resolved[edit.layer] = {
                token: rows[index] for index, token in enumerate(tokens)
            }
        vectors: dict[str, torch.Tensor] = {}
        if edit.source_token is not None:
            vectors["source"] = resolved[edit.layer][edit.source_token]
        if edit.token is not None:
            vectors["target"] = resolved[edit.layer][edit.token]
        grouped.setdefault(edit.layer, []).append((edit, vectors))
    return grouped


class ResidualEditor:
    """Context manager that applies :class:`ResidualEdit` specs to block outputs."""

    def __init__(
        self,
        model: LlavaLensModel,
        lens: JacobianLens,
        edits: Sequence[ResidualEdit],
    ) -> None:
        self._model = model
        self._grouped = _vectors_for(model, lens, list(edits))
        self._handles: list[Any] = []
        self.forward_calls = 0

    def __enter__(self) -> ResidualEditor:
        for layer, entries in self._grouped.items():
            self._handles.append(
                self._model.layers[layer].register_forward_hook(self._make_hook(entries))
            )
        return self

    def __exit__(self, *exc: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def _make_hook(
        self, entries: Sequence[tuple[ResidualEdit, dict[str, torch.Tensor]]]
    ) -> Callable[..., Any]:
        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            tensor = output if torch.is_tensor(output) else output[0]
            hidden = tensor
            for edit, vectors in entries:
                if edit.positions == "last":
                    hidden = torch.cat(
                        [hidden[:, :-1], apply_edit(hidden[:, -1:], edit, vectors)], dim=1
                    )
                else:
                    hidden = apply_edit(hidden, edit, vectors)
            self.forward_calls += 1
            if torch.is_tensor(output):
                return hidden
            return (hidden, *output[1:])

        return hook


class _NullEditor:
    """No-op editor so the generation path is identical with and without edits."""

    forward_calls = 0

    def __enter__(self) -> _NullEditor:
        return self

    def __exit__(self, *exc: Any) -> None:
        return None


@torch.no_grad()
def generate_with_edits(
    model: LlavaLensModel,
    lens: JacobianLens | None,
    sample: FitSample | MultimodalBatch | str,
    *,
    edits: Sequence[ResidualEdit] = (),
    max_new_tokens: int = 64,
    max_seq_len: int = 1536,
    **generate_kwargs: Any,
) -> dict[str, Any]:
    """Greedy generation with the given edits active (empty ``edits`` = baseline).

    Returns ``{"text", "token_ids", "prompt_len", "edits", "n_edit_forwards"}``.
    """
    if edits and lens is None:
        raise ValueError("edits require a fitted lens (for the J-lens vectors)")
    batch = as_batch(model, sample, max_seq_len)
    editor_ctx: Any = (
        ResidualEditor(model, lens, list(edits)) if edits else _NullEditor()  # type: ignore[arg-type]
    )
    with editor_ctx as editor:
        generated = model.hf_model.generate(
            **batch.hf_kwargs(),
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            **generate_kwargs,
        )
    new_tokens = generated[0, batch.seq_len :].detach().cpu()
    return {
        "text": model._decode(new_tokens),
        "token_ids": [int(t) for t in new_tokens],
        "prompt_len": batch.seq_len,
        "edits": [edit.to_json() for edit in edits],
        "n_edit_forwards": getattr(editor, "forward_calls", 0),
    }


__all__ = [
    "EDIT_MODES",
    "ResidualEdit",
    "ResidualEditor",
    "apply_edit",
    "generate_with_edits",
    "lens_vectors",
]
