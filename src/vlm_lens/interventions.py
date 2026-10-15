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
    ``h - alpha * (h . v̂_t) v̂_t`` (removing token ``t``'s direction). With ``mean=m``
    set, the removed projection is ``(h - m) . v̂_t`` instead — the Belrose App. D mean
    replacement ``h' = h + P_v(m - h)``: the residual's ``v̂_t``-component is swapped for
    the mean residual's, so at ``alpha=1`` the ``(h - m)`` projection along ``v̂_t`` is
    exactly zero. ``m`` is a per-layer tensor on the edit spec (excluded from
    ``to_json``). With ``position_ids=(...)`` set, ``ablate`` is instead a WHOLE-ROW
    replacement at those absolute sequence positions — ``mean=m`` fills the rows with
    ``m``, ``mean=None`` zeroes them — aimed at image-token positions (which live only
    in the prefill forward; later single-token decode forwards cannot contain them, so
    the edit is a no-op there). This vector-free form needs no lens.
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
from dataclasses import asdict, dataclass, field
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
    """Apply one edit to a residual tensor ``[..., d_model]`` (dtype preserved).

    For ``ablate``, ``edit.mean`` (when given) retargets the projection removal at
    ``h - m``: the ``v̂_t``-component of ``h`` is replaced by the mean residual's
    (``h' = h + P_v(m - h)``); ``mean=None`` removes ``h``'s component outright.
    """
    dtype = hidden.dtype
    h = hidden.float()
    if edit.mode == "add":
        out = h + edit.alpha * _aligned(vectors["target"], h)
    elif edit.mode == "ablate":
        direction = _as_direction(_aligned(vectors["target"], h))
        if edit.mean is None:
            coeff = (h * direction).sum(dim=-1, keepdim=True)
        else:
            mean = _aligned(edit.mean, h)
            coeff = ((h - mean) * direction).sum(dim=-1, keepdim=True)
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
    """One edit spec: what to do, at which layer, with which vector(s).

    ``mean`` (``ablate`` only) is the per-layer mean residual for the Belrose App. D
    mean replacement (see module docstring). It is a tensor, so it opts out of equality
    and hash (``compare=False``) and is excluded from :meth:`to_json` — a tensor field
    would otherwise break both frozen-dataclass comparison and JSON spec reporting.

    ``position_ids`` (``ablate`` only) switches the edit to the vector-free whole-row
    replacement of the module docstring; the ``positions`` string scope is then not
    consulted (``position_ids`` IS the scope).
    """

    layer: int
    mode: str
    alpha: float = 1.0
    token: str | None = None  # target token (all modes)
    source_token: str | None = None  # source token for "swap"
    positions: str = "last"  # "last" (current token) | "all"
    #: Absolute prefill positions for the vector-free ``ablate`` row replacement (see
    #: the module docstring). When set, the ``positions`` string scope is not
    #: consulted — ``position_ids`` IS the scope — and the edit needs no vectors.
    position_ids: tuple[int, ...] | None = None
    mean: torch.Tensor | None = field(default=None, compare=False, repr=False)

    def __post_init__(self) -> None:
        if self.mode not in EDIT_MODES:
            raise ValueError(f"mode must be one of {EDIT_MODES}, got {self.mode!r}")
        if self.position_ids is not None:
            # vector-free whole-row replacement: its own constraints, no token roles
            if self.mode != "ablate":
                raise ValueError(
                    f"position_ids= is only used with mode 'ablate', got {self.mode!r}"
                )
            if self.token is not None or self.source_token is not None:
                raise ValueError(
                    "position_ids= edits replace whole residual rows and take no "
                    "token=/source_token="
                )
            if any(int(p) < 0 for p in self.position_ids):
                raise ValueError(
                    f"position_ids must be non-negative, got {self.position_ids!r}"
                )
        else:
            if self.mode in {"add", "ablate"} and self.token is None:
                raise ValueError(f"mode {self.mode!r} needs token=")
            if self.mode == "swap" and (self.token is None or self.source_token is None):
                raise ValueError("mode 'swap' needs token= (target) and source_token=")
            if self.positions not in {"last", "all"}:
                raise ValueError(
                    f"positions must be 'last' or 'all', got {self.positions!r}"
                )
        if self.mean is not None and self.mode != "ablate":
            raise ValueError(f"mean= is only used with mode 'ablate', got {self.mode!r}")

    def to_json(self) -> dict[str, Any]:
        payload = asdict(self)
        del payload["mean"]  # tensor field: JSON specs carry the mode/alpha/token only
        return payload


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
            tokens = [t for t in (edit.source_token, edit.token) if t is not None]
            if tokens:
                rows = lens_vectors(model, lens, tokens, layers=[edit.layer])[edit.layer]
                resolved[edit.layer] = {
                    token: rows[index] for index, token in enumerate(tokens)
                }
            else:  # position_ids row replacement: no vectors to resolve
                resolved[edit.layer] = {}
        vectors: dict[str, torch.Tensor] = {}
        if edit.source_token is not None:
            vectors["source"] = resolved[edit.layer][edit.source_token]
        if edit.token is not None:
            vectors["target"] = resolved[edit.layer][edit.token]
        grouped.setdefault(edit.layer, []).append((edit, vectors))
    return grouped


def _replace_position_rows(hidden: torch.Tensor, edit: ResidualEdit) -> torch.Tensor:
    """The ``position_ids`` ablate branch: replace whole residual rows ``[batch, seq, d]``.

    Rows outside the current forward's sequence are skipped, so a prefill-scoped
    position edit is a no-op on the later single-token decode forwards (image-token
    positions do not recur under KV caching). ``edit.mean`` (when given) fills the rows
    with the mean residual; ``mean=None`` zeroes them.
    """
    seq = int(hidden.shape[1])
    positions = [int(p) for p in edit.position_ids or () if 0 <= int(p) < seq]
    if not positions:
        return hidden
    index = torch.tensor(positions, dtype=torch.long, device=hidden.device)
    if edit.mean is not None:
        row = _aligned(edit.mean, hidden).to(hidden.dtype)
    else:
        row = torch.zeros(int(hidden.shape[2]), dtype=hidden.dtype, device=hidden.device)
    replacement = row.expand(len(positions), -1).unsqueeze(0).expand(hidden.shape[0], -1, -1)
    return hidden.index_copy(1, index, replacement)


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
                if edit.position_ids is not None:
                    hidden = _replace_position_rows(hidden, edit)
                elif edit.positions == "last":
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
    if edits and lens is None and any(edit.position_ids is None for edit in edits):
        raise ValueError("edits require a fitted lens (for the J-lens vectors); "
                         "position_ids row-replacement edits are the exception")
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
