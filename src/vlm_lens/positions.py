# SPDX-License-Identifier: Apache-2.0
"""Source-position masks for multimodal Jacobian fitting.

The paper's estimator averages ``dh_final[p'] / dh_l[p]`` over a set of *source*
positions ``p``. For text-only models the reference implementation uses
"all positions after the first 16, excluding the last" (attention sinks and the final
position, which has no next-token target). A VLM sequence additionally mixes two kinds
of tokens, which call for separate reductions:

``text``
    Text tokens only — the reduction that matches the paper's semantics ("what is this
    state disposed to say") and the primary lens for captioning analysis.
``image``
    Image placeholder positions — the fused visual patch states. A readout here is not a
    next-token prediction but a first-order disposition to verbalize; validate with
    :mod:`vlm_lens.evaluate` before drawing conclusions.
``all``
    Everything valid (dominated by image tokens when images are present).
``image-q0``..``image-q3``
    The *valid* image positions split into four contiguous quarters in sequence order
    (X2). Causal masking inside the block (F8) gives each quarter different information
    content and different norms, so a block average can hide a positional gradient.

All masks exclude the leading ``skip_first`` positions (BOS is an attention sink) and,
by default, the final position (no next-token target). For text-only control fits use
``skip_first=16`` to match the paper exactly; for multimodal fits ``skip_first=1`` is
appropriate because each of the 576 image positions carries patch-level content.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch

DEFAULT_MASKS: tuple[str, ...] = ("text", "image", "all")
#: X2: the valid image positions split into four contiguous quarters, in sequence order.
IMAGE_QUARTER_MASKS: tuple[str, ...] = ("image-q0", "image-q1", "image-q2", "image-q3")
#: Every mask name this module can build.
ALL_MASKS: tuple[str, ...] = DEFAULT_MASKS + IMAGE_QUARTER_MASKS


def build_position_masks(
    input_ids: torch.Tensor,
    image_token_id: int,
    *,
    skip_first: int = 1,
    exclude_last: bool = True,
    masks: Sequence[str] = DEFAULT_MASKS,
) -> dict[str, torch.Tensor]:
    """Boolean ``[seq_len]`` masks selecting which source positions feed the average.

    Args:
        input_ids: ``[seq_len]`` or ``[1, seq_len]`` token ids.
        image_token_id: Id of the image placeholder token, e.g. 32000 for LLaVA-1.5.
        skip_first: Leading positions to drop (attention sinks).
        exclude_last: Drop the final position, which has no next-token target.
        masks: Subset of :data:`ALL_MASKS`; ``image-q0``..``image-q3`` are contiguous
            quarters of the (post-``skip_first``) image positions, so each quarter of a
            sample holds the same number of positions.

    Returns:
        ``{name: BoolTensor[seq_len]}``. Masks may be empty (e.g. ``image`` for a
        text-only sequence); the fit loop decides per sample whether that is fatal
        or merely means "no contribution to that reduction".
    """
    if input_ids.dim() == 2:
        if input_ids.shape[0] != 1:
            raise ValueError(f"expected a single sequence, got shape {tuple(input_ids.shape)}")
        input_ids = input_ids[0]
    input_ids = input_ids.reshape(-1)
    seq_len = int(input_ids.shape[0])
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")

    unknown = sorted(set(masks) - set(ALL_MASKS))
    if unknown:
        raise ValueError(f"unknown masks {unknown}; expected subset of {list(ALL_MASKS)}")

    valid = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
    stop = seq_len - 1 if exclude_last else seq_len
    if skip_first < stop:
        valid[skip_first:stop] = True

    is_image = (input_ids == image_token_id) & valid
    quarters: dict[str, torch.Tensor] = {}
    if any(name.startswith("image-q") for name in masks):
        groups = torch.tensor_split(is_image.nonzero(as_tuple=True)[0], 4)
        for index, group in enumerate(groups):
            quarter = torch.zeros(seq_len, dtype=torch.bool, device=input_ids.device)
            quarter[group] = True
            quarters[f"image-q{index}"] = quarter
    out: dict[str, torch.Tensor] = {}
    for name in masks:
        if name == "text":
            out[name] = valid & ~is_image
        elif name == "image":
            out[name] = is_image
        elif name == "all":
            out[name] = valid
        else:  # image-q0 .. image-q3
            out[name] = quarters[name]
    return out


def mask_summary(
    masks: Mapping[str, torch.Tensor],
    *,
    input_ids: torch.Tensor | None = None,
    image_token_id: int | None = None,
) -> dict[str, dict[str, float | int]]:
    """Per-mask counts plus, when ``input_ids`` is given, the image-token fraction."""
    summary: dict[str, dict[str, float | int]] = {}
    total = None
    if input_ids is not None and image_token_id is not None:
        flat = input_ids.reshape(-1)
        total = int(flat.shape[0])
    for name, mask in masks.items():
        n = int(mask.sum())
        entry: dict[str, float | int] = {"n_positions": n}
        if total:
            entry["frac_of_seq"] = round(n / total, 4)
        summary[name] = entry
    return summary


__all__ = [
    "ALL_MASKS",
    "DEFAULT_MASKS",
    "IMAGE_QUARTER_MASKS",
    "build_position_masks",
    "mask_summary",
]
