# SPDX-License-Identifier: Apache-2.0
"""Request bodies for the demo API.

Responses are plain JSON dictionaries built by :mod:`vlmj.engine`; the shapes are
the frozen wire contract documented in ``README.md``. Only request validation is
modelled here, so FastAPI rejects malformed bodies with a 422 before any model
work starts.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

#: Hard cap on generated tokens per job (also advertised in ``/api/meta``).
MAX_NEW_TOKENS_CAP = 64

AttributionMetric = Literal["lens_prob", "lens_logit", "attn_rollout"]
SteerMode = Literal["add", "ablate", "swap"]
KnockoutMode = Literal["zero", "mean"]
SteerPositions = Literal["last", "all"]
TargetKind = Literal["gen", "prompt", "patch"]


class SessionRequest(BaseModel):
    """``POST /api/session``: one image source for ``image``, none for the twin.

    ``variant`` defaults to ``'image'``; ``'no_image'`` builds the text-only twin
    (the literal ``<image>`` placeholder is stripped from the prompt).
    """

    variant: str = "image"
    image_b64: str | None = None
    image_path: str | None = None
    prompt: str | None = None


class GenerateRequest(BaseModel):
    """``POST /api/generate`` — greedy caption with per-token logprobs."""

    session_id: str
    max_new_tokens: int = Field(default=24, ge=1, le=MAX_NEW_TOKENS_CAP)


class LensTarget(BaseModel):
    """One position to read the lens at.

    ``gen`` is a 0-based index into the session's generated caption, ``prompt``
    an absolute position in ``input_ids`` and ``patch`` a 0..575 image-patch
    index in row-major order.
    """

    kind: TargetKind
    i: int


class LensRequest(BaseModel):
    """``POST /api/lens`` — top-k lens readouts (+ optional tracked tokens)."""

    session_id: str
    layers: list[int] | None = None
    targets: list[LensTarget] = Field(min_length=1)
    topk: int = Field(default=8, ge=1, le=MAX_NEW_TOKENS_CAP)
    track: list[str] | None = None


class AttributionRequest(BaseModel):
    """``POST /api/attribution`` — per-patch attribution over the 24x24 grid."""

    session_id: str
    layer: int
    token: str
    metric: AttributionMetric = "lens_prob"
    rollout: bool = False


class KnockoutRequest(BaseModel):
    """``POST /api/knockout`` — patch ablation through the projector."""

    session_id: str
    patches: list[int] = Field(min_length=1)
    mode: KnockoutMode = "zero"
    target_tokens: list[str] | None = None
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS_CAP)


class SteerRequest(BaseModel):
    """``POST /api/steer`` — residual-stream edit mapped onto ``ResidualEdit``."""

    session_id: str
    layer: int
    mode: SteerMode
    token: str | None = None
    source_token: str | None = None
    alpha: float
    positions: SteerPositions = "last"
    max_new_tokens: int | None = Field(default=None, ge=1, le=MAX_NEW_TOKENS_CAP)


__all__ = [
    "MAX_NEW_TOKENS_CAP",
    "AttributionRequest",
    "GenerateRequest",
    "KnockoutRequest",
    "LensRequest",
    "LensTarget",
    "SessionRequest",
    "SteerRequest",
]
