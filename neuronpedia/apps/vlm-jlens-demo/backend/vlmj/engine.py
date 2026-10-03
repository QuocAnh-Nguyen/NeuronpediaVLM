# SPDX-License-Identifier: Apache-2.0
"""Model, lens, sessions and the work behind every job.

The engine owns one ``LlavaLensModel`` and at most one lens set, and serializes
all model work behind a single re-entrant lock. Job functions (:func:`job_generate`,
:func:`job_lens`, :func:`job_attribution`, :func:`job_knockout`, :func:`job_steer`)
run inside :meth:`Engine.execute`, which also enters ``torch.inference_mode()``
and flushes the CUDA allocator afterwards.

Demo-only helpers that the research library does not provide live here: the
image-feature knockout hook on the multimodal projector, scored greedy
generation (per-token logprobs with ``ResidualEditor`` active), attention
rollout over the patch grid, the per-patch rank and knockout-outcome readouts,
the per-edit steering diagnostics (vector/residual norms and the swap-basis
condition number), the text-only twin session, and the session/patch geometry
around ``build_position_masks``.
"""

from __future__ import annotations

import base64
import contextlib
import dataclasses
import io
import json
import logging
import math
import os
import random
import threading
import time
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
from PIL import Image, ImageDraw
from transformers import LogitsProcessor
from vlm_lens.artifacts import build_provenance, load_lens_set, save_lens_set
from vlm_lens.data.dummy import build_dummy_manifest
from vlm_lens.data.manifest import read_manifest
from vlm_lens.fitting import fit_masked
from vlm_lens.interventions import ResidualEdit, ResidualEditor, lens_vectors
from vlm_lens.models.llava import IMAGE_PLACEHOLDER, LlavaLensModel, MultimodalBatch
from vlm_lens.models.tiny_llava import TinyLlavaConfig, build_tiny_llava
from vlm_lens.positions import build_position_masks
from vlm_lens.readout import lens_readout

if TYPE_CHECKING:
    from jlens.lens import JacobianLens

logger = logging.getLogger(__name__)

#: LLaVA-1.5 chat template; sessions default to this prompt.
DEFAULT_PROMPT = "USER: <image>\nDescribe this image.\nASSISTANT:"
DEFAULT_MAX_NEW_TOKENS = 24
MAX_NEW_TOKENS_CAP = 64
MAX_SESSIONS = 64
BACKENDS = ("tiny", "hf-llava")
DEFAULT_HF_MODEL = "llava-hf/llava-1.5-7b-hf"
TINY_MODEL_NAME = "tiny-llava-demo"
#: Mask preference: the text lens is the captioning-analysis default.
LENS_MASK_PREFERENCE = ("text", "all", "image")
ATTRIBUTION_METRICS = ("lens_prob", "lens_logit", "attn_rollout")
STEER_MODES = ("add", "ablate", "swap")
#: Session variants: ``no_image`` is the text-only twin (no patches, no image).
SESSION_VARIANTS = ("image", "no_image")
#: Knockout outcome heuristic (V4); the thresholds are documented in the README.
KNOCKOUT_REMOVED_DELTA = -1.0
KNOCKOUT_PERSISTED_ABS_DELTA = 0.2
KNOCKOUT_CLASSIFICATION_NOTE = (
    "heuristic: 'removed' if delta <= -1.0 and the token is absent from caption_after; "
    "'persisted' if |delta| < 0.2 and the token is present in caption_after; else 'changed'"
)
#: Swap-basis condition number above which the edit is flagged unstable (X7).
STEER_COND_WARN = 1e3

#: Tiny demo geometry: real 336/14 patch grid (576 image tokens), miniature LM.
TINY_DEMO_CONFIG: dict[str, Any] = {
    "d_model": 16,
    "n_layers": 3,
    "n_heads": 2,
    "vision_hidden": 16,
    "vision_layers": 1,
    "image_size": 336,
    "patch_size": 14,
    "vocab_size": 64,
    "image_token_id": 50,
    "seed": 0,
}

_DTYPE_ALIASES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "fp32": torch.float32,
    "float16": torch.float16,
    "fp16": torch.float16,
    "bfloat16": torch.bfloat16,
    "bf16": torch.bfloat16,
}

_SAMPLE_LABELS = (
    "Synthetic pattern #1 (procedural blocks)",
    "Synthetic pattern #2 (procedural blocks)",
    "Synthetic pattern #3 (procedural blocks)",
)


class ApiError(Exception):
    """HTTP-shaped error raised by engine code and translated by the app."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(detail)
        self.status_code = int(status_code)
        self.detail = str(detail)


# --------------------------------------------------------------------------- config


@dataclass
class Config:
    """Server configuration (``Config.from_env()`` mirrors the environment)."""

    backend: str = "tiny"
    model_id: str = DEFAULT_HF_MODEL
    lens_dir: Path | None = None
    device: str | None = None
    dtype: str | None = None
    autofit: bool = True
    cache_dir: Path | None = None

    def __post_init__(self) -> None:
        if self.backend not in BACKENDS:
            raise ValueError(f"VLMJ_BACKEND must be one of {BACKENDS}, got {self.backend!r}")
        if self.lens_dir is not None:
            self.lens_dir = Path(self.lens_dir)
        if self.cache_dir is None:
            self.cache_dir = Path(__file__).resolve().parent / ".cache"
        else:
            self.cache_dir = Path(self.cache_dir)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> Config:
        """Build from ``VLMJ_*`` environment variables (documented in the README)."""
        source: dict[str, str] = dict(env if env is not None else {})

        def get(name: str) -> str | None:
            return source[name] if name in source else os.environ.get(name)

        autofit = str(get("VLMJ_AUTOFIT") or "1").strip().lower() not in {"0", "false", "no", "off"}
        lens_dir = get("VLMJ_LENS_DIR")
        cache = get("VLMJ_CACHE")
        model = get("VLMJ_MODEL")
        return cls(
            backend=str(get("VLMJ_BACKEND") or "tiny").strip().lower(),
            model_id=str(model or DEFAULT_HF_MODEL),
            lens_dir=Path(lens_dir) if lens_dir else None,
            device=get("VLMJ_DEVICE") or None,
            dtype=get("VLMJ_DTYPE") or None,
            autofit=autofit,
            cache_dir=Path(cache) if cache else None,
        )

    @property
    def cache_path(self) -> Path:
        """``cache_dir`` as resolved by ``__post_init__`` (never ``None`` afterwards)."""
        cache = self.cache_dir
        if cache is None:  # pragma: no cover - __post_init__ always assigns one
            raise RuntimeError("cache_dir is unset; build Config through its constructor")
        return cache


def resolve_device(backend: str, requested: str | None) -> torch.device:
    """Requested device, else ``cuda`` for the 7B path and ``cpu`` for tiny."""
    if requested:
        return torch.device(requested)
    if backend == "hf-llava" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def resolve_dtype(backend: str, requested: str | None) -> torch.dtype:
    """Requested dtype, else fp32 for tiny and bf16 for the 7B path."""
    if requested:
        key = str(requested).strip().lower()
        if key not in _DTYPE_ALIASES:
            raise ValueError(f"unknown dtype {requested!r}; expected one of {sorted(_DTYPE_ALIASES)}")
        return _DTYPE_ALIASES[key]
    return torch.float32 if backend == "tiny" else torch.bfloat16


def dtype_name(dtype: torch.dtype) -> str:
    return str(dtype).replace("torch.", "")


def gpu_info() -> dict[str, int] | None:
    """``{free_mb, total_mb}`` when CUDA is visible, else ``None``."""
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {"free_mb": int(free_bytes // (1024 * 1024)), "total_mb": int(total_bytes // (1024 * 1024))}


def grid_for(image_seq_length: int) -> list[int]:
    """Square patch grid for an image token count (``576 -> [24, 24]``)."""
    if image_seq_length <= 0:
        return [0, 0]
    side = math.isqrt(int(image_seq_length))
    if side * side == int(image_seq_length):
        return [side, side]
    return [1, int(image_seq_length)]


def capabilities_payload() -> dict[str, Any]:
    """Backend capabilities (identical when the model fails to load)."""
    return {
        "lens_readout": True,
        "attribution_metrics": list(ATTRIBUTION_METRICS),
        "knockout": True,
        "steer_modes": list(STEER_MODES),
        "max_new_tokens": MAX_NEW_TOKENS_CAP,
    }


# --------------------------------------------------------------------------- images


def decode_image_b64(value: str) -> Image.Image:
    """Decode a base64 PNG/JPEG (``data:`` URL prefixes accepted) to RGB."""
    payload = value.split(",", 1)[1] if value.startswith("data:") and "," in value else value
    try:
        raw = base64.b64decode(payload, validate=True)
    except Exception as exc:  # noqa: BLE001 - any decode failure is a client error
        raise ApiError(422, f"invalid base64 image: {exc}") from exc
    try:
        return Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ApiError(422, f"could not decode image: {exc}") from exc


def encode_image_b64(image: Image.Image) -> str:
    """PNG ``data:`` URL for a PIL image."""
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


def procedural_image(seed: int, *, size: int = 336) -> Image.Image:
    """Deterministic, clearly synthetic image (gradient + seeded shapes + label)."""
    rng = random.Random(seed)
    image = Image.new("RGB", (size, size))
    draw = ImageDraw.Draw(image)
    for y in range(0, size, 8):
        fraction = y / size
        draw.rectangle(
            [0, y, size, y + 8],
            fill=(int(255 * fraction), (200 + seed * 20) % 256, int(255 * (1.0 - fraction))),
        )
    for _ in range(8):
        x0 = rng.randrange(0, max(1, size - 60))
        y0 = rng.randrange(0, max(1, size - 60))
        x1 = min(size - 1, x0 + rng.randrange(24, 140))
        y1 = min(size - 1, y0 + rng.randrange(24, 140))
        draw.rectangle(
            [x0, y0, x1, y1],
            outline=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
            width=6,
        )
    for _ in range(4):
        cx, cy = rng.randrange(size), rng.randrange(size)
        radius = rng.randrange(12, 46)
        draw.ellipse(
            [cx - radius, cy - radius, cx + radius, cy + radius],
            outline=(rng.randrange(256), rng.randrange(256), rng.randrange(256)),
            width=5,
        )
    draw.rectangle([0, 0, size - 1, 28], fill=(0, 0, 0))
    draw.text((6, 8), f"SYNTHETIC {seed + 1}", fill=(255, 255, 255))
    return image


def sample_images(size: int = 336) -> list[dict[str, str]]:
    """The three canned ``GET /api/samples`` entries (deterministic, synthetic)."""
    return [
        {
            "id": f"sample-{index}",
            "label": _SAMPLE_LABELS[index % len(_SAMPLE_LABELS)],
            "image_b64": encode_image_b64(procedural_image(index, size=size)),
        }
        for index in range(3)
    ]


# --------------------------------------------------------------------------- progress


class _ProgressProcessor(LogitsProcessor):
    """Calls ``callback(step, total)`` before every decoding step."""

    def __init__(self, callback: Callable[[int, int], None], total: int) -> None:
        self._callback = callback
        self._total = int(total)
        self.steps = 0

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        self.steps += 1
        self._callback(self.steps, self._total)
        return scores


# --------------------------------------------------------------------------- sessions


@dataclass
class Session:
    """One encoded prompt (image optional), plus the lazily cached baseline caption."""

    session_id: str
    created_utc: str
    prompt_text: str
    batch: MultimodalBatch
    image_start: int
    image_end: int
    image_positions: list[int]
    quarter_of_patch: list[int]
    text_tokens: list[dict[str, Any]]
    variant: str = "image"  # "image" | "no_image" (the text-only twin)
    baseline: dict[str, Any] | None = field(default=None, repr=False)

    @property
    def has_baseline(self) -> bool:
        return self.baseline is not None

    @property
    def image_count(self) -> int:
        return len(self.image_positions)


def _token_payloads(
    token_ids: Sequence[int], token_strs: Sequence[str], logprobs: Sequence[float]
) -> list[dict[str, Any]]:
    """``[{i, id, str, logprob}]`` wire form for generated tokens."""
    return [
        {"i": i, "id": int(token_id), "str": str(token_str), "logprob": float(logprob)}
        for i, (token_id, token_str, logprob) in enumerate(zip(token_ids, token_strs, logprobs))
    ]


# --------------------------------------------------------------------------- engine


class Engine:
    """The process-wide model + lens, the session store and the job bodies."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.lock = threading.RLock()
        self.device = resolve_device(config.backend, config.device)
        self.dtype = resolve_dtype(config.backend, config.dtype)
        self.model_name = TINY_MODEL_NAME if config.backend == "tiny" else config.model_id
        self.sessions: OrderedDict[str, Session] = OrderedDict()
        self._sessions_lock = threading.Lock()
        self._reverse_vocab: dict[str, int] | None = None
        self.lens: JacobianLens | None = None
        self.lens_error: str | None = None
        self.lens_source: str | None = None
        self.lens_mask: str | None = None
        self.lens_masks: list[str] = []
        self.lens_n_prompts: dict[str, int] = {}
        self.lens_notes: str | None = None
        self.lens_mock = False
        self.model = self._build_model()
        self.grid = grid_for(int(self.model.image_seq_length))
        self._load_lens()
        logger.info(
            "vlmj engine ready: model=%s device=%s dtype=%s lens=%s",
            self.model_name,
            self.device,
            dtype_name(self.dtype),
            self.lens_mask or f"unavailable ({self.lens_error})",
        )

    # ------------------------------------------------------------------ model
    def _build_model(self) -> LlavaLensModel:
        if self.config.backend == "tiny":
            hf_model, processor = build_tiny_llava(TinyLlavaConfig(**TINY_DEMO_CONFIG))
            model = LlavaLensModel(hf_model, processor)
            model.hf_model.to(device=self.device, dtype=self.dtype)
            model.hf_model.eval()
            return model
        logger.info(
            "loading %s on %s (%s); the first request can take several minutes",
            self.config.model_id,
            self.device,
            dtype_name(self.dtype),
        )
        return LlavaLensModel.from_pretrained(self.config.model_id, dtype=self.dtype, device=str(self.device))

    @property
    def n_layers(self) -> int:
        return int(self.model.n_layers)

    @property
    def vocab_size(self) -> int:
        return int(self.model.unembed_weight().shape[0])

    # ------------------------------------------------------------------ lens
    def _load_lens(self) -> None:
        explicit = self.config.lens_dir
        if explicit is not None:
            try:
                mask, lens, provenance = self._load_lens_dir(Path(explicit))
            except Exception as exc:  # noqa: BLE001 - never crash at boot on bad artifacts
                self.lens_error = f"{type(exc).__name__}: {exc}"
                # Report the attempted directory so /api/meta says *where* it looked.
                self.lens_source = str(explicit)
                logger.warning("lens unavailable at %s: %s", explicit, self.lens_error)
                return
            self._adopt_lens(mask, lens, provenance, str(explicit))
            return
        if self.config.backend == "tiny" and self.config.autofit:
            cache = self.config.cache_path / "tiny-lens"
            try:
                mask, lens, provenance = self._load_lens_dir(cache)
            except Exception as exc:  # noqa: BLE001 - refit when the cache is absent/stale
                logger.info("no reusable tiny lens at %s (%s); auto-fitting now", cache, exc)
                self._fit_tiny_lens(cache)
                # Re-load from disk so the in-memory view (masks, per-mask n_prompts)
                # comes from the same artifacts a later run would read.
                mask, lens, provenance = self._load_lens_dir(cache)
            self._adopt_lens(mask, lens, provenance, str(cache))
            return
        self.lens_error = (
            "no lens configured: set VLMJ_LENS_DIR to an artifacts directory, or run the tiny "
            "backend with VLMJ_AUTOFIT=1 to fit a demo lens on synthetic data"
        )

    def _choose_mask(self, masks: Sequence[str]) -> str:
        for preferred in LENS_MASK_PREFERENCE:
            if preferred in masks:
                return preferred
        if not masks:
            raise FileNotFoundError("no lens masks available")
        return sorted(masks)[0]

    def _load_lens_dir(self, path: Path) -> tuple[str, JacobianLens, dict[str, Any]]:
        """Load one mask's lens (plus sidecar provenance) from a dir or single file."""
        provenance: dict[str, Any] = {}
        if path.is_file():
            candidates = {path.stem.replace("lens-", "", 1): path}
        else:
            candidates = {
                candidate.stem.replace("lens-", "", 1): candidate for candidate in sorted(path.glob("lens-*.pt"))
            }
            provenance_path = path / "provenance.json"
            if provenance_path.exists():
                provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        if not candidates:
            raise FileNotFoundError(f"no lens-*.pt files under {path}")
        mask = self._choose_mask(sorted(candidates))
        # Load only the chosen mask: a 7B lens is ~0.7 GiB in fp16 per mask, so a
        # directory holding text/image/all must not be materialized three times.
        # ``load_lens_set`` still does the schema work; it is simply handed one file.
        loaded, _ = load_lens_set(candidates[mask])
        lens = loaded.get(mask) or next(iter(loaded.values()))
        self._validate_lens(lens)
        return mask, lens, provenance

    def _validate_lens(self, lens: JacobianLens) -> None:
        if int(lens.d_model) != int(self.model.d_model):
            raise ValueError(f"lens d_model={int(lens.d_model)} does not match the model's {int(self.model.d_model)}")
        layers = [int(layer) for layer in lens.source_layers]
        if not layers:
            raise ValueError("lens has no source layers")
        bad = [layer for layer in layers if not 0 <= layer < self.n_layers]
        if bad:
            raise ValueError(f"lens source layers {bad} are outside [0, {self.n_layers}) for this model")

    def _adopt_lens(self, mask: str, lens: JacobianLens, provenance: dict[str, Any], source: str) -> None:
        self.lens = lens
        self.lens_error = None
        self.lens_mask = mask
        self.lens_source = source
        artifacts = provenance.get("artifacts") or {}
        masks = sorted(artifacts) if artifacts else [mask]
        self.lens_masks = [str(item) for item in masks]
        n_prompts = {str(name): int((entry or {}).get("n_prompts", 0)) for name, entry in (artifacts or {}).items()}
        self.lens_n_prompts = n_prompts or {mask: int(lens.n_prompts)}
        self.lens_notes = provenance.get("notes")
        extra = provenance.get("extra") or {}
        self.lens_mock = bool(extra.get("mock"))
        logger.info(
            "lens ready: mask=%s layers=%s n_prompts=%s mock=%s source=%s",
            mask,
            list(lens.source_layers),
            self.lens_n_prompts.get(mask),
            self.lens_mock,
            source,
        )

    def _fit_tiny_lens(self, out_dir: Path) -> None:
        """Fit and save ``text``/``image``/``all`` lenses on synthetic dummy samples."""
        manifest_dir = self.config.cache_path / "dummy"
        manifest_path = build_dummy_manifest(
            manifest_dir,
            n_samples=3,
            image_size=int(TINY_DEMO_CONFIG["image_size"]),
            seed=0,
        )
        samples = read_manifest(manifest_path)
        source_layers = list(range(max(1, self.n_layers - 1)))
        started = time.time()
        result = fit_masked(
            self.model,
            samples,
            source_layers=source_layers,
            dim_batch=8,
            skip_first=1,
            masks=("text", "image", "all"),
        )
        elapsed = time.time() - started
        provenance = build_provenance(
            model=self.model,
            tasks=["vlmj-autofit"],
            masks=sorted(result.lenses),
            n_prompts=result.n_prompts,
            fit_config=result.config,
            corpus={
                "name": "dummy-synthetic",
                "n_samples": len(samples),
                "manifest": str(manifest_path),
            },
            manifest_path=str(manifest_path),
            notes=(
                "Auto-fitted by vlmj on synthetic dummy images with a randomly initialised tiny "
                f"model ({elapsed:.1f}s). Numerically valid, not meaningful: ignore ranks/probs "
                "for research claims and set VLMJ_LENS_DIR to use a trained lens."
            ),
        )
        save_lens_set(out_dir, result.lenses, provenance=provenance, dtype=torch.float32)
        logger.info("fitted tiny demo lens at %s in %.1fs", out_dir, elapsed)

    # ------------------------------------------------------------------ info
    def lens_info(self) -> dict[str, Any]:
        """Wire shape of the ``lens`` block in ``/api/meta`` and session replies."""
        lens = self.lens
        return {
            "available": lens is not None,
            "dir": self.lens_source,
            "source_layers": [int(layer) for layer in lens.source_layers] if lens else [],
            "masks": list(self.lens_masks) if lens else [],
            "n_prompts": dict(self.lens_n_prompts) if lens else {},
            "mock": bool(self.lens_mock),
            "notes": self.lens_notes,
            "error": self.lens_error,
        }

    def model_info(self) -> dict[str, Any]:
        return {
            "name": self.model_name,
            "n_layers": self.n_layers,
            "d_model": int(self.model.d_model),
            "image_seq_length": int(self.model.image_seq_length),
            "grid": list(self.grid),
            "vocab_size": self.vocab_size,
        }

    def capabilities(self) -> dict[str, Any]:
        return capabilities_payload()

    # ------------------------------------------------------------------ validation
    def require_lens(self) -> JacobianLens:
        if self.lens is None:
            raise ApiError(409, self.lens_error or "lens is unavailable")
        return self.lens

    def readout_layers(self) -> list[int]:
        """Fitted layers plus the vanilla final layer."""
        lens = self.require_lens()
        return sorted({int(layer) for layer in lens.source_layers} | {self.n_layers - 1})

    def resolve_layers(self, requested: Sequence[int] | None) -> list[int]:
        lens = self.require_lens()
        if requested:
            layers = sorted({int(layer) for layer in requested})
        else:
            layers = [int(layer) for layer in lens.source_layers]
        if not layers:
            raise ApiError(422, "layers must not be empty")
        allowed = set(self.readout_layers())
        unknown = [layer for layer in layers if layer not in allowed]
        if unknown:
            raise ApiError(
                422,
                f"layers {unknown} are not available; fitted layers are "
                f"{sorted(int(layer) for layer in lens.source_layers)} and layer "
                f"{self.n_layers - 1} is the vanilla model layer",
            )
        return layers

    def validate_readout_layer(self, layer: int) -> int:
        layer = int(layer)
        if layer not in set(self.readout_layers()):
            raise ApiError(
                422,
                f"layer {layer} has no lens; available layers: {self.readout_layers()} "
                f"(layer {self.n_layers - 1} reads the vanilla model)",
            )
        return layer

    def validate_steer_layer(self, layer: int) -> int:
        lens = self.require_lens()
        layer = int(layer)
        if layer not in {int(item) for item in lens.source_layers}:
            raise ApiError(
                422,
                f"steering at layer {layer} needs a lens there; fitted layers: "
                f"{sorted(int(item) for item in lens.source_layers)}",
            )
        return layer

    def resolve_token(self, token: str) -> int:
        """Token string (as shown by the backwards view) to a vocabulary id."""
        if token is None or str(token) == "":
            raise ApiError(422, "token must be a non-empty string")
        key = str(token)
        table = self._reverse_vocabulary()
        if key in table:
            return table[key]
        ids = self.model.tokenizer(key, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ApiError(422, f"token {key!r} is not in the vocabulary")
        return int(ids[-1])

    def _reverse_vocabulary(self) -> dict[str, int]:
        if self._reverse_vocab is None:
            strings = self._display_tokens(range(self.vocab_size))
            table: dict[str, int] = {}
            for token_id, text in enumerate(strings):
                table.setdefault(text, token_id)
            self._reverse_vocab = table
        return self._reverse_vocab

    def _display_tokens(self, token_ids: Sequence[int]) -> list[str]:
        ids = [int(token_id) for token_id in token_ids]
        tokenizer = getattr(self.model, "tokenizer", None)
        raw: list[Any] = []
        if tokenizer is not None and hasattr(tokenizer, "convert_ids_to_tokens"):
            raw = list(tokenizer.convert_ids_to_tokens(ids))
        return [
            f"id:{token_id}" if index >= len(raw) or raw[index] is None else str(raw[index])
            for index, token_id in enumerate(ids)
        ]

    def check_position(self, value: int, limit: int, what: str) -> int:
        value = int(value)
        if not 0 <= value < int(limit):
            raise ApiError(422, f"{what} {value} is out of range [0, {int(limit)})")
        return value

    # ------------------------------------------------------------------ sessions
    def create_session(
        self,
        *,
        image_b64: str | None = None,
        image_path: str | None = None,
        prompt: str | None = None,
        variant: str = "image",
    ) -> Session:
        """Encode one prompt (plus an image) into a session (lock-guarded).

        ``variant="image"`` takes exactly one image source and a prompt with one
        ``<image>`` placeholder. ``variant="no_image"`` is the text-only twin: the
        literal placeholder is stripped from the prompt and nothing is encoded on
        the vision side, so the session has no patches (patch endpoints answer 409).
        """
        variant = str(variant or "image")
        if variant not in SESSION_VARIANTS:
            raise ApiError(422, f"variant must be one of {list(SESSION_VARIANTS)}, got {variant!r}")
        if variant == "no_image" and (image_b64 is not None or image_path is not None):
            raise ApiError(422, "variant 'no_image' takes no image; omit image_b64/image_path")
        if variant == "image" and (image_b64 is None) == (image_path is None):
            raise ApiError(422, "provide exactly one of image_b64 or image_path")
        image: Image.Image | None = None
        if variant == "image":
            if image_b64 is not None:
                image = decode_image_b64(image_b64)
            else:
                path = Path(str(image_path))
                if not path.is_file():
                    raise ApiError(422, f"image_path does not exist: {path}")
                try:
                    image = Image.open(path).convert("RGB")
                except Exception as exc:  # noqa: BLE001
                    raise ApiError(422, f"could not read image_path: {exc}") from exc
        prompt_text = str(prompt) if prompt else DEFAULT_PROMPT
        if variant == "no_image":
            prompt_text = prompt_text.replace(IMAGE_PLACEHOLDER, "")
        with self.lock, torch.inference_mode():
            try:
                batch = self.model.encode_mm(prompt_text, image)
            except ValueError as exc:
                raise ApiError(422, str(exc)) from exc
            if variant == "no_image":
                positions, quarter_of_patch = [], []
            else:
                positions, quarter_of_patch = self._image_geometry(batch)
            text_tokens = self._text_token_payloads(batch)
        session = Session(
            session_id=uuid.uuid4().hex,
            created_utc=datetime.now(UTC).isoformat(timespec="seconds"),
            prompt_text=prompt_text,
            batch=batch,
            image_start=int(positions[0]) if positions else 0,
            image_end=int(positions[-1]) + 1 if positions else 0,
            image_positions=positions,
            quarter_of_patch=quarter_of_patch,
            text_tokens=text_tokens,
            variant=variant,
        )
        with self._sessions_lock:
            self.sessions[session.session_id] = session
            while len(self.sessions) > MAX_SESSIONS:
                self.sessions.popitem(last=False)
        return session

    def _image_geometry(self, batch: MultimodalBatch) -> tuple[list[int], list[int]]:
        """Patch positions in grid order plus the quarter index of every patch."""
        image_token_id = int(self.model.image_token_id)
        expected = int(self.model.image_seq_length)
        exclude_last = True
        mask = build_position_masks(batch.input_ids, image_token_id, skip_first=1, exclude_last=True, masks=("image",))[
            "image"
        ]
        if int(mask.sum()) != expected:
            exclude_last = False
            mask = build_position_masks(
                batch.input_ids, image_token_id, skip_first=1, exclude_last=False, masks=("image",)
            )["image"]
        positions = [int(position) for position in mask.nonzero().reshape(-1).tolist()]
        if len(positions) != expected:
            raise ApiError(
                422,
                f"expected {expected} image tokens in the prompt, found {len(positions)}; the "
                "prompt must expand to exactly one image placeholder",
            )
        quarter_masks = build_position_masks(
            batch.input_ids,
            image_token_id,
            skip_first=1,
            exclude_last=exclude_last,
            masks=("image-q0", "image-q1", "image-q2", "image-q3"),
        )
        patch_of_position = {position: index for index, position in enumerate(positions)}
        quarter_of_patch = [0] * expected
        for quarter, name in enumerate(("image-q0", "image-q1", "image-q2", "image-q3")):
            for position in quarter_masks[name].nonzero().reshape(-1).tolist():
                patch = patch_of_position.get(int(position))
                if patch is not None:
                    quarter_of_patch[patch] = quarter
        return positions, quarter_of_patch

    def _text_token_payloads(self, batch: MultimodalBatch) -> list[dict[str, Any]]:
        """Non-image prompt tokens as ``[{i, id, str}]`` (absolute positions)."""
        image_mask = batch.image_token_mask[0]
        keep = [index for index in range(batch.seq_len) if not bool(image_mask[index])]
        ids = [int(batch.input_ids[0, index]) for index in keep]
        strings = self._display_tokens(ids)
        return [{"i": index, "id": token_id, "str": text} for index, token_id, text in zip(keep, ids, strings)]

    def get_session(self, session_id: str) -> Session:
        with self._sessions_lock:
            session = self.sessions.get(str(session_id))
        if session is None:
            raise ApiError(404, f"unknown session_id: {session_id}")
        return session

    def session_payload(self, session: Session) -> dict[str, Any]:
        """Wire shape of ``POST /api/session`` (``image`` is null for the twin)."""
        image = None
        if session.variant == "image":
            image = {
                "start": session.image_start,
                "end": session.image_end,
                "count": session.image_count,
                "grid": list(self.grid),
                "quarter_of_patch": list(session.quarter_of_patch),
            }
        return {
            "session_id": session.session_id,
            "mode": self.config.backend,
            "model": self.model_info(),
            "lens": self.lens_info(),
            "variant": session.variant,
            "prompt_text": session.prompt_text,
            "text_tokens": session.text_tokens,
            "image": image,
            "has_baseline": session.has_baseline,
        }

    def session_info(self, session: Session) -> dict[str, Any]:
        """Wire shape of ``GET /api/session/{id}``."""
        baseline = session.baseline or {}
        return {
            "session_id": session.session_id,
            "variant": session.variant,
            "has_baseline": session.has_baseline,
            "baseline_caption": baseline.get("caption"),
            "lens": self.lens_info(),
            "created_utc": session.created_utc,
        }

    def require_image_session(self, session: Session) -> None:
        """Refuse patch endpoints on the ``no_image`` twin (409: there are no patches)."""
        if session.variant != "image":
            raise ApiError(
                409,
                f"session variant {session.variant!r} has no image patches; this endpoint needs "
                "a session created with variant 'image'",
            )


    # ------------------------------------------------------------------ generation
    def generate(
        self,
        batch: MultimodalBatch,
        max_new_tokens: int,
        *,
        edits: Sequence[ResidualEdit] = (),
        hooks: Sequence[contextlib.AbstractContextManager[Any]] = (),
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Greedy caption with per-token logprobs, optionally under edits/hooks."""
        max_new_tokens = int(max_new_tokens)
        if not 1 <= max_new_tokens <= MAX_NEW_TOKENS_CAP:
            raise ApiError(422, f"max_new_tokens must be in [1, {MAX_NEW_TOKENS_CAP}]")
        prompt_len = int(batch.seq_len)
        n_edit_forwards = 0
        with contextlib.ExitStack() as stack:
            editor: ResidualEditor | None = None
            if edits:
                editor = ResidualEditor(self.model, self.require_lens(), list(edits))
                stack.enter_context(editor)
            for hook in hooks:
                if hook is not None:
                    stack.enter_context(hook)
            kwargs: dict[str, Any] = dict(batch.hf_kwargs())
            kwargs.update(
                max_new_tokens=max_new_tokens,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
                use_cache=True,
            )
            if progress is not None:
                kwargs["logits_processor"] = [_ProgressProcessor(progress, max_new_tokens)]
            with torch.inference_mode():
                output = self.model.hf_model.generate(**kwargs)
            if editor is not None:
                n_edit_forwards = int(getattr(editor, "forward_calls", 0))
        new_tokens = output.sequences[0, prompt_len:]
        token_ids = [int(token) for token in new_tokens.tolist()]
        logprobs: list[float] = []
        for step, token_id in enumerate(token_ids):
            if step < len(output.scores):
                row = output.scores[step][0].float()
                logprobs.append(float(torch.log_softmax(row, dim=-1)[token_id]))
            else:
                logprobs.append(float("nan"))
        return {
            "caption": str(self.model._decode(new_tokens)),
            "token_ids": token_ids,
            "token_strs": self._display_tokens(token_ids),
            "logprobs": logprobs,
            "prompt_len": prompt_len,
            "n_edit_forwards": n_edit_forwards,
        }

    def ensure_baseline(self, session: Session, max_new_tokens: int | None = None) -> dict[str, Any]:
        """Lazily generate (and cache) the unedited caption for a session."""
        if session.baseline is None:
            length = int(max_new_tokens or DEFAULT_MAX_NEW_TOKENS)
            session.baseline = self.generate(session.batch, length)
        return session.baseline

    def teacher_forced_logprobs(
        self,
        batch: MultimodalBatch,
        prompt_len: int,
        token_ids: Sequence[int],
        *,
        hook: contextlib.AbstractContextManager[Any] | None = None,
    ) -> list[float]:
        """Logprob the model assigns to each token when teacher-forced."""
        with contextlib.ExitStack() as stack:
            if hook is not None:
                stack.enter_context(hook)
            with torch.inference_mode():
                logits = self.model.hf_model(**batch.hf_kwargs(), use_cache=False).logits
            rows = logits[0, int(prompt_len) - 1 :, :].float()
            del logits
            log_probs = torch.log_softmax(rows, dim=-1)
            return [float(log_probs[step, int(token_id)]) for step, token_id in enumerate(token_ids)]

    @contextlib.contextmanager
    def projector_patch(self, patches: Sequence[int], mode: str = "zero") -> Iterator[None]:
        """Zero (or replace with the row mean) the projector rows of ``patches``."""
        rows = sorted({int(patch) for patch in patches})
        if not rows:
            raise ApiError(422, "patches must not be empty")
        if any(row < 0 for row in rows):
            raise ApiError(422, f"patch indices must be >= 0, got {rows}")
        if mode not in ("zero", "mean"):
            raise ApiError(422, f"unknown knockout mode {mode!r}")
        projector = self.model.hf_model.model.multi_modal_projector

        def hook(_module: Any, args: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = args[0]
            patched = hidden.clone()
            selected = patched[..., rows, :]
            if mode == "mean":
                patched[..., rows, :] = selected.mean(dim=-2, keepdim=True).expand_as(selected)
            else:
                patched[..., rows, :] = 0
            return (patched, *args[1:])

        handle = projector.register_forward_pre_hook(hook)
        try:
            yield
        finally:
            handle.remove()

    def _empty_cache(self) -> None:
        if self.device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def execute(self, ctx: Any, fn: Callable[[Any, Engine], Any]) -> Any:
        """Run a job body under the global model lock and inference mode."""
        with self.lock, torch.inference_mode():
            try:
                return fn(ctx, self)
            finally:
                self._empty_cache()

    # ------------------------------------------------------------------ readouts
    def distribution_payload(
        self,
        logits: torch.Tensor,
        vocab: Sequence[str],
        topk: int,
        tracked_ids: dict[str, int] | None,
    ) -> dict[str, Any]:
        """``{topk, tracked}`` for one distribution row (1-based ranks).

        Shared by every lens layer and by the model's own logits row
        (``model_row``); ranks are competition-style over the full vocabulary.
        """
        probs = logits.softmax(dim=-1)
        k = max(1, min(int(topk), int(probs.shape[-1])))
        values, indices = probs.topk(k)
        selected_logits = logits[indices]
        ranks = (logits.unsqueeze(0) > selected_logits.unsqueeze(1)).sum(dim=1) + 1
        topk_entries = [
            {
                "str": str(vocab[int(index)]),
                "prob": float(value),
                "logit": float(logit),
                "rank": int(rank),
            }
            for value, index, logit, rank in zip(values, indices, selected_logits, ranks)
        ]
        tracked: dict[str, dict[str, float]] = {}
        for name, token_id in (tracked_ids or {}).items():
            token_id = int(token_id)
            tracked[str(name)] = {
                "prob": float(probs[token_id]),
                "rank": int((logits > logits[token_id]).sum().item()) + 1,
                "logit": float(logits[token_id]),
            }
        return {"topk": topk_entries, "tracked": tracked}

    def layer_payload(
        self,
        readout: Any,
        layer: int,
        row: int,
        topk: int,
        tracked_ids: dict[str, int] | None,
    ) -> dict[str, Any]:
        """``{layer, topk, tracked}`` for one position of one lens layer."""
        logits = readout.lens_logits[int(layer)][int(row)]
        return {"layer": int(layer), **self.distribution_payload(logits, readout.vocab, topk, tracked_ids)}

    def model_row_payload(
        self,
        readout: Any,
        row: int,
        topk: int,
        tracked_ids: dict[str, int] | None,
    ) -> dict[str, Any]:
        """``{topk, tracked}`` from the model's own logits at one position.

        This row is the model by construction, not lens evidence: it is the
        next-token distribution at the target's position, so its top-1 token is
        what greedy decoding would emit. The UI labels it "model output".
        """
        return self.distribution_payload(readout.model_logits[int(row)], readout.vocab, topk, tracked_ids)

    def _grid_side(self, count: int) -> int:
        """Patch-grid side for ``count`` flat values (24 for the demo geometry)."""
        side = int(self.grid[0]) if int(self.grid[0]) > 0 else math.isqrt(int(count))
        if side * side != int(count):
            side = max(1, math.isqrt(int(count)))
        if side * side != int(count):
            raise ApiError(422, f"cannot reshape {count} patch values into a square grid")
        return side

    def to_grid(self, values: Sequence[float], session: Session) -> list[list[float]]:
        """Flat patch-order values to the square patch grid."""
        side = self._grid_side(len(values))
        return [[float(values[row * side + column]) for column in range(side)] for row in range(side)]

    def to_int_grid(self, values: Sequence[int]) -> list[list[int]]:
        """Flat patch-order ints to the square patch grid (``grid_rank``)."""
        side = self._grid_side(len(values))
        return [[int(values[row * side + column]) for column in range(side)] for row in range(side)]

    def quarter_means(self, grid: Sequence[Sequence[float]], quarter_of_patch: Sequence[int]) -> list[float]:
        """Mean attribution per image quarter (top-left, top-right, bottom-left, bottom-right)."""
        flat = [float(value) for row in grid for value in row]
        totals = [0.0, 0.0, 0.0, 0.0]
        counts = [0, 0, 0, 0]
        for value, quarter in zip(flat, quarter_of_patch):
            quarter = int(quarter)
            totals[quarter] += value
            counts[quarter] += 1
        return [totals[q] / counts[q] if counts[q] else 0.0 for q in range(4)]

    def attribution_values(
        self, session: Session, layer: int, token_id: int | None, metric: str, rollout: bool
    ) -> tuple[list[float], list[int] | None]:
        """Per-patch values in patch order, plus per-patch ranks for lens metrics.

        Returns ``(values, ranks)``; ``ranks`` is the 1-based rank of the requested
        token in each patch's lens distribution, and ``None`` for ``attn_rollout``
        (attention weights have no token distribution to rank).
        """
        if metric in ("lens_prob", "lens_logit"):
            readout = lens_readout(
                self.model,
                self.require_lens(),
                session.batch,
                layers=[int(layer)],
                positions=list(session.image_positions),
            )
            logits = readout.lens_logits[int(layer)]
            if token_id is None:  # only the attention metrics above may omit the token
                raise ApiError(422, f"metric {metric!r} needs a token")
            column = logits.softmax(dim=-1)[:, int(token_id)] if metric == "lens_prob" else logits[:, int(token_id)]
            target = logits[:, int(token_id)]
            ranks = (logits > target.unsqueeze(-1)).sum(dim=-1) + 1
            return [float(value) for value in column], [int(value) for value in ranks.tolist()]
        return self.attention_rollout_values(session, layer, rollout), None

    def attention_rollout_values(self, session: Session, layer: int, rollout: bool) -> list[float]:
        """Attention mass from the last prompt token to every image patch.

        With ``rollout`` the per-layer head-mean attention rows are mixed with the
        identity (``0.5 * A + 0.5 * I``, Abnar & Zuidema style) and multiplied
        across layers ``0..layer``; without it only ``layer``'s row is used. MVP1
        only supports the step-0 query (the last prompt position).
        """
        hf_model = self.model.hf_model
        previous = getattr(hf_model, "_attn_implementation", None)
        if previous is None:
            previous = getattr(getattr(hf_model, "config", None), "_attn_implementation", None)
        restore = previous or "sdpa"
        if str(previous) != "eager":
            if not hasattr(hf_model, "set_attn_implementation"):
                raise ApiError(500, "attention rollout needs an eager attention implementation")
            hf_model.set_attn_implementation("eager")
        try:
            with torch.inference_mode():
                output = hf_model(**session.batch.hf_kwargs(), use_cache=False, output_attentions=True)
            attentions = getattr(output, "attentions", None)
            if not attentions:
                raise ApiError(500, "model did not return attention matrices")
            query = int(session.batch.seq_len) - 1
            if rollout:
                scores = attentions[0][0, :, query, :].float().mean(dim=0) * 0.5
                scores[query] += 0.5
                for index in range(1, int(layer) + 1):
                    attention = attentions[index][0].float().mean(dim=0) * 0.5
                    attention.diagonal().add_(0.5)
                    scores = scores @ attention
            else:
                scores = attentions[int(layer)][0, :, query, :].float().mean(dim=0)
            del output
        finally:
            if str(previous) != "eager":
                hf_model.set_attn_implementation(restore)
        return [float(value) for value in scores[session.image_positions]]

    def baseline_residual_norm(self, batch: MultimodalBatch, layer: int) -> float | None:
        """``||residual||`` at the last position of ``batch`` at ``layer``.

        One :class:`~jlens.hooks.ActivationRecorder` forward without edits (the
        baseline ``h_norm`` of the steering diagnostics). ``None`` when the forward
        or the hook is unavailable, so a diagnostics-only failure never fails a job.
        """
        from jlens.hooks import ActivationRecorder  # noqa: PLC0415 - jlens is vendored by vlm_lens

        try:
            with ActivationRecorder(self.model.layers, at=[int(layer)]) as recorder:
                self.model.forward_mm(batch)
            hidden = recorder.activations[int(layer)]
            return float(hidden[0, -1].detach().float().norm())
        except Exception as exc:  # noqa: BLE001 - diagnostics are best-effort
            logger.warning("baseline residual norm unavailable at layer %s: %s", layer, exc)
            return None

    def edit_diagnostics(self, batch: MultimodalBatch, edit: ResidualEdit) -> dict[str, Any]:
        """One ``/api/steer`` diagnostics entry: ``{layer, mode, v_norm, h_norm, cond}``.

        ``v_norm`` is ``||v_t||`` of the *target* token (for ``swap`` that is the
        target direction, not the source); ``h_norm`` is the baseline residual norm
        at that layer's last position (one unedited forward, ``None`` if
        unavailable); ``cond`` is ``torch.linalg.cond`` of the stacked
        ``[v_s, v_t]`` basis for ``swap`` and ``None`` otherwise — and also
        ``None`` when that basis is numerically singular.
        """
        layer = int(edit.layer)
        tokens = [str(edit.token)]
        if edit.mode == "swap":
            tokens.append(str(edit.source_token))
        rows = lens_vectors(self.model, self.require_lens(), tokens, layers=[layer])[layer]
        v_norm = float(rows[0].float().norm())
        cond: float | None = None
        if edit.mode == "swap":
            basis = torch.stack([rows[1].float().reshape(-1), rows[0].float().reshape(-1)], dim=1)
            computed = float(torch.linalg.cond(basis))
            if not math.isfinite(computed):
                logger.warning("swap basis at layer %s is numerically singular; cond reported as null", layer)
            else:
                cond = computed
                if cond > STEER_COND_WARN:
                    logger.warning(
                        "swap basis at layer %s is ill-conditioned (cond=%.3g > %.0f): the "
                        "pseudo-inverse edit is unstable (X7)",
                        layer,
                        cond,
                        STEER_COND_WARN,
                    )
        return {
            "layer": layer,
            "mode": str(edit.mode),
            "v_norm": v_norm,
            "h_norm": self.baseline_residual_norm(batch, layer),
            "cond": cond,
        }

    def extend_batch(self, batch: MultimodalBatch, token_ids: Sequence[int]) -> MultimodalBatch:
        """Append generated tokens to a batch (attention on, no image mask)."""
        new_ids = torch.tensor(
            [[int(token) for token in token_ids]],
            dtype=batch.input_ids.dtype,
            device=batch.input_ids.device,
        )
        image_token_mask = torch.cat([batch.image_token_mask, torch.zeros_like(new_ids, dtype=torch.bool)], dim=1)
        return dataclasses.replace(
            batch,
            input_ids=torch.cat([batch.input_ids, new_ids], dim=1),
            attention_mask=torch.cat([batch.attention_mask, torch.ones_like(new_ids)], dim=1),
            image_token_mask=image_token_mask,
        )


# --------------------------------------------------------------------------- jobs


def _step_progress(ctx: Any, label: str, low: float, high: float) -> Callable[[int, int], None]:

    def report(step: int, total: int) -> None:
        ctx.update(f"{label} {step}/{total}", low + (high - low) * step / max(int(total), 1))

    return report


def job_generate(ctx: Any, engine: Engine, session: Session, max_new_tokens: int) -> dict[str, Any]:
    """Greedy caption; caches it as the session baseline."""
    total = int(max_new_tokens)
    ctx.update(f"generating 0/{total}", 0.05)
    caption = engine.generate(session.batch, total, progress=_step_progress(ctx, "generating", 0.05, 0.9))
    if session.baseline is None:
        session.baseline = caption
    ctx.update("saving", 0.95)
    return {
        "caption": caption["caption"],
        "tokens": _token_payloads(caption["token_ids"], caption["token_strs"], caption["logprobs"]),
        "prompt_len": int(caption["prompt_len"]),
    }


def _target_label(engine: Engine, session: Session, kind: str, index: int, baseline: dict[str, Any] | None) -> str:
    if kind == "patch":
        side = int(engine.grid[0])
        return f"patch {index} (row {index // side}, col {index % side})"
    if kind == "prompt":
        text = next((entry["str"] for entry in session.text_tokens if int(entry["i"]) == index), None)
        if text is None:
            text = engine._display_tokens([int(session.batch.input_ids[0, index])])[0]
        return f"prompt[{index}] {text!r}"
    token_str = baseline["token_strs"][index] if baseline else "?"
    return f"gen[{index}] {token_str!r}"


def job_lens(ctx: Any, engine: Engine, session: Session, request: Any) -> dict[str, Any]:
    """J-lens readouts for ``gen`` / ``prompt`` / ``patch`` targets.

    Each target carries ``per_layer`` (lens rows) plus ``model_row``, the model's
    own logits at that position: the "model output" anchor, which is model by
    construction rather than lens evidence.
    """
    lens = engine.require_lens()
    layers = engine.resolve_layers(getattr(request, "layers", None))
    targets = list(request.targets)
    topk = int(getattr(request, "topk", 8))
    tracked_ids = {str(name): engine.resolve_token(str(name)) for name in (getattr(request, "track", None) or [])}
    baseline = session.baseline
    payloads: dict[tuple[str, int], list[dict[str, Any]]] = {}
    model_rows: dict[tuple[str, int], dict[str, Any]] = {}
    fixed: dict[tuple[str, int], int] = {}
    for target in targets:
        kind = str(target.kind)
        index = int(target.i)
        if kind == "prompt":
            fixed[(kind, index)] = engine.check_position(index, session.batch.seq_len, "prompt position")
        elif kind == "patch":
            fixed[(kind, index)] = session.image_start + engine.check_position(
                index, session.image_count, "patch index"
            )
        elif kind == "gen":
            if baseline is None:
                raise ApiError(422, "no generated caption yet; run POST /api/generate first")
            engine.check_position(index, len(baseline["token_ids"]), "generated-token index")
        else:
            raise ApiError(422, f"unknown target kind {kind!r}")
    ctx.update("forward", 0.15)
    if fixed:
        positions = sorted(set(fixed.values()))
        readout = lens_readout(engine.model, lens, session.batch, layers=layers, positions=positions)
        row_of = {position: row for row, position in enumerate(positions)}
        for key, position in fixed.items():
            row = row_of[position]
            payloads[key] = [engine.layer_payload(readout, layer, row, topk, tracked_ids) for layer in layers]
            model_rows[key] = engine.model_row_payload(readout, row, topk, tracked_ids)
    gen_targets = [int(t.i) for t in targets if str(t.kind) == "gen"]
    if gen_targets:
        assert baseline is not None  # gen targets were validated before the loop
        token_ids = list(baseline["token_ids"])
        longest = max(gen_targets)
        batch = engine.extend_batch(session.batch, token_ids[: longest + 1])
        offset = int(session.batch.seq_len) - 1
        positions = [offset + index for index in gen_targets]
        readout = lens_readout(engine.model, lens, batch, layers=layers, positions=positions)
        row_of = {position: row for row, position in enumerate(positions)}
        for index in gen_targets:
            row = row_of[offset + index]
            payloads[("gen", index)] = [
                engine.layer_payload(readout, layer, row, topk, tracked_ids) for layer in layers
            ]
            model_rows[("gen", index)] = engine.model_row_payload(readout, row, topk, tracked_ids)
    ctx.update("transport", 0.55)
    ctx.update("scoring targets", 0.8)
    out_targets = []
    for target in targets:
        kind = str(target.kind)
        index = int(target.i)
        out_targets.append(
            {
                "kind": kind,
                "i": index,
                "label": _target_label(engine, session, kind, index, baseline),
                "per_layer": payloads[(kind, index)],
                "model_row": model_rows[(kind, index)],
            }
        )
    ctx.update("saving", 0.95)
    return {"targets": out_targets, "vocab_size": engine.vocab_size}


def job_attribution(ctx: Any, engine: Engine, session: Session, request: Any) -> dict[str, Any]:
    """Per-patch attribution over the 24x24 grid for one token and metric.

    ``grid_rank`` is the 1-based rank of the requested token at each patch, and
    ``None`` for ``attn_rollout`` (attention weights rank no token).
    """
    engine.require_lens()
    engine.require_image_session(session)
    layer = engine.validate_readout_layer(int(request.layer))
    metric = str(request.metric)
    if metric not in ATTRIBUTION_METRICS:
        raise ApiError(422, f"unknown metric {metric!r}; expected one of {list(ATTRIBUTION_METRICS)}")
    token_id = engine.resolve_token(str(request.token)) if metric in ("lens_prob", "lens_logit") else None
    ctx.update("forward", 0.15)
    values, rank_values = engine.attribution_values(session, layer, token_id, metric, bool(request.rollout))
    if metric in ("lens_prob", "lens_logit"):
        ctx.update("transport", 0.55)
    ctx.update("scoring patches", 0.8)
    grid = engine.to_grid(values, session)
    grid_rank = engine.to_int_grid(rank_values) if rank_values is not None else None
    quarters = engine.quarter_means(grid, session.quarter_of_patch)
    vmin = min(min(row) for row in grid)
    vmax = max(max(row) for row in grid)
    ctx.update("saving", 0.95)
    return {
        "layer": layer,
        "metric": metric,
        "rollout": bool(request.rollout),
        "grid": grid,
        "grid_rank": grid_rank,
        "quarters": quarters,
        "vmin": float(vmin),
        "vmax": float(vmax),
    }


def _token_filter(engine: Engine, request: Any) -> tuple[set[int], set[str]]:
    """``(ids, display strings)`` selected by ``target_tokens``."""
    ids: set[int] = set()
    names: set[str] = set()
    for token in getattr(request, "target_tokens", None) or []:
        try:
            ids.add(engine.resolve_token(str(token)))
        except ApiError:
            names.add(str(token))
            names.add(str(token).strip())
    return ids, names


def _selected_token(token_id: int, token_str: str, ids: set[int], names: set[str]) -> bool:
    if not ids and not names:
        return True
    return int(token_id) in ids or str(token_str) in names or str(token_str).strip() in names


def _classify_outcome(delta: float, in_caption_after: bool) -> str:
    """V4 outcome class for one baseline caption token (thresholds in ``KNOCKOUT_*``)."""
    if delta <= KNOCKOUT_REMOVED_DELTA and not in_caption_after:
        return "removed"
    if abs(delta) < KNOCKOUT_PERSISTED_ABS_DELTA and in_caption_after:
        return "persisted"
    return "changed"


def job_knockout(ctx: Any, engine: Engine, session: Session, request: Any) -> dict[str, Any]:
    """Ablate image patches through the projector and rescore the baseline caption.

    ``outcomes`` classifies each selected baseline token as ``removed`` /
    ``persisted`` / ``changed`` (heuristic documented in
    :data:`KNOCKOUT_CLASSIFICATION_NOTE`) over the same token filter as ``deltas``.
    """
    engine.require_lens()
    engine.require_image_session(session)
    patches = sorted({int(patch) for patch in request.patches})
    for patch in patches:
        engine.check_position(patch, session.image_count, "patch index")
    mode = str(request.mode)
    cached = session.baseline
    length = int(
        request.max_new_tokens
        if request.max_new_tokens is not None
        else (len(cached["token_ids"]) if cached else DEFAULT_MAX_NEW_TOKENS)
    )
    ctx.update("scoring baseline", 0.1)
    baseline = engine.ensure_baseline(session, length)
    caption_ids = list(baseline["token_ids"])
    prompt_len = int(session.batch.seq_len)
    forced = engine.extend_batch(session.batch, caption_ids)
    before = engine.teacher_forced_logprobs(forced, prompt_len, caption_ids)
    ctx.update("knockout forward", 0.3)
    after = engine.teacher_forced_logprobs(forced, prompt_len, caption_ids, hook=engine.projector_patch(patches, mode))
    ctx.update(f"generating 0/{length}", 0.5)
    regenerated = engine.generate(
        session.batch,
        length,
        hooks=[engine.projector_patch(patches, mode)],
        progress=_step_progress(ctx, "generating", 0.5, 0.9),
    )
    selected_ids, selected_names = _token_filter(engine, request)
    caption_after_tokens = {str(token_str) for token_str in regenerated["token_strs"]}
    deltas = []
    outcomes = []
    for step, token_id in enumerate(caption_ids):
        token_str = str(baseline["token_strs"][step])
        if not _selected_token(token_id, token_str, selected_ids, selected_names):
            continue
        delta = float(after[step] - before[step])
        deltas.append(
            {
                "token": token_str,
                "logprob_before": float(before[step]),
                "logprob_after": float(after[step]),
                "delta": delta,
            }
        )
        in_caption_after = token_str in caption_after_tokens
        outcomes.append(
            {
                "token": token_str,
                "class": _classify_outcome(delta, in_caption_after),
                "delta": delta,
                "in_caption_after": in_caption_after,
            }
        )
    ctx.update("saving", 0.95)
    return {
        "patches": patches,
        "mode": mode,
        "baseline_caption": baseline["caption"],
        "caption_after": regenerated["caption"],
        "tokens_after": _token_payloads(regenerated["token_ids"], regenerated["token_strs"], regenerated["logprobs"]),
        "deltas": deltas,
        "outcomes": outcomes,
        "classification_note": KNOCKOUT_CLASSIFICATION_NOTE,
    }


def _edit_payload(edit: ResidualEdit) -> dict[str, Any]:
    return {
        "layer": int(edit.layer),
        "mode": str(edit.mode),
        "token": edit.token,
        "source_token": edit.source_token,
        "alpha": float(edit.alpha),
        "positions": str(edit.positions),
    }


def job_steer(ctx: Any, engine: Engine, session: Session, request: Any) -> dict[str, Any]:
    """Apply one residual-stream edit, regenerate the caption, and report diagnostics."""
    engine.require_lens()
    layer = engine.validate_steer_layer(int(request.layer))
    mode = str(request.mode)
    if mode not in STEER_MODES:
        raise ApiError(422, f"unknown steer mode {mode!r}; expected one of {list(STEER_MODES)}")
    token = request.token
    source_token = request.source_token
    if mode in ("add", "ablate") and not token:
        raise ApiError(422, f"steering mode {mode!r} needs a target token")
    if mode == "swap" and (not token or not source_token):
        raise ApiError(422, "steering mode 'swap' needs both token and source_token")
    try:
        edit = ResidualEdit(
            layer=layer,
            mode=mode,
            alpha=float(request.alpha),
            token=str(token) if token else None,
            source_token=str(source_token) if source_token else None,
            positions=str(request.positions),
        )
    except ValueError as exc:
        raise ApiError(422, str(exc)) from exc
    cached = session.baseline
    length = int(
        request.max_new_tokens
        if request.max_new_tokens is not None
        else (len(cached["token_ids"]) if cached else DEFAULT_MAX_NEW_TOKENS)
    )
    ctx.update("baseline", 0.05)
    baseline = engine.ensure_baseline(session, length)
    ctx.update("diagnostics", 0.15)
    diagnostics = [engine.edit_diagnostics(session.batch, edit)]
    ctx.update(f"generating 0/{length}", 0.2)
    caption = engine.generate(
        session.batch,
        length,
        edits=[edit],
        progress=_step_progress(ctx, "generating", 0.2, 0.9),
    )
    ctx.update("saving", 0.95)
    return {
        "edits": [_edit_payload(edit)],
        "caption_before": baseline["caption"],
        "caption_after": caption["caption"],
        "tokens_before": _token_payloads(baseline["token_ids"], baseline["token_strs"], baseline["logprobs"]),
        "tokens_after": _token_payloads(caption["token_ids"], caption["token_strs"], caption["logprobs"]),
        "n_edit_forwards": int(caption["n_edit_forwards"]),
        "diagnostics": diagnostics,
    }


__all__ = [
    "ATTRIBUTION_METRICS",
    "ApiError",
    "Config",
    "DEFAULT_HF_MODEL",
    "DEFAULT_MAX_NEW_TOKENS",
    "DEFAULT_PROMPT",
    "Engine",
    "MAX_NEW_TOKENS_CAP",
    "MAX_SESSIONS",
    "STEER_MODES",
    "Session",
    "TINY_DEMO_CONFIG",
    "TINY_MODEL_NAME",
    "capabilities_payload",
    "decode_image_b64",
    "dtype_name",
    "encode_image_b64",
    "gpu_info",
    "grid_for",
    "job_attribution",
    "job_generate",
    "job_knockout",
    "job_lens",
    "job_steer",
    "procedural_image",
    "resolve_device",
    "resolve_dtype",
    "sample_images",
]
