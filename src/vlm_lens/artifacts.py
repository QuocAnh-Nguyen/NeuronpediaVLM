# SPDX-License-Identifier: Apache-2.0
"""Lens artifacts: upstream-compatible lens files, affine-correction bias files, provenance.

The lens file format is exactly the reference's (``J`` / ``n_prompts`` / ``source_layers`` /
``d_model``), so a file written here loads with ``jlens.JacobianLens.load`` and with
TransformerLens's ``JacobianLens.load``, and vice versa. Provenance is written as a
sidecar ``.json`` next to each lens (and optionally embedded under an extra ``provenance``
key, which upstream ``load`` ignores). The bias family (``bias-<mask>.pt``, written by the
moment census) holds the per-layer affine correction ``readout.lens_readout`` applies -
``unembed(s_l * (J_l @ h + b_l))`` - as ``{"bias": Tensor[d_model], "scale": float}`` per
layer string plus a ``"meta"`` key; entries may additionally carry ``"temp": float`` and
``"logit_bias": Tensor[vocab]`` (applied after ``unembed`` as ``z / temp + logit_bias``),
loadable with ``weights_only=True``.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from jlens.lens import JacobianLens

from vlm_lens._vendor import VENDOR_DIR
from vlm_lens.data.manifest import sha256_file

PROVENANCE_VERSION = 1
MASK_FILENAME = "lens-{mask}.pt"
PROVENANCE_FILENAME = "provenance.json"

def _plainify(value: Any) -> Any:
    """Recursively coerce to types the ``weights_only`` unpickler accepts.

    ``torch.__version__`` is a ``TorchVersion`` (a ``str`` subclass), which
    ``torch.load(..., weights_only=True)`` rejects — and that loader is exactly what
    upstream tools use. Embedded provenance must stay plain, so string subclasses are
    copied to builtin ``str`` and anything exotic is stringified.
    """
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _plainify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_plainify(item) for item in value]
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return item()
        except Exception:  # pragma: no cover - exotic scalar
            return str(value)
    return str(value)


def environment_info() -> dict[str, Any]:
    """Versions and vendored-reference identity, for artifact provenance."""
    info: dict[str, Any] = {
        "python": str(sys.version.split()[0]),
        "platform": str(platform.platform()),
        "torch": str(torch.__version__),
    }
    try:  # pragma: no cover - transformers is always present in practice
        import transformers

        info["transformers"] = str(transformers.__version__)
    except Exception:  # pragma: no cover
        info["transformers"] = None
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=VENDOR_DIR,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover - vendored tree may not be a git checkout
        commit = None
    info["jlens_vendored_dir"] = str(VENDOR_DIR)
    info["jlens_vendored_commit"] = commit
    try:
        info["vlm_lens_commit"] = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parents[2],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # pragma: no cover
        info["vlm_lens_commit"] = None
    return info


def build_provenance(
    *,
    model: Any,
    tasks: Sequence[str],
    masks: Sequence[str],
    n_prompts: Mapping[str, int],
    fit_config: Mapping[str, Any],
    corpus: Mapping[str, Any] | None = None,
    manifest_path: str | Path | None = None,
    notes: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Everything needed to reproduce and interpret a lens set."""
    from vlm_lens.fitting import model_fingerprint

    provenance: dict[str, Any] = {
        "provenance_version": PROVENANCE_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "estimator": "reference jlens: sum over later target positions, mean over source positions",
        "tasks": list(tasks),
        "masks": list(masks),
        "model": model_fingerprint(model),
        "fit_config": dict(fit_config),
        "n_prompts": {mask: int(n) for mask, n in n_prompts.items()},
        "corpus": dict(corpus or {}),
        "environment": environment_info(),
    }
    if manifest_path is not None:
        manifest_path = Path(manifest_path)
        provenance["manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path) if manifest_path.exists() else None,
        }
    if notes:
        provenance["notes"] = notes
    if extra:
        provenance["extra"] = dict(extra)
    return provenance


def save_lens_set(
    out_dir: str | Path,
    lenses: Mapping[str, JacobianLens],
    *,
    provenance: Mapping[str, Any],
    dtype: torch.dtype = torch.float16,
    embed_provenance: bool = False,
) -> dict[str, Path]:
    """Write ``lens-{mask}.pt`` per mask plus ``provenance.json``.

    Returns ``{mask: path}``. Each lens file stays exactly the upstream schema unless
    ``embed_provenance`` is set, which only adds an ignored key.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}
    files_meta: dict[str, Any] = {}

    for mask, lens in lenses.items():
        path = out_dir / MASK_FILENAME.format(mask=mask)
        payload: dict[str, Any] = {
            "J": {layer: J.to(dtype) for layer, J in lens.jacobians.items()},
            "n_prompts": int(lens.n_prompts),
            "source_layers": list(lens.source_layers),
            "d_model": int(lens.d_model),
        }
        if embed_provenance:
            payload["provenance"] = _plainify(dict(provenance))
        torch.save(payload, path)
        written[mask] = path
        files_meta[mask] = {
            "file": path.name,
            "dtype": str(dtype),
            "n_prompts": int(lens.n_prompts),
            "source_layers": list(lens.source_layers),
            "sha256": sha256_file(path),
        }

    provenance_out = {**dict(provenance), "artifacts": files_meta}
    (out_dir / PROVENANCE_FILENAME).write_text(
        json.dumps(provenance_out, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return written


def load_lens_set(
    path: str | Path,
) -> tuple[dict[str, JacobianLens], dict[str, Any]]:
    """Load a lens directory (or single ``lens-*.pt`` file) plus its provenance.

    Returns ``({mask: JacobianLens}, provenance)``; provenance is ``{}`` when no sidecar
    exists (e.g. an upstream release directory).
    """
    path = Path(path)
    if path.is_file():
        mask = path.stem.replace("lens-", "")
        return {mask: JacobianLens.load(path)}, {}
    lenses: dict[str, JacobianLens] = {}
    for lens_path in sorted(path.glob("lens-*.pt")):
        mask = lens_path.stem.replace("lens-", "")
        lenses[mask] = JacobianLens.load(lens_path)
    if not lenses:
        raise FileNotFoundError(f"no lens-*.pt files under {path}")
    provenance_path = path / PROVENANCE_FILENAME
    provenance = (
        json.loads(provenance_path.read_text(encoding="utf-8"))
        if provenance_path.exists()
        else {}
    )
    return lenses, provenance


def merge_shards(shard_dirs: Sequence[str | Path], *, masks: Sequence[str] | None = None) -> dict[str, JacobianLens]:
    """Merge per-shard lens sets with the reference's ``n_prompts``-weighted mean."""
    per_mask: dict[str, list[JacobianLens]] = {}
    for shard in shard_dirs:
        lenses, _ = load_lens_set(shard)
        for mask, lens in lenses.items():
            if masks is not None and mask not in masks:
                continue
            per_mask.setdefault(mask, []).append(lens)
    if not per_mask:
        raise FileNotFoundError(f"no lenses found under {list(shard_dirs)}")
    return {mask: JacobianLens.merge(group) for mask, group in per_mask.items()}


def save_bias(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write an affine-correction bias file (the ``bias-<mask>.pt`` family).

    ``payload`` maps each layer string to ``{"bias": Tensor[d_model], "scale": float}``
    plus a ``"meta"`` key; entries may additionally carry ``"temp": float`` and
    ``"logit_bias": Tensor[vocab]`` (both round-trip through the same path). Tensors are
    stored verbatim (``weights_only=True`` accepts them); every other non-primitive goes
    through :func:`_plainify`, so exotic types cannot break the load the way a
    ``TorchVersion`` would break the lens provenance.
    """
    path = Path(path)
    entries: dict[str, Any] = {}
    for layer_str, entry in payload.items():
        if layer_str == "meta":
            continue
        if not isinstance(entry, Mapping) or "bias" not in entry or "scale" not in entry:
            raise ValueError(
                f"bias entry for layer {layer_str!r} must be a mapping with "
                f"'bias' and 'scale' keys, got {type(entry).__name__}"
            )
        if not isinstance(entry["bias"], torch.Tensor):
            raise ValueError(
                f"bias for layer {layer_str!r} must be a torch.Tensor, "
                f"got {type(entry['bias']).__name__}"
            )
        entries[str(layer_str)] = {
            key: (value if isinstance(value, torch.Tensor) else _plainify(value))
            for key, value in entry.items()
        }
    out = {**entries, "meta": _plainify(dict(payload.get("meta") or {}))}
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out, path)
    return path


def load_bias(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a bias file written by :func:`save_bias`.

    Returns ``(payload, meta)``: ``payload`` maps each layer string to its
    ``{"bias": Tensor[d_model], "scale": float}`` entry (``meta`` excluded); entries may
    carry the optional ``"temp"``/``"logit_bias"`` keys, and
    ``meta`` is the provenance dict. Raises ``FileNotFoundError`` on a missing path.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"no bias file at {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    meta = checkpoint.pop("meta", {})
    return checkpoint, meta


__all__ = [
    "MASK_FILENAME",
    "PROVENANCE_FILENAME",
    "PROVENANCE_VERSION",
    "build_provenance",
    "environment_info",
    "load_bias",
    "load_lens_set",
    "merge_shards",
    "save_bias",
    "save_lens_set",
]
