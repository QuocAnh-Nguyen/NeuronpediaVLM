#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write structurally valid but numerically random J-Lens files.

The tiny demo backend auto-fits its own lens on synthetic data, so a mock lens is
only needed to exercise the loading path or to bring up the 7B UI before a real
fit lands. Everything this script writes is random noise: ``provenance.json``
carries ``extra.mock = true`` plus a loud note, ``/api/meta`` reports
``lens.mock = true``, and no rank/probability from it means anything.

    python scripts/make_mock_lens.py --preset tiny --out /tmp/mock-lens
    VLMJ_LENS_DIR=/tmp/mock-lens VLMJ_BACKEND=tiny python -m vlmj.app

The ``llava-7b`` preset writes ``d_model=4096`` matrices (3 masks x 31 layers,
~1 GiB per mask in fp16), i.e. the same order of magnitude as a real lens set.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

BACKEND_ROOT = Path(__file__).resolve().parents[1]

#: Geometry a mock set must satisfy for each backend (the engine validates
#: ``d_model`` and ``source_layers`` against the loaded model).
PRESETS: dict[str, dict[str, int]] = {
    "tiny": {"d_model": 16, "n_layers": 3, "image_token_id": 50},
    "llava-7b": {"d_model": 4096, "n_layers": 32, "image_token_id": 32000},
}

MASKS = ("text", "image", "all")


def _bootstrap() -> None:
    """Make ``vlmj`` (and through it ``vlm_lens``/``jlens``) importable."""
    if str(BACKEND_ROOT) not in sys.path:
        sys.path.insert(0, str(BACKEND_ROOT))
    import vlmj

    vlmj.ensure_vlm_lens()


def build_lenses(preset: str, *, seed: int) -> dict[str, Any]:
    """One random lens per mask, with a per-layer scale decay (mock only)."""
    import torch
    from jlens.lens import JacobianLens

    geometry = PRESETS[preset]
    layers = list(range(int(geometry["n_layers"]) - 1))
    lenses: dict[str, Any] = {}
    for offset, mask in enumerate(MASKS):
        generator = torch.Generator().manual_seed(seed + offset)
        jacobians = {}
        for layer in layers:
            scale = 1.0 / (1.0 + 0.25 * layer)
            jacobians[layer] = (
                torch.randn(int(geometry["d_model"]), int(geometry["d_model"]), generator=generator) * scale
            )
        lenses[mask] = JacobianLens(jacobians=jacobians, n_prompts=0, d_model=int(geometry["d_model"]))
    return lenses


def write_mock_lens(preset: str, out_dir: Path, *, seed: int = 0, dtype_name: str = "float16") -> dict[str, Path]:
    """Write ``lens-{mask}.pt`` + ``provenance.json`` and return the paths."""
    import torch
    from vlm_lens.artifacts import build_provenance, save_lens_set

    dtype = {"float16": torch.float16, "float32": torch.float32}[dtype_name]
    geometry = PRESETS[preset]
    lenses = build_lenses(preset, seed=seed)
    mock_model = SimpleNamespace(
        d_model=int(geometry["d_model"]),
        n_layers=int(geometry["n_layers"]),
        image_token_id=int(geometry["image_token_id"]),
    )
    provenance = build_provenance(
        model=mock_model,
        tasks=["vlmj-mock"],
        masks=sorted(lenses),
        n_prompts=dict.fromkeys(lenses, 0),
        fit_config={"mock": True, "preset": preset, "seed": seed, "dtype": dtype_name},
        corpus={"name": "mock", "n_samples": 0},
        notes=(
            "MOCK LENS: random matrices written by scripts/make_mock_lens.py. "
            "d_model/source_layers are structurally valid so the demo can load it, but every "
            "entry is noise and n_prompts is 0. Do not interpret ranks, probabilities or "
            "attributions from this set."
        ),
        extra={
            "mock": True,
            "generated_by": "scripts/make_mock_lens.py",
            "preset": preset,
            "seed": seed,
        },
    )
    return save_lens_set(Path(out_dir), lenses, provenance=provenance, dtype=dtype)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--preset", choices=sorted(PRESETS), default="tiny")
    parser.add_argument("--out", type=Path, required=True, help="directory to write into")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("float16", "float32"), default="float16")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _bootstrap()
    written = write_mock_lens(args.preset, args.out, seed=int(args.seed), dtype_name=str(args.dtype))
    print(f"wrote mock lens ({args.preset}) to {args.out} — random noise, mock=true")
    for mask, path in sorted(written.items()):
        print(f"  lens-{mask}.pt  {path.stat().st_size / (1024 * 1024):.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
