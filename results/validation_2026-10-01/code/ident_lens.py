#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
"""Write an identity-matrix lens dir for the bias-only decomposition (R3 P5).

The calibrated logit lens (z = unembed(s_l*(h + b_l))) is the identity composed with the
moment-census bias+scale; scoring it against the J-lens with the SAME payload decomposes the
mid-layer fix into (a) the affine correction and (b) the actual Jacobian transport. Zero
fitting: the lens is exactly I per fitted layer.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import torch  # noqa: I001  # the vlm_lens import must precede jlens: it installs the vendored path

import vlm_lens  # noqa: F401,E402  # ensure_jlens() runs on package import
from jlens.lens import JacobianLens  # noqa: E402

from vlm_lens.artifacts import load_lens_set, save_lens_set  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src-lens-dir", required=True)
    parser.add_argument("--mask", default="text")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    lenses, _ = load_lens_set(args.src_lens_dir)
    src = lenses[args.mask]
    d = src.d_model
    ident = JacobianLens(
        jacobians={layer: torch.eye(d, dtype=torch.float32) for layer in src.source_layers},
        n_prompts=src.n_prompts,
        d_model=d,
    )
    out = Path(args.out_dir)
    save_lens_set(
        out / "artifacts",
        {args.mask: ident},
        provenance={
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "note": "identity lens for the bias-only decomposition (R3 P5; synthetic, zero fitting)",
            "derived_from": {"lens_dir": str(args.src_lens_dir), "mask": args.mask},
        },
    )
    print(
        f"wrote identity lens: layers {src.source_layers[0]}..{src.source_layers[-1]}, "
        f"d={d}, n_prompts={src.n_prompts}"
    )
    print(f"wrote {out / 'artifacts' / f'lens-{args.mask}.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
