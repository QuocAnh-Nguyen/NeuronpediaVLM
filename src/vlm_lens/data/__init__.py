# SPDX-License-Identifier: Apache-2.0
"""Decoupled dataset layer: corpora in, ``FitSample`` manifests out.

Nothing here imports the model except the optional caption generator; the fit scripts
consume manifests only. That keeps the fitting core swappable across datasets (COCO
captions now, POPE/yes-no later, arbitrary text for the S1 control fit).
"""

from vlm_lens.data.manifest import (
    FitSample,
    manifest_meta,
    read_manifest,
    sha256_file,
    write_manifest,
)

__all__ = [
    "FitSample",
    "manifest_meta",
    "read_manifest",
    "sha256_file",
    "write_manifest",
]
