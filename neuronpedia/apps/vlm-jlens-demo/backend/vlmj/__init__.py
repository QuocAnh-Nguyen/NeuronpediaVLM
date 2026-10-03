# SPDX-License-Identifier: Apache-2.0
"""``vlmj`` — demo backend for the interactive J-Lens VLM workbench.

The package wraps the research code in ``vlm_lens`` (Jacobian lenses for
LLaVA-1.5-7B) in a small FastAPI service: encode a session, generate a caption,
read lens distributions, attribute them over the 24x24 patch grid, knock patches
out through the multimodal projector and steer the residual stream.

Importing this package makes ``vlm_lens`` importable; when the library is not on
``sys.path`` the source checkout named by ``VLMJ_LENS_SRC`` (default
``/home/vacpls/Workspace/NeuronpediaVLM/src``) is prepended.
"""

from __future__ import annotations

import os
import sys

__all__ = ["ensure_vlm_lens"]

DEFAULT_VLM_LENS_SRC = "/home/vacpls/Workspace/NeuronpediaVLM/src"


def ensure_vlm_lens() -> str:
    """Import ``vlm_lens``, retrying from ``VLMJ_LENS_SRC`` on ``ImportError``.

    Idempotent. Returns the module's file path. Raises the original
    ``ImportError`` when the fallback path does not provide the package either.
    """
    try:
        import vlm_lens  # noqa: PLC0415
    except ImportError:
        source = os.environ.get("VLMJ_LENS_SRC", DEFAULT_VLM_LENS_SRC)
        if source and source not in sys.path:
            sys.path.insert(0, source)
        import vlm_lens  # noqa: PLC0415

    return str(getattr(vlm_lens, "__file__", "?"))


ensure_vlm_lens()
