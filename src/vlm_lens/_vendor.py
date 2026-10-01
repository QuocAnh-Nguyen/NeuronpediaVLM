# SPDX-License-Identifier: Apache-2.0
"""Make the vendored Anthropic reference implementation importable as ``jlens``.

The reference lives in ``third_party/jacobian-lens`` (see ``third_party/NOTICE.md``).
We prepend that directory to ``sys.path`` so imports are reproducible regardless of
whether a ``jlens`` distribution also happens to be installed in the environment.
The vendored tree wins when it is present (it is prepended to ``sys.path``); an
installed ``jlens`` distribution is used only as a fallback when the checkout is absent.
"""

from __future__ import annotations

import sys
from pathlib import Path

VENDOR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "jacobian-lens"

_DONE = False


def ensure_jlens() -> str:
    """Prepend the vendored reference to ``sys.path`` and return the import path.

    Idempotent. The vendored tree is preferred when it exists; otherwise an installed
    ``jlens`` is imported (and the import error propagates if neither exists).
    """
    global _DONE
    if not _DONE:
        if VENDOR_DIR.is_dir():
            path = str(VENDOR_DIR)
            if path not in sys.path:
                sys.path.insert(0, path)
        _DONE = True
    import jlens  # noqa: PLC0415  (import after sys.path setup, by design)

    return str(getattr(jlens, "__file__", "?"))
