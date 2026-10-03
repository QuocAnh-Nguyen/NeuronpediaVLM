# SPDX-License-Identifier: Apache-2.0
"""Make the backend package importable however pytest is launched.

``python -m pytest`` puts the working directory on ``sys.path``, but a bare
``pytest`` call does not; this conftest pins the behaviour so ``import vlmj``
works either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

BACKEND_ROOT = Path(__file__).resolve().parents[1]
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
