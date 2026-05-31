"""Pytest configuration shared across the suite.

The `src/` modules use flat imports (`from load import ...`, `from prepare
import ...`) rather than package-relative ones, so `src/` has to be on
sys.path before any test imports them. Doing it here means individual test
files can just `import features`, `import models`, etc.
"""
from __future__ import annotations

import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / 'src'
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
