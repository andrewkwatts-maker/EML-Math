"""
Physical and mathematical constants for Mirror Phase Mathematics.

All values are pure definitions — no computation here.
"""
from __future__ import annotations

import math
import sys

# ── Planck-scale quantization ──────────────────────────────────────────────────

PLANCK_D: float = 6.187e34
"""Planck-scale quantization factor D = 1 / l_planck.

Used in discrete mode only. Pass as D= to TensionPoint/TensionKnot
to enable Planck-quantized integer steps.
"""

PLANCK_LENGTH: float = 1.616255e-35
"""Planck length in metres."""

PLANCK_ENERGY: float = 1.956e9
"""Planck energy in Joules."""

# ── EML / MPM dynamics ────────────────────────────────────────────────────────

FLIP_YIELD: int = 2
"""Net reality units created per complete 3:1 Flip cycle (Axiom 9)."""

FLIP_RATIO: tuple[int, int] = (3, 1)
"""Growth-to-reflection ratio of one Flip: 3 direct + 1 mirrored."""

OVERFLOW_THRESHOLD: float = math.log(sys.float_info.max)
"""Maximum safe x before exp(x) overflows (the Slipping Wheel threshold).

For x above this value, mirror_pulse() applies logarithmic dampening:
  x → ln(x)  (the mirror operator applied to itself — self-braking).
"""

# ── Backwards-compat alias ────────────────────────────────────────────────────

DEFAULT_D = PLANCK_D
"""Alias for PLANCK_D. Prefer PLANCK_D in new code."""
