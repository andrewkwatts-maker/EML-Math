"""
Constants used internally by eml-math core.

Physical constants (Planck length/energy, fundamental units, etc.) live
in the sister package ``eml-spectral`` — this module keeps only the
defaults the core needs to operate.
"""
from __future__ import annotations

import math
import sys

# ── Numerical guards ─────────────────────────────────────────────────────────

OVERFLOW_THRESHOLD: float = math.log(sys.float_info.max)
"""Maximum safe ``x`` before ``exp(x)`` overflows in IEEE-754.

``EMLPoint.iterate()`` clamps ``x`` to ``ln(x)`` when it exceeds this
threshold — the same self-braking guard that powered v1.x's pulse."""

# ── Discrete-mode quantization defaults ──────────────────────────────────────

PLANCK_D: float = 6.187e34
"""Planck-scale quantization factor ``D = 1 / ℓ_Planck`` — the default
for ``EMLPoint(D=…)`` discrete mode. Past v1.x compat: callers used to
pass this as the ``D`` argument; still works."""

DEFAULT_D = PLANCK_D
"""Alias for :data:`PLANCK_D` — kept for backwards compatibility."""

# ── Iteration shape ──────────────────────────────────────────────────────────

FLIP_YIELD: int = 2
"""Net reality units created per complete 3:1 flip cycle. Used by
``eml_spectral.iterate.simulate_flips`` for the trajectory-counting
helpers."""

FLIP_RATIO: tuple[int, int] = (3, 1)
"""Growth-to-reflection ratio of one flip: 3 direct + 1 mirrored."""
