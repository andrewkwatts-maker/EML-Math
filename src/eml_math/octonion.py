"""
Octonion — 8-component non-associative normed division algebra.

Components are EMLPoints. Multiplication uses the standard Fano-plane
multiplication table. Norm is multiplicative: |ab| = |a|·|b|.
"""
from __future__ import annotations

import math
from typing import List, Callable, Optional

from eml_math.point import EMLPoint

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False

try:
    from eml_math import eml_core as _core
    _RUST_OCTONION = True
except ImportError:
    _RUST_OCTONION = False


# ── Fano-plane multiplication table ──────────────────────────────────────────
# MULT_TABLE[i][j] = (sign, index) meaning e_i * e_j = sign * e_index
# For i==j: e_i * e_i = -1 (for i>0), e_0 * e_0 = +1 (real unit)
# Table for e_1..e_7 follows the standard (124)(235)(346)(457)(156)(267)(137) lines.

_FANO_LINES = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (1, 5, 6),
    (2, 6, 7),
    (1, 3, 7),
]

def _build_mult_table():
    # Returns an 8×8 array of (sign: int, index: int)
    table = [[(1, 0)] * 8 for _ in range(8)]
    # e_0 is real unit
    for i in range(8):
        table[0][i] = (1, i)
        table[i][0] = (1, i)
    # e_i * e_i = -e_0 for i>0
    for i in range(1, 8):
        table[i][i] = (-1, 0)
    # Fill from Fano lines: for (a,b,c): e_a*e_b=e_c, e_b*e_c=e_a, e_c*e_a=e_b
    for a, b, c in _FANO_LINES:
        table[a][b] = (1, c);  table[b][a] = (-1, c)
        table[b][c] = (1, a);  table[c][b] = (-1, a)
        table[c][a] = (1, b);  table[a][c] = (-1, b)
    return table

MULT_TABLE = _build_mult_table()


# ── Octonion class ────────────────────────────────────────────────────────────

class Octonion:
    """
    An octonion with 8 EMLPoint components.

    The real unit is component 0 (e_0). Components 1–7 are the imaginary units.
    Multiplication is non-associative; division algebra property holds via the
    Moufang identities.

    Parameters
    ----------
    components : list[EMLPoint]
        Exactly 8 EMLPoints. The x-coordinate of each EMLPoint is used as the
        scalar coefficient.
    """

    __slots__ = ("_comps",)

    def __init__(self, components: List[EMLPoint]) -> None:
        if len(components) != 8:
            raise ValueError(
                f"Octonion requires exactly 8 components, got {len(components)}"
            )
        self._comps = list(components)

    # ── component access ──────────────────────────────────────────────────────

    def _scalars(self) -> List[float]:
        """Extract scalar coefficients (x-values)."""
        return [p.x for p in self._comps]

    def component(self, i: int) -> float:
        """Return the scalar coefficient of e_i."""
        return self._comps[i].x

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __mul__(self, other: "Octonion") -> "Octonion":
        """Octonion product using the Fano-plane table."""
        a = self._scalars()
        b = other._scalars()
        if _RUST_OCTONION:
            result = list(_core.octonion_mul_n([a], [b])[0])
        else:
            result = [0.0] * 8
            for i in range(8):
                for j in range(8):
                    sign, k = MULT_TABLE[i][j]
                    result[k] += sign * a[i] * b[j]
        return Octonion([EMLPoint(v, 1.0) for v in result])

    def __add__(self, other: "Octonion") -> "Octonion":
        a = self._scalars()
        b = other._scalars()
        return Octonion([EMLPoint(ai + bi, 1.0) for ai, bi in zip(a, b)])

    def __sub__(self, other: "Octonion") -> "Octonion":
        a = self._scalars()
        b = other._scalars()
        return Octonion([EMLPoint(ai - bi, 1.0) for ai, bi in zip(a, b)])

    def conjugate(self) -> "Octonion":
        """Octonion conjugate: flip sign of imaginary components e_1..e_7."""
        scalars = self._scalars()
        result = [scalars[0]] + [-s for s in scalars[1:]]
        return Octonion([EMLPoint(v, 1.0) for v in result])

    def norm_sq(self) -> float:
        """Squared norm = Σ aᵢ²."""
        return sum(s * s for s in self._scalars())

    def norm(self) -> float:
        """Euclidean norm √(Σ aᵢ²). Multiplicative: |ab| = |a|·|b|."""
        return math.sqrt(self.norm_sq())

    def to_ndvector(self) -> "EMLNDVector":
        """Convert to an EMLNDVector of dimension 8."""
        from eml_math.ndim import EMLNDVector
        return EMLNDVector(list(self._comps))

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        s = self._scalars()
        return f"Octonion({s[0]:.4g} + {s[1]:.4g}e1 + {s[2]:.4g}e2 + ...)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Octonion):
            return NotImplemented
        return all(
            abs(self.component(i) - other.component(i)) < 1e-9
            for i in range(8)
        )


# ── helpers ───────────────────────────────────────────────────────────────────

def basis_octonion(i: int) -> Octonion:
    """
    Return the i-th basis octonion e_i (i = 0..7).

    e_0 is the real unit (1 + 0·e_1 + ... + 0·e_7).
    """
    if not 0 <= i <= 7:
        raise ValueError(f"basis index must be 0–7, got {i}")
    scalars = [0.0] * 8
    scalars[i] = 1.0
    return Octonion([EMLPoint(s, 1.0) for s in scalars])


def is_g2_automorphism(
    o1: Octonion,
    o2: Octonion,
    g2_map: Callable[[Octonion], Octonion],
) -> bool:
    """
    Check the automorphism condition at the specific pair (o1, o2) — not globally.

    Returns True if g2_map satisfies:
        g2_map(o1 * o2) == g2_map(o1) * g2_map(o2)   (multiplicativity at this pair)
        |g2_map(o1)| == |o1|                           (norm preservation at o1)

    This is a pointwise test only. Passing for a single pair (o1, o2) does NOT
    prove that g2_map is a global automorphism of the algebra. To verify a true
    G₂ automorphism, the condition must hold for all pairs in the algebra.

    Parameters
    ----------
    o1, o2 : Octonion
    g2_map : Callable[[Octonion], Octonion]
        Candidate automorphism map.
    """
    product_then_map = g2_map(o1 * o2)
    map_then_product = g2_map(o1) * g2_map(o2)
    preserves_product = (product_then_map == map_then_product)
    preserves_norm = abs(g2_map(o1).norm() - o1.norm()) < 1e-9
    return preserves_product and preserves_norm
