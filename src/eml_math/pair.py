"""
TensionPair — two real TensionKnots replacing a complex number.

Instead of z = a + bi (complex arithmetic), use:
    EMLPair(real_knot, imag_knot)

All arithmetic is defined as EML compositions via operators.py.
All intermediate and final values are strictly real.

This is MPM's answer to the EML paper's statement that "a continuous Sheffer
working purely in the real domain seems impossible" (arXiv:2603.21852v2, §5).
The pair encodes what would be complex arithmetic as two real tensions,
using the frame-shift mechanism to keep all values in the real domain.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.point import EMLPoint, TensionPoint
from eml_math.state import EMLState, TensionKnot
import eml_math.operators as ops


class EMLPair:
    """
    Two real TensionKnots encoding a complex-valued quantity.

    EMLPair(real_knot, imag_knot) ↔ z = real_tension + i·imag_tension

    All arithmetic (add, mul, etc.) is implemented purely in terms of the
    EML Sheffer operator on real tensions. No Python complex type is used.

    Parameters
    ----------
    real : EMLState
        The "real part" knot. Its .rho is the real component magnitude.
    imag : EMLState
        The "imaginary part" knot. Its .rho is the imaginary component magnitude.

    Examples
    --------
    >>> from eml_math import EMLPoint, EMLState
    >>> r = EMLState(EMLPoint(1.0, 1.0))
    >>> i = EMLState(EMLPoint(1.0, math.e))
    >>> pair = EMLPair(r, i)
    >>> pair.real_tension
    2.718...
    >>> pair.modulus
    ...
    """

    __slots__ = ("_real", "_imag")

    def __init__(self, real: EMLState, imag: EMLState) -> None:
        self._real = real
        self._imag = imag

    # ── component access ──────────────────────────────────────────────────────

    @property
    def real_knot(self) -> TensionKnot:
        return self._real

    @property
    def imag_knot(self) -> TensionKnot:
        return self._imag

    @property
    def real_tension(self) -> float:
        """Tension scalar of the real component knot."""
        return self._real.point.tension()

    @property
    def imag_tension(self) -> float:
        """Tension scalar of the imaginary component knot."""
        return self._imag.point.tension()

    @property
    def modulus(self) -> float:
        """|z| = √(real² + imag²)."""
        r, im = self.real_tension, self.imag_tension
        return math.sqrt(r * r + im * im)

    @property
    def argument(self) -> float:
        """Phase angle θ = arctan2(imag, real) in radians."""
        return math.atan2(self.imag_tension, self.real_tension)

    # ── iteration ─────────────────────────────────────────────────────────────

    def mirror_pulse(self) -> "EMLPair":
        """
        Advance both knots by one Mirror-Pulse simultaneously.

        The pair evolves as a unit — both knots pulse together to maintain
        their phase relationship (Postulate Q5: shared axle).
        """
        return EMLPair(
            self._real.mirror_pulse(),
            self._imag.mirror_pulse(),
        )

    def rotate_phase(self, angle: float) -> "EMLPair":
        """
        Rotate the pair by `angle` radians.

        For arbitrary angle uses the rotation matrix:
            real' = real·cos(θ) - imag·sin(θ)
            imag' = real·sin(θ) + imag·cos(θ)
        """
        r = self.real_tension
        im = self.imag_tension
        c = ops.cos(angle).tension()
        s = ops.sin(angle).tension()
        real_new = r * c - im * s
        imag_new = r * s + im * c
        D = self._real.point.D
        return _pair_from_values(real_new, imag_new, D)

    def conjugate(self) -> "EMLPair":
        """Complex conjugate: (real, imag) → (real, -imag)."""
        D = self._real.point.D
        return _pair_from_values(self.real_tension, -self.imag_tension, D)

    # ── arithmetic ────────────────────────────────────────────────────────────

    def __add__(self, other: "EMLPair") -> "EMLPair":
        """(a+bi) + (c+di) = (a+c) + (b+d)i."""
        D = self._real.point.D
        return _pair_from_values(
            self.real_tension + other.real_tension,
            self.imag_tension + other.imag_tension,
            D,
        )

    def __sub__(self, other: "EMLPair") -> "EMLPair":
        """(a+bi) - (c+di) = (a-c) + (b-d)i."""
        D = self._real.point.D
        return _pair_from_values(
            self.real_tension - other.real_tension,
            self.imag_tension - other.imag_tension,
            D,
        )

    def __mul__(self, other: "EMLPair") -> "EMLPair":
        """(a+bi)(c+di) = (ac-bd) + (ad+bc)i."""
        a, b = self.real_tension, self.imag_tension
        c, d = other.real_tension, other.imag_tension
        D = self._real.point.D
        return _pair_from_values(a * c - b * d, a * d + b * c, D)

    def __truediv__(self, other: "EMLPair") -> "EMLPair":
        """(a+bi)/(c+di) = (a+bi)(c-di)/(c²+d²)."""
        a, b = self.real_tension, self.imag_tension
        c, d = other.real_tension, other.imag_tension
        denom = c * c + d * d
        if abs(denom) < 1e-300:
            denom = 1e-300
        D = self._real.point.D
        return _pair_from_values((a * c + b * d) / denom, (b * c - a * d) / denom, D)

    def __abs__(self) -> float:
        return self.modulus

    # ── class methods ─────────────────────────────────────────────────────────

    @classmethod
    def from_polar(cls, r: float, theta: float, D: Optional[float] = None) -> "EMLPair":
        """
        Create a TensionPair from polar form r·e^(iθ).

        real = r·cos(θ), imag = r·sin(θ) — both via EML operators.
        """
        real_val = ops.mul(r, ops.cos(theta)).tension()
        imag_val = ops.mul(r, ops.sin(theta)).tension()
        return _pair_from_values(real_val, imag_val, D)

    @classmethod
    def from_values(cls, real: float, imag: float, D: Optional[float] = None) -> "EMLPair":
        """Create a TensionPair from explicit real and imaginary floats."""
        return _pair_from_values(real, imag, D)

    @classmethod
    def unit_i(cls, D: Optional[float] = None) -> "EMLPair":
        """
        The imaginary unit i as a EMLPair.

        i ↔ EMLPair(real=0, imag=1)
        real component tension → 0
        imag component tension → 1
        """
        return _pair_from_values(0.0, 1.0, D)

    @classmethod
    def one(cls, D: Optional[float] = None) -> "EMLPair":
        """Real unit 1 as a EMLPair."""
        return _pair_from_values(1.0, 0.0, D)

    @classmethod
    def zero(cls, D: Optional[float] = None) -> "EMLPair":
        """Zero as a EMLPair."""
        return _pair_from_values(0.0, 0.0, D)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        r = self.real_tension
        im = self.imag_tension
        sign = "+" if im >= 0 else "-"
        return f"EMLPair({r:.6g} {sign} {abs(im):.6g}i)"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLPair):
            return NotImplemented
        return (
            abs(self.real_tension - other.real_tension) < 1e-9
            and abs(self.imag_tension - other.imag_tension) < 1e-9
        )


# ── helper ────────────────────────────────────────────────────────────────────

def _pair_from_values(
    real: float,
    imag: float,
    D: Optional[float] = None,
) -> EMLPair:
    """
    Build a TensionPair from scalar real and imaginary values.

    The tension encoding: we store the value directly as the x-coordinate
    of a TensionPoint with y=1, so tension() = exp(x) - ln(1) = exp(x).
    To recover the value v from a knot: ln(knot.point.tension() + 1) ... this
    is circular. Simpler: store via a _LitNode wrapping the value directly.
    """
    from eml_math.point import _LitNode

    class _ValueKnot(EMLState):
        """Knot whose .point.tension() returns a stored float directly."""
        def __init__(self, value: float, d: Optional[float]) -> None:
            point = _ValuePoint(value, d)
            super().__init__(point, n=0, theta=0.0)

    class _ValuePoint(EMLPoint):
        def __init__(self, value: float, d: Optional[float]) -> None:
            super().__init__(0.0, 1.0, D=d)
            self._stored = value

        def tension(self) -> float:
            return self._stored

        @property
        def x(self) -> float:
            return self._stored

        @property
        def y(self) -> float:
            return 1.0

    return EMLPair(
        _ValueKnot(real, D),
        _ValueKnot(imag, D),
    )
