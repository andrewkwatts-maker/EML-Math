"""
FourMomentum — relativistic four-momentum encoded as an EMLPoint.

Energy maps to exp(x) and spatial momentum to ln(|y|)/c.
All quantities are purely real and derived from the EML operator.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.point import EMLPoint


class FourMomentum:
    """
    Relativistic four-momentum (energy, spatial momentum) encoded as an EMLPoint.

    Encoding
    --------
    - Energy    E  = exp(x_coord)
    - Momentum  p  = ln(|y_coord|) / c

    The invariant mass follows from the Minkowski delta:
        m = Δ_M / c²  where  Δ_M = √|E² − (p·c)²|

    Parameters
    ----------
    point : EMLPoint
        The EML coordinate pair (x, y).
    c : float
        Speed of light constant (default 1.0).
    """

    __slots__ = ("_point", "_c")

    def __init__(self, point: EMLPoint, c: float = 1.0) -> None:
        self._point = point
        self._c = c

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def point(self) -> EMLPoint:
        return self._point

    @property
    def c(self) -> float:
        return self._c

    @property
    def energy(self) -> float:
        """Relativistic energy E = exp(x)."""
        from eml_math.constants import OVERFLOW_THRESHOLD
        xv = math.log(self._point.x) if self._point.x > OVERFLOW_THRESHOLD else self._point.x
        return math.exp(xv)

    @property
    def momentum(self) -> float:
        """Spatial momentum p = ln(|y|) / c."""
        y = self._point.y
        y_safe = abs(y) if y <= 0 else y
        y_safe = max(y_safe, 1e-300)
        return math.log(y_safe) / self._c

    @property
    def mass(self) -> float:
        """Invariant mass m = Δ_M / c²."""
        return self._point.minkowski_delta(c=self._c) / (self._c * self._c)

    def gamma(self) -> float:
        """
        Lorentz factor γ = E / (mc²) for massive particles.

        For massive particles: γ = E / (mc²) ≥ 1.
        For massless particles (m = 0): γ is physically undefined — a photon
        has no rest frame. This function returns float('inf') as a
        computational convention only; it does not imply infinite energy.
        """
        m = self.mass
        if m < 1e-300:
            return math.inf
        return self.energy / (m * self._c * self._c)

    def boost(self, phi: float) -> "FourMomentum":
        """
        Apply a Lorentz boost by rapidity phi.

        Returns a new FourMomentum with the boosted EMLPoint.
        The invariant mass (Δ_M) is preserved.
        """
        boosted = self._point.boost(phi, c=self._c)
        return FourMomentum(boosted, c=self._c)

    @classmethod
    def from_mass_velocity(
        cls,
        mass: float,
        v: float,
        c: float = 1.0,
    ) -> "FourMomentum":
        """
        Construct a FourMomentum from rest mass and velocity.

        Uses the standard relativistic relations:
            E  = γ·m·c²
            p  = γ·m·v

        Parameters
        ----------
        mass : float
            Rest mass (m > 0).
        v : float
            Velocity. Must satisfy |v| < c.
        c : float
            Speed of light (default 1.0).
        """
        if abs(v) >= c:
            raise ValueError(
                f"velocity |v|={abs(v):.6g} must be strictly less than c={c:.6g}"
            )
        if mass <= 0:
            raise ValueError(f"mass must be positive, got {mass}")

        beta = v / c
        gamma = 1.0 / math.sqrt(1.0 - beta * beta)
        E = gamma * mass * c * c
        p = gamma * mass * v

        # Encode: E = exp(x) → x = ln(E); p = ln(y)/c → y = exp(p*c)
        x = math.log(max(E, 1e-300))
        y = math.exp(max(min(p * c, 709.0), -709.0))
        return cls(EMLPoint(x, y), c=c)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"FourMomentum(E={self.energy:.6g}, p={self.momentum:.6g}, "
            f"m={self.mass:.6g}, c={self._c})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, FourMomentum):
            return NotImplemented
        return (
            abs(self.energy - other.energy) < 1e-9
            and abs(self.momentum - other.momentum) < 1e-9
            and abs(self._c - other._c) < 1e-12
        )
