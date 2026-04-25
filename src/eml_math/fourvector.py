"""
MinkowskiFourVector — relativistic four-vector in 3+1 dimensions.

Encodes (t, x, y, z) as four EMLPoints. The time component uses EML encoding
(energy = exp(x_t)), spatial components use direct EML coordinates.
"""
from __future__ import annotations

import math
from typing import Optional, List

from eml_math.point import EMLPoint

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class MinkowskiFourVector:
    """
    Four-vector in Minkowski spacetime: (t, x, y, z).

    Parameters
    ----------
    t, x, y, z : EMLPoint
        The four components. ``t`` is the time component.
    c : float
        Speed of light (default 1.0).
    metric : optional
        A MetricTensor instance. When None, uses flat Minkowski.
    """

    __slots__ = ("_t", "_x", "_y", "_z", "_c")

    def __init__(
        self,
        t: EMLPoint,
        x: EMLPoint,
        y: EMLPoint,
        z: EMLPoint,
        c: float = 1.0,
        metric=None,
    ) -> None:
        self._t = t
        self._x = x
        self._y = y
        self._z = z
        self._c = c

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def t_component(self) -> EMLPoint:
        return self._t

    @property
    def spatial_components(self) -> List[EMLPoint]:
        return [self._x, self._y, self._z]

    @property
    def c(self) -> float:
        return self._c

    # ── physical quantities ───────────────────────────────────────────────────

    def four_momentum(self, mass: Optional[float] = None):
        """
        Return the four-momentum [E/c, p_x, p_y, p_z] as a list of floats.

        E = exp(t.x), p_i = ln(|spatial_i.y|) / c.

        If numpy is available, returns np.ndarray; otherwise a plain list.
        """
        from eml_math.constants import OVERFLOW_THRESHOLD
        tv = math.log(self._t.x) if self._t.x > OVERFLOW_THRESHOLD else self._t.x
        E = math.exp(tv) / self._c

        def _p(p: EMLPoint) -> float:
            y_safe = max(abs(p.y), 1e-300)
            return math.log(y_safe) / self._c

        components = [E, _p(self._x), _p(self._y), _p(self._z)]
        if _HAS_NUMPY:
            return np.array(components, dtype=float)
        return components

    def minkowski_norm(self) -> float:
        """
        Minkowski spacetime interval in 3+1 dimensions.

        Returns √|g_{μν} x^μ x^ν| = √|c²t² - x² - y² - z²|
        using the (+,-,-,-) metric signature convention, i.e.
        g = diag(+c², -1, -1, -1).

        For a timelike separation this equals the proper time times c;
        for a spacelike separation it equals the proper distance.
        Uses the x-coordinate of each EMLPoint as the scalar value.
        """
        ct = self._c * self._t.x
        sx = self._x.x
        sy = self._y.x
        sz = self._z.x
        ds2 = ct * ct - sx * sx - sy * sy - sz * sz
        return math.sqrt(abs(ds2))

    def boost(self, rapidity_phi: float, direction: str = "x") -> "MinkowskiFourVector":
        """
        Apply a Lorentz boost by rapidity φ along the given direction.

        Sequential boosts in the same direction are additive in rapidity:
        boosting by φ₁ then φ₂ is equivalent to a single boost by φ₁ + φ₂.
        This additivity property makes rapidity the natural boost parameter.

        Parameters
        ----------
        rapidity_phi : float
            Boost rapidity φ. Related to velocity by φ = atanh(v/c).
        direction : str
            "x", "y", or "z" — boost direction.
        """
        ch = math.cosh(rapidity_phi)
        sh = math.sinh(rapidity_phi)

        t_val = self._t.x
        x_val = self._x.x
        y_val = self._y.x
        z_val = self._z.x

        if direction == "x":
            t_new = t_val * ch - x_val * sh
            x_new = x_val * ch - t_val * sh
            y_new = y_val
            z_new = z_val
        elif direction == "y":
            t_new = t_val * ch - y_val * sh
            x_new = x_val
            y_new = y_val * ch - t_val * sh
            z_new = z_val
        elif direction == "z":
            t_new = t_val * ch - z_val * sh
            x_new = x_val
            y_new = y_val
            z_new = z_val * ch - t_val * sh
        else:
            raise ValueError(f"direction must be 'x', 'y', or 'z', got {direction!r}")

        return MinkowskiFourVector(
            EMLPoint(t_new, self._t.y),
            EMLPoint(x_new, self._x.y),
            EMLPoint(y_new, self._y.y),
            EMLPoint(z_new, self._z.y),
            c=self._c,
        )

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"MinkowskiFourVector(t={self._t.x:.6g}, x={self._x.x:.6g}, "
            f"y={self._y.x:.6g}, z={self._z.x:.6g})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MinkowskiFourVector):
            return NotImplemented
        return (
            abs(self._t.x - other._t.x) < 1e-9
            and abs(self._x.x - other._x.x) < 1e-9
            and abs(self._y.x - other._y.x) < 1e-9
            and abs(self._z.x - other._z.x) < 1e-9
        )
