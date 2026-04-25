"""
MetricTensor — EML-native spacetime metric with analytic and numeric Christoffel symbols.

Encodes general-relativistic and higher-dimensional metrics in terms of EMLPoint
coordinates. The EML encoding maps: time-coordinate ↔ exp(x), space-coordinate ↔ ln(y).
"""
from __future__ import annotations

import math
from typing import Callable, Optional, List

try:
    from eml_math import eml_core as _core
    _RUST_METRIC = True
except ImportError:
    _RUST_METRIC = False

from eml_math.point import EMLPoint


class MetricTensor:
    """
    A spacetime metric g_{μν}(p) operating in EML coordinate space.

    The metric tensor is defined by a callable ``g`` that takes an EMLPoint
    and returns a 2×2 (or dim×dim) list of lists representing g_{μν}.

    Parameters
    ----------
    g : Callable[[EMLPoint], list[list[float]]]
        Function mapping an EMLPoint to the metric components g_{μν}.
    dim : int
        Dimension of the metric (default 2 for the EML (t, x) plane).
    """

    __slots__ = ("_g", "_dim", "_schwarzschild_rs")

    def __init__(
        self,
        g: Callable[[EMLPoint], List[List[float]]],
        dim: int = 2,
    ) -> None:
        self._g = g
        self._dim = dim
        self._schwarzschild_rs: Optional[float] = None

    # ── line element ──────────────────────────────────────────────────────────

    def ds2(self, p: EMLPoint, dx: float = 0.0, dy: float = 0.0) -> float:
        """
        Compute the line element ds² = g_{μν} dx^μ dx^ν at point p.

        ds² is the line element (infinitesimal squared interval). It may be
        positive (spacelike), negative (timelike), or zero (lightlike) depending
        on the metric signature and the direction of the displacement.
        In the 2D EML plane the displacement vector is (dx, dy).
        """
        g = self._g(p)
        return g[0][0] * dx * dx + (g[0][1] + g[1][0]) * dx * dy + g[1][1] * dy * dy

    def proper_time(self, p: EMLPoint, dx: float = 0.0, dy: float = 0.0) -> float:
        """Proper time element: dτ = √|ds²| = √|g_{μν} dx^μ dx^ν|.

        τ (tau) is the standard symbol for proper time.
        """
        return math.sqrt(abs(self.ds2(p, dx, dy)))

    def is_curved(self, p: Optional[EMLPoint] = None, tol: float = 1e-9) -> bool:
        """
        Return True if the metric differs from flat Minkowski (+, -) at point p.

        Uses the probe point EMLPoint(1.0, math.e) if p is None.
        """
        if p is None:
            p = EMLPoint(1.0, math.e)
        g = self._g(p)
        flat = [[1.0, 0.0], [0.0, -1.0]]
        for i in range(min(2, self._dim)):
            for j in range(min(2, self._dim)):
                if abs(g[i][j] - flat[i][j]) > tol:
                    return True
        return False

    # ── Christoffel symbols ───────────────────────────────────────────────────

    def christoffel(
        self,
        lam: int,
        mu: int,
        nu: int,
        p: EMLPoint,
        h: float = 1e-5,
    ) -> float:
        """
        Numeric Christoffel symbol Γ^λ_{μν} via central finite differences.

        Uses the standard formula:
            Γ^λ_{μν} = ½ g^{λσ} (∂_μ g_{νσ} + ∂_ν g_{μσ} - ∂_σ g_{μν})

        Note: indices follow the standard upper/lower convention Γ^λ_{μν}
        (upper index λ is the contravariant "result" index; lower indices
        μ, ν are the covariant "derivative" indices).

        Limitation: implemented for the 2D case (dim=2) only. For higher
        dimensions the caller is responsible for ensuring the metric is 2×2.

        Parameters
        ----------
        lam, mu, nu : int
            Index triple following the Γ^λ_{μν} convention.
        p : EMLPoint
            Point at which to evaluate.
        h : float
            Finite-difference step size.
        """
        if _RUST_METRIC and self._schwarzschild_rs is not None:
            return _core.christoffel_batch_n(
                [(p.x, p.y)], lam, mu, nu, self._schwarzschild_rs
            )[0]

        x, y = p.x, p.y

        def g_at(xi: float, yi: float) -> List[List[float]]:
            yi_safe = max(yi, 1e-300)
            return self._g(EMLPoint(xi, yi_safe))

        def dg_dx(i: int, j: int) -> float:
            return (g_at(x + h, y)[i][j] - g_at(x - h, y)[i][j]) / (2 * h)

        def dg_dy(i: int, j: int) -> float:
            return (g_at(x, y + h)[i][j] - g_at(x, y - h)[i][j]) / (2 * h)

        dg = [[dg_dx, dg_dy], [dg_dx, dg_dy]]

        partials = [dg_dx, dg_dy]

        g = g_at(x, y)
        # Invert 2×2 metric
        det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
        if abs(det) < 1e-300:
            return 0.0
        ginv = [
            [g[1][1] / det, -g[0][1] / det],
            [-g[1][0] / det, g[0][0] / det],
        ]

        result = 0.0
        for sig in range(2):
            term = (
                partials[mu](nu, sig)
                + partials[nu](mu, sig)
                - partials[sig](mu, nu)
            )
            result += 0.5 * ginv[lam][sig] * term
        return result

    @staticmethod
    def schwarzschild_christoffel(
        lam: int,
        mu: int,
        nu: int,
        r: float,
        rs: float = 2.0,
    ) -> float:
        """
        Analytic non-zero Christoffel symbols for the Schwarzschild metric.

        Only the (t, r) components are computed for the 2D radial slice.
        r is the radial coordinate, rs = 2GM/c² is the Schwarzschild radius.

        Non-zero symbols in 2D (t=0, r=1):
            Γ^t_{tr} = Γ^t_{rt} = rs / (2r(r - rs))
            Γ^r_{tt} = rs(r - rs) / (2r³)
            Γ^r_{rr} = -rs / (2r(r - rs))
        """
        if r <= rs or r <= 0:
            return 0.0
        f = 1.0 - rs / r
        if lam == 0 and mu == 0 and nu == 1:
            return rs / (2.0 * r * (r - rs))
        if lam == 0 and mu == 1 and nu == 0:
            return rs / (2.0 * r * (r - rs))
        if lam == 1 and mu == 0 and nu == 0:
            return rs * f / (2.0 * r * r)
        if lam == 1 and mu == 1 and nu == 1:
            return -rs / (2.0 * r * (r - rs))
        return 0.0

    # ── factory methods ───────────────────────────────────────────────────────

    @classmethod
    def flat(cls) -> "MetricTensor":
        """Flat Minkowski metric (+, -): g = diag(1, -1)."""
        def _g(p: EMLPoint) -> List[List[float]]:
            return [[1.0, 0.0], [0.0, -1.0]]
        return cls(_g, dim=2)

    @classmethod
    def schwarzschild(cls, rs: float = 2.0) -> "MetricTensor":
        """
        Schwarzschild metric in the EML (t, r) plane.

        g_{tt} = -(1 - rs/r),  g_{rr} = +1/(1 - rs/r)
        where r = exp(x) (the EML time-coordinate acts as the radial variable).

        Signature convention: (-,+) — i.e. g_{tt} < 0, g_{rr} > 0.
        This is the opposite sign convention from the flat() metric, which
        uses (+,-). When mixing flat() and schwarzschild() results, take care
        to account for this signature difference.
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            r = math.exp(p.x) if p.x < 709.0 else math.exp(709.0)
            r = max(r, rs + 1e-9)
            f = 1.0 - rs / r
            f = max(f, 1e-300)
            return [[-(f), 0.0], [0.0, 1.0 / f]]
        m = cls(_g, dim=2)
        m._schwarzschild_rs = rs
        return m

    @classmethod
    def flrw(
        cls,
        scale_factor_a: Callable[[float], float],
        k: float = 0.0,
    ) -> "MetricTensor":
        """
        FLRW cosmological metric.

        g_{tt} = -1,  g_{rr} = a(t)² / (1 - k·r²)
        where t = exp(x), r = ln(|y|).
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            t = math.exp(p.x) if p.x < 709.0 else math.exp(709.0)
            y_safe = max(abs(p.y), 1e-300)
            r = math.log(y_safe)
            a = scale_factor_a(t)
            denom = max(1.0 - k * r * r, 1e-300)
            return [[-1.0, 0.0], [0.0, a * a / denom]]
        return cls(_g, dim=2)

    @classmethod
    def calabi_yau_3(
        cls,
        kahler_potential: Optional[Callable[[EMLPoint], float]] = None,
    ) -> "MetricTensor":
        """
        Kähler metric on a Calabi–Yau 3-fold (2D projection).

        g_{ij} = ∂_i ∂_j̄ K  where K is the Kähler potential.
        Defaults to K(p) = exp(x)·ln(|y|) as the EML-native Kähler form.
        """
        if kahler_potential is None:
            def kahler_potential(p: EMLPoint) -> float:
                y_safe = max(abs(p.y), 1e-300)
                return math.exp(p.x) * math.log(y_safe)

        def _g(p: EMLPoint) -> List[List[float]]:
            h = 1e-5
            x, y = p.x, p.y
            y_safe = max(abs(y), 1e-300)

            def K(xi: float, yi: float) -> float:
                return kahler_potential(EMLPoint(xi, max(yi, 1e-300)))

            g00 = (K(x + h, y_safe) - 2 * K(x, y_safe) + K(x - h, y_safe)) / (h * h)
            g11 = (K(x, y_safe + h) - 2 * K(x, y_safe) + K(x, y_safe - h)) / (h * h)
            g01 = (K(x + h, y_safe + h) - K(x + h, y_safe - h)
                   - K(x - h, y_safe + h) + K(x - h, y_safe - h)) / (4 * h * h)
            return [[g00, g01], [g01, g11]]
        return cls(_g, dim=2)

    @classmethod
    def klebanov_strassler(cls, gsM: float = 0.1) -> "MetricTensor":
        """
        Klebanov–Strassler (KS) warped deformed conifold metric (2D slice).

        Warp factor h(τ) ≈ (gsM)^{4/3} / τ^{4/3} for large τ,
        where τ = exp(x) is the radial coordinate.
        g_{ij} = h(τ)^{-1/2} diag(1, 1) (string frame, large-τ approximation).
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            tau = max(math.exp(p.x) if p.x < 709.0 else math.exp(709.0), 1e-9)
            h_warp = (gsM ** (4.0 / 3.0)) / max(tau ** (4.0 / 3.0), 1e-300)
            w = max(h_warp ** (-0.5), 1e-300)
            return [[w, 0.0], [0.0, w]]
        return cls(_g, dim=2)

    @classmethod
    def heterotic_e8x8(cls, radius: float = 1.0) -> "MetricTensor":
        """
        Heterotic E₈×E₈ compactification metric (2D torus projection).

        g = (2πR)² · diag(1, 1) where R is the compactification radius.
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            scale = (2.0 * math.pi * radius) ** 2
            return [[scale, 0.0], [0.0, scale]]
        return cls(_g, dim=2)

    @classmethod
    def ads5_x_s5(cls, L: float = 1.0) -> "MetricTensor":
        """
        AdS₅ × S⁵ metric (2D slice in Poincaré coordinates).

        g_{tt} = (r/L)²,  g_{rr} = (L/r)²
        where r = exp(x) is the AdS radial coordinate.
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            r = max(math.exp(p.x) if p.x < 709.0 else math.exp(709.0), 1e-300)
            g00 = (r / L) ** 2
            g11 = (L / r) ** 2
            return [[g00, 0.0], [0.0, g11]]
        return cls(_g, dim=2)

    @classmethod
    def g2_holonomy(cls, octo_structure: Optional[object] = None) -> "MetricTensor":
        """
        G₂-holonomy metric (2D projection).

        For the Bryant–Salamon cone metric:
            ds² = dr² + r² g_{S⁶}
        mapped to EML: r = exp(x), g_{S⁶} contribution ≈ ln(y).
        The 2D slice: g_{rr} = 1, g_{SS} = exp(2x).
        """
        def _g(p: EMLPoint) -> List[List[float]]:
            r2 = math.exp(2.0 * p.x) if 2.0 * p.x < 709.0 else math.exp(709.0)
            return [[1.0, 0.0], [0.0, r2]]
        return cls(_g, dim=2)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"MetricTensor(dim={self._dim})"
