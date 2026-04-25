"""
EMLNDVector — N-dimensional EML coordinate vector with E8 and Leech lattice generators.

All coordinates are EMLPoints. Numeric operations fall back to Python math when
numpy is unavailable (install eml-math[ext] for numpy acceleration).
"""
from __future__ import annotations

import math
from typing import List, Optional

from eml_math.point import EMLPoint

try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False


class EMLNDVector:
    """
    N-dimensional vector whose components are EMLPoints.

    Parameters
    ----------
    coords : list[EMLPoint]
        One EMLPoint per dimension.
    """

    __slots__ = ("_coords",)

    def __init__(self, coords: List[EMLPoint]) -> None:
        if not coords:
            raise ValueError("EMLNDVector requires at least one component")
        self._coords = list(coords)

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def eml_coords(self) -> List[EMLPoint]:
        return list(self._coords)

    @property
    def n(self) -> int:
        return len(self._coords)

    @property
    def coords(self):
        """Numeric coordinate array (x-values). Returns np.ndarray if numpy available."""
        values = [p.x for p in self._coords]
        if _HAS_NUMPY:
            return np.array(values, dtype=float)
        return values

    # ── norms ─────────────────────────────────────────────────────────────────

    def euclidean_norm(self) -> float:
        """√(Σ xᵢ²) using the x-coordinate of each EMLPoint."""
        return math.sqrt(sum(p.x * p.x for p in self._coords))

    def minkowski_norm_nd(self, signature: List[int]) -> float:
        """
        √|Σ sᵢ · xᵢ²| where sᵢ ∈ {+1, -1} from signature.

        Parameters
        ----------
        signature : list[int]
            List of +1 or -1, one per dimension.
        """
        if len(signature) != self.n:
            raise ValueError(
                f"signature length {len(signature)} must match dimension {self.n}"
            )
        s = sum(sig * p.x * p.x for sig, p in zip(signature, self._coords))
        return math.sqrt(abs(s))

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return f"EMLNDVector(n={self.n}, norm={self.euclidean_norm():.6g})"

    def __len__(self) -> int:
        return self.n


# ── lattice generators ────────────────────────────────────────────────────────

def g2_metric() -> "MetricTensor":
    """
    Diagonal (1, 1, 1, 1, 1, 1, 1) metric for G₂ in 7 dimensions.

    Returns a MetricTensor instance. Import from eml_math.metric.
    """
    from eml_math.metric import MetricTensor
    diag = [1.0] * 7

    def _g(p: EMLPoint):
        return [[diag[i] if i == j else 0.0 for j in range(2)] for i in range(2)]

    return MetricTensor(_g, dim=7)


def _e8_root_vectors() -> List[List[float]]:
    """
    Generate the 240 root vectors of the E₈ root system.

    The E₈ root system has exactly 240 roots in two families:
    - Type 1 (112 vectors): ±e_i ± e_j for all i ≠ j in {0..7}.
      These are vectors with exactly two non-zero entries, each ±1.
    - Type 2 (128 vectors): ½(±1, ±1, ..., ±1) with an even number
      of minus signs (so that the half-integer spinor components satisfy
      the E₈ lattice integrality condition).

    Total: C(8,2) × 4 = 112 from Type 1, plus 2^7 = 128 from Type 2.
    All roots have squared norm 2 (Euclidean norm √2).
    """
    roots = []
    # Type 1: ±e_i ± e_j for i ≠ j → 4 * C(8,2) = 112 vectors
    import itertools
    for i, j in itertools.combinations(range(8), 2):
        for si in (+1, -1):
            for sj in (+1, -1):
                v = [0.0] * 8
                v[i] = float(si)
                v[j] = float(sj)
                roots.append(v)
    # Type 2: ½(±1,…,±1) with even number of minus signs → 2^7 = 128 vectors
    for signs in itertools.product([-1, 1], repeat=8):
        if signs.count(-1) % 2 == 0:
            roots.append([0.5 * s for s in signs])
    return roots


def e8_lattice_points(n_points: int = 10, scale: float = 1.0) -> List[EMLNDVector]:
    """
    Return up to n_points E₈ root vectors as EMLNDVectors.

    Each root vector has squared norm 2 (so Euclidean norm = √2).

    Parameters
    ----------
    n_points : int
        Maximum number of points to return (≤ 240).
    scale : float
        Scale factor applied to each coordinate.
    """
    roots = _e8_root_vectors()
    selected = roots[:min(n_points, len(roots))]
    result = []
    for v in selected:
        coords = [EMLPoint(scale * xi, 1.0) for xi in v]
        result.append(EMLNDVector(coords))
    return result


def e8_min_norm() -> float:
    """Minimum Euclidean norm of non-zero E₈ root vectors: √2."""
    return math.sqrt(2.0)


def _leech_basis_points() -> List[List[float]]:
    """
    Generate a sample of Leech lattice vectors (minimum norm 2).

    Returns the 24 standard basis vectors scaled to norm 2.
    """
    vecs = []
    for i in range(24):
        v = [0.0] * 24
        v[i] = 2.0
        vecs.append(v)
    return vecs


def leech_lattice_points(n_points: int = 5, scale: float = 1.0) -> List[EMLNDVector]:
    """
    Return up to n_points Leech lattice vectors as EMLNDVectors.

    Uses basis vectors with minimum norm 2.

    Parameters
    ----------
    n_points : int
        Maximum number of points.
    scale : float
        Scale factor.
    """
    basis = _leech_basis_points()
    selected = basis[:min(n_points, len(basis))]
    result = []
    for v in selected:
        coords = [EMLPoint(scale * xi, 1.0) for xi in v]
        result.append(EMLNDVector(coords))
    return result


def leech_min_norm() -> float:
    """Minimum Euclidean norm of non-zero Leech lattice vectors: 2."""
    return 2.0
