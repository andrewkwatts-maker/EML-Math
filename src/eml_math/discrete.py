"""
Discrete / Planck-scale quantization helpers for EMLPoint coordinates.

Provides a quantization grid at scale D and lattice-adjacency detection.

Note: the "Planck lattice" here is a computational discretization whose
grid spacing is set to the Planck length (≈ 1.616 × 10⁻³⁵ m) via the
constant PLANCK_D = 1/l_planck. This is a numerical convenience — it is
not derived from a physical theory of quantum gravity and should not be
interpreted as a physically fundamental lattice structure.
"""
from __future__ import annotations

import math

from eml_math.point import EMLPoint
from eml_math.constants import PLANCK_D


def planck_delta(point: EMLPoint, D: float = PLANCK_D) -> float:
    """
    Quantize the Minkowski delta of ``point`` to the Planck lattice at scale D.

    Returns round(Δ_M × D) / D — the nearest representable grid value.

    Parameters
    ----------
    point : EMLPoint
    D : float
        Quantization scale. Default is PLANCK_D from constants.
    """
    delta = point.minkowski_delta()
    return round(delta * D) / D


def lattice_distance(p1: EMLPoint, p2: EMLPoint, D: float = PLANCK_D) -> float:
    """
    Planck-quantized distance between two EMLPoints in the EML coordinate frame.

    Computes the Minkowski delta of the displacement point
        EMLPoint(p2.x - p1.x, p2.y / p1.y)
    and quantizes the result at scale D.

    Parameters
    ----------
    p1, p2 : EMLPoint
    D : float
        Quantization scale.
    """
    dx = p2.x - p1.x
    y1_safe = max(abs(p1.y), 1e-300)
    y2_safe = max(abs(p2.y), 1e-300)
    dy_ratio = y2_safe / y1_safe
    displacement = EMLPoint(dx, max(dy_ratio, 1e-300))
    return planck_delta(displacement, D=D)


def is_lattice_neighbor(
    p1: EMLPoint,
    p2: EMLPoint,
    D: float = PLANCK_D,
) -> bool:
    """
    Return True if p1 and p2 are adjacent on the Planck lattice.

    Two points are neighbors when their lattice_distance is within ½ a grid
    step of the minimal cell size 1/D:

        |lattice_distance(p1, p2) - 1/D| < 0.5/D

    Parameters
    ----------
    p1, p2 : EMLPoint
    D : float
        Quantization scale.
    """
    dist = lattice_distance(p1, p2, D=D)
    cell = 1.0 / D
    return abs(dist - cell) < 0.5 * cell
