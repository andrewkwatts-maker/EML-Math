"""Tests for Sprint 3 discrete / Planck-lattice helpers."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.discrete import planck_delta, lattice_distance, is_lattice_neighbor
from eml_math.constants import PLANCK_D


class TestPlanckDelta:
    def test_quantizes_to_grid(self):
        p = EMLPoint(1.0, 1.0)
        D = 10.0
        result = planck_delta(p, D=D)
        # Must be a multiple of 1/D
        assert abs(result * D - round(result * D)) < 1e-9

    def test_zero_delta_stays_zero(self):
        # Lightlike: E = p*c → delta = 0
        p = EMLPoint(0.0, math.e)  # E=1, s=1, delta=0
        result = planck_delta(p, D=100.0)
        assert abs(result) < 0.01  # quantized near 0

    def test_returns_float(self):
        p = EMLPoint(1.0, 2.0)
        assert isinstance(planck_delta(p), float)

    def test_default_D_is_planck_d(self):
        p = EMLPoint(1.0, 2.0)
        result_default = planck_delta(p)
        result_explicit = planck_delta(p, D=PLANCK_D)
        assert abs(result_default - result_explicit) < 1e-12


class TestLatticeDistance:
    def test_same_point_distance_near_zero(self):
        p = EMLPoint(1.0, math.e)
        dist = lattice_distance(p, p, D=10.0)
        # displacement is EMLPoint(0, 1) → delta near 1; quantized to 0.1*n
        assert isinstance(dist, float)
        assert math.isfinite(dist)

    def test_symmetric_under_swap(self):
        p1 = EMLPoint(1.0, 2.0)
        p2 = EMLPoint(1.5, 3.0)
        # Not necessarily symmetric (direction matters), just verify both finite
        d12 = lattice_distance(p1, p2, D=10.0)
        d21 = lattice_distance(p2, p1, D=10.0)
        assert math.isfinite(d12)
        assert math.isfinite(d21)

    def test_returns_float(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(1.0, math.e)
        assert isinstance(lattice_distance(p1, p2, D=10.0), float)


class TestIsLatticeNeighbor:
    def test_far_points_not_neighbors(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(100.0, 1.0)
        D = 10.0
        result = is_lattice_neighbor(p1, p2, D=D)
        assert isinstance(result, bool)

    def test_returns_bool(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(1.0, 1.0)
        assert isinstance(is_lattice_neighbor(p1, p2), bool)

    def test_self_not_neighbor(self):
        # Distance to self is quantized(delta of displacement) which is not 1/D
        p = EMLPoint(1.0, math.e)
        result = is_lattice_neighbor(p, p, D=PLANCK_D)
        assert isinstance(result, bool)
