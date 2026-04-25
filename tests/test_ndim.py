"""Tests for Sprint 5 EMLNDVector and lattice functions."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.ndim import (
    EMLNDVector,
    e8_lattice_points,
    e8_min_norm,
    leech_lattice_points,
    leech_min_norm,
    g2_metric,
)


class TestEMLNDVector:
    def test_basic_construction(self):
        coords = [EMLPoint(1.0, 1.0), EMLPoint(2.0, 1.0)]
        v = EMLNDVector(coords)
        assert v.n == 2

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            EMLNDVector([])

    def test_euclidean_norm_unit(self):
        # EMLPoint(1,1) has x=1; two such → √(1+1)=√2
        coords = [EMLPoint(1.0, 1.0), EMLPoint(1.0, 1.0)]
        v = EMLNDVector(coords)
        assert abs(v.euclidean_norm() - math.sqrt(2.0)) < 1e-9

    def test_euclidean_norm_zero(self):
        coords = [EMLPoint(0.0, 1.0)] * 4
        v = EMLNDVector(coords)
        assert abs(v.euclidean_norm()) < 1e-9

    def test_minkowski_norm_nd(self):
        # (+,+,-,-) signature, all x=1 → |1+1-1-1|=0
        coords = [EMLPoint(1.0, 1.0)] * 4
        v = EMLNDVector(coords)
        assert abs(v.minkowski_norm_nd([1, 1, -1, -1])) < 1e-9

    def test_minkowski_norm_nd_wrong_length_raises(self):
        coords = [EMLPoint(1.0, 1.0)] * 3
        v = EMLNDVector(coords)
        with pytest.raises(ValueError):
            v.minkowski_norm_nd([1, 1])

    def test_repr(self):
        v = EMLNDVector([EMLPoint(1.0, 1.0)])
        assert "EMLNDVector" in repr(v)

    def test_len(self):
        coords = [EMLPoint(float(i), 1.0) for i in range(5)]
        v = EMLNDVector(coords)
        assert len(v) == 5


class TestE8Lattice:
    def test_count(self):
        points = e8_lattice_points(n_points=10)
        assert len(points) == 10

    def test_all_eml_nd_vector(self):
        for p in e8_lattice_points(n_points=5):
            assert isinstance(p, EMLNDVector)
            assert p.n == 8

    def test_min_norm(self):
        assert abs(e8_min_norm() - math.sqrt(2.0)) < 1e-9

    def test_root_vector_norm_sqrt2(self):
        points = e8_lattice_points(n_points=1)
        v = points[0]
        # Euclidean norm of ±e_i ± e_j root: √(1+1)=√2
        assert abs(v.euclidean_norm() - math.sqrt(2.0)) < 1e-9

    def test_scale_applies(self):
        points_1 = e8_lattice_points(n_points=1, scale=1.0)
        points_2 = e8_lattice_points(n_points=1, scale=2.0)
        norm1 = points_1[0].euclidean_norm()
        norm2 = points_2[0].euclidean_norm()
        assert abs(norm2 - 2.0 * norm1) < 1e-9

    def test_max_240(self):
        points = e8_lattice_points(n_points=300)
        assert len(points) == 240  # exactly 240 E8 roots


class TestLeechLattice:
    def test_count(self):
        points = leech_lattice_points(n_points=5)
        assert len(points) == 5

    def test_all_eml_nd_vector(self):
        for p in leech_lattice_points(n_points=3):
            assert isinstance(p, EMLNDVector)
            assert p.n == 24

    def test_min_norm(self):
        assert abs(leech_min_norm() - 2.0) < 1e-9

    def test_basis_vector_norm(self):
        points = leech_lattice_points(n_points=1)
        v = points[0]
        # Basis vector (2, 0, ..., 0) → norm = 2
        assert abs(v.euclidean_norm() - 2.0) < 1e-9


# ── new expanded tests ────────────────────────────────────────────────────────

class TestE8RootCount:
    def test_exactly_240_roots(self):
        points = e8_lattice_points(n_points=240)
        assert len(points) == 240

    def test_request_300_returns_240(self):
        points = e8_lattice_points(n_points=300)
        assert len(points) == 240


class TestE8MinNorm:
    def test_all_240_roots_have_norm_sqrt2(self):
        points = e8_lattice_points(n_points=240)
        sqrt2 = math.sqrt(2.0)
        for i, v in enumerate(points):
            assert abs(v.euclidean_norm() - sqrt2) < 1e-10, \
                f"E8 root {i} has norm {v.euclidean_norm():.12g} != √2"

    def test_min_norm_function_returns_sqrt2(self):
        from eml_math.ndim import e8_min_norm
        assert abs(e8_min_norm() - math.sqrt(2.0)) < 1e-10


class TestE8OrthogonalityRange:
    # Fixed pairs of root indices to check integer inner products
    _INDEX_PAIRS = [(0, 1), (0, 2), (1, 5), (10, 20), (50, 100), (113, 200)]

    @pytest.mark.parametrize("i,j", _INDEX_PAIRS)
    def test_integer_inner_product(self, i, j):
        points = e8_lattice_points(n_points=240)
        vi = points[i]
        vj = points[j]
        inner = sum(a * b for a, b in zip(vi.coords, vj.coords))
        # E8 inner products are always integers: 0, ±1, ±2
        assert abs(inner - round(inner)) < 1e-9, \
            f"E8 roots {i},{j}: inner product {inner:.12g} is not integer"


class TestNDVectorMinkowskiNorm:
    def test_mixed_signature_nonzero(self):
        # (+,+,-,-): (1,1,1,1) → |1+1-1-1| = 0
        coords = [EMLPoint(1.0, 1.0)] * 4
        v = EMLNDVector(coords)
        assert abs(v.minkowski_norm_nd([1, 1, -1, -1])) < 1e-9

    def test_all_positive_signature(self):
        coords = [EMLPoint(3.0, 1.0), EMLPoint(4.0, 1.0)]
        v = EMLNDVector(coords)
        # (1,1): 3² + 4² = 25 → norm = 5
        assert abs(v.minkowski_norm_nd([1, 1]) - 5.0) < 1e-9

    def test_lorentzian_timelike(self):
        # (+,-,-,-): t²-x²-y²-z² with t=5, x=y=z=0 → 5
        coords = [EMLPoint(5.0, 1.0)] + [EMLPoint(0.0, 1.0)] * 3
        v = EMLNDVector(coords)
        assert abs(v.minkowski_norm_nd([1, -1, -1, -1]) - 5.0) < 1e-9

    def test_lorentzian_spacelike(self):
        # (+,-,-,-): t=0, x=3, y=4 → -(9+16)=-25 → |.| = 5
        coords = [EMLPoint(0.0, 1.0), EMLPoint(3.0, 1.0), EMLPoint(4.0, 1.0), EMLPoint(0.0, 1.0)]
        v = EMLNDVector(coords)
        assert abs(v.minkowski_norm_nd([1, -1, -1, -1]) - 5.0) < 1e-9
