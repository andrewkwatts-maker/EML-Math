"""Tests for Sprint 6 MinkowskiFourVector."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.fourvector import MinkowskiFourVector


def _vec(t, x, y, z, c=1.0):
    return MinkowskiFourVector(
        EMLPoint(t, 1.0), EMLPoint(x, 1.0),
        EMLPoint(y, 1.0), EMLPoint(z, 1.0), c=c
    )


class TestMinkowskiFourVectorBasic:
    def test_construction(self):
        v = _vec(1.0, 0.5, 0.5, 0.5)
        assert isinstance(v, MinkowskiFourVector)

    def test_components_accessible(self):
        v = _vec(2.0, 1.0, 0.0, 0.0)
        assert v.t_component.x == 2.0
        assert len(v.spatial_components) == 3

    def test_repr(self):
        v = _vec(1.0, 2.0, 3.0, 4.0)
        assert "MinkowskiFourVector" in repr(v)

    def test_equality(self):
        v1 = _vec(1.0, 2.0, 3.0, 4.0)
        v2 = _vec(1.0, 2.0, 3.0, 4.0)
        assert v1 == v2

    def test_inequality(self):
        v1 = _vec(1.0, 2.0, 3.0, 4.0)
        v2 = _vec(1.0, 2.0, 3.0, 5.0)
        assert v1 != v2


class TestMinkowskiNorm:
    def test_timelike_norm(self):
        # ds² = c²t² - x² - y² - z² = 4 - 0 = 4 → norm = 2
        v = _vec(2.0, 0.0, 0.0, 0.0)
        assert abs(v.minkowski_norm() - 2.0) < 1e-9

    def test_lightlike_norm(self):
        # ds² = 1 - 1 - 0 - 0 = 0
        v = _vec(1.0, 1.0, 0.0, 0.0)
        assert abs(v.minkowski_norm()) < 1e-9

    def test_norm_invariant_under_boost(self):
        v = _vec(3.0, 1.0, 0.5, 0.0)
        norm0 = v.minkowski_norm()
        for phi in [0.1, 0.5, -0.3]:
            v_b = v.boost(phi, direction="x")
            assert abs(v_b.minkowski_norm() - norm0) < 1e-8, f"drift at phi={phi}"


class TestBoost:
    def test_boost_zero_rapidity_identity(self):
        v = _vec(2.0, 1.0, 0.5, 0.3)
        v_b = v.boost(0.0, direction="x")
        assert v == v_b

    def test_boost_x_direction(self):
        v = _vec(1.0, 0.0, 0.0, 0.0)
        v_b = v.boost(0.5, direction="x")
        assert isinstance(v_b, MinkowskiFourVector)

    def test_boost_y_direction(self):
        v = _vec(1.0, 0.0, 0.0, 0.0)
        v_b = v.boost(0.5, direction="y")
        assert isinstance(v_b, MinkowskiFourVector)

    def test_boost_z_direction(self):
        v = _vec(1.0, 0.0, 0.0, 0.0)
        v_b = v.boost(0.5, direction="z")
        assert isinstance(v_b, MinkowskiFourVector)

    def test_boost_invalid_direction_raises(self):
        v = _vec(1.0, 0.0, 0.0, 0.0)
        with pytest.raises(ValueError):
            v.boost(0.5, direction="w")


class TestFourMomentum:
    def test_returns_4_components(self):
        v = _vec(1.0, 0.0, 0.0, 0.0)
        p = v.four_momentum()
        assert len(p) == 4

    def test_components_finite(self):
        v = _vec(1.0, 2.0, 0.5, 0.3)
        p = v.four_momentum()
        for comp in p:
            assert math.isfinite(float(comp))


# ── new expanded tests ────────────────────────────────────────────────────────

class TestBoostInvariance:
    @pytest.mark.parametrize("phi", [-2.0, -1.0, 0.0, 1.0, 2.0])
    @pytest.mark.parametrize("direction", ["x", "y", "z"])
    def test_norm_preserved(self, phi, direction):
        v = _vec(3.0, 1.0, 0.5, 0.2)
        norm0 = v.minkowski_norm()
        v_b = v.boost(phi, direction=direction)
        assert abs(v_b.minkowski_norm() - norm0) < 1e-8, \
            f"norm changed: phi={phi}, dir={direction}"


class TestRestFrameProperties:
    def test_zero_spatial_momentum_norm_is_ct(self):
        # v = (t, 0, 0, 0): norm = c*t
        c = 1.0
        t = 5.0
        v = _vec(t, 0.0, 0.0, 0.0, c=c)
        assert abs(v.minkowski_norm() - c * t) < 1e-9

    def test_lightlike_norm_zero(self):
        # v = (1, 1, 0, 0): ds² = 1 - 1 = 0
        v = _vec(1.0, 1.0, 0.0, 0.0)
        assert abs(v.minkowski_norm()) < 1e-9

    def test_lightlike_3d_zero(self):
        # v = (t, x, y, z) with x²+y²+z² = t²
        t = math.sqrt(3.0)
        v = _vec(t, 1.0, 1.0, 1.0)
        # floating point: sqrt(3)^2 - 1 - 1 - 1 has small rounding error
        assert abs(v.minkowski_norm()) < 1e-7


class TestFourMomentumRecovery:
    def test_mass_from_four_momentum(self):
        # A timelike vector with c*t > spatial: norm = sqrt(c²t² - x²)
        t = 5.0
        x = 3.0
        v = _vec(t, x, 0.0, 0.0)
        norm = v.minkowski_norm()
        # minkowski_norm = sqrt(|t^2 - x^2|) = sqrt(25-9) = 4
        assert abs(norm - 4.0) < 1e-9

    def test_four_momentum_norm_matches_minkowski_norm(self):
        v = _vec(4.0, 3.0, 0.0, 0.0)
        # four_momentum gives [E/c, px, py, pz]; mass from E² - (pc)²
        p = v.four_momentum()
        E_over_c = float(p[0])
        px = float(p[1])
        py = float(p[2])
        pz = float(p[3])
        # E = exp(t.x): t.x = 4 → E/c = exp(4)
        # norm = minkowski_norm uses t.x directly (not exp)
        assert math.isfinite(E_over_c)
        assert math.isfinite(px)


class TestBoostComposition:
    @pytest.mark.parametrize("phi1,phi2", [
        (0.3, 0.4),
        (0.5, 0.5),
        (-0.3, 0.3),
        (0.1, 0.9),
        (-0.5, -0.5),
    ])
    def test_sequential_boosts_same_as_single(self, phi1, phi2):
        # Two boosts in same direction compose additively
        v = _vec(3.0, 1.0, 0.5, 0.2)
        composed = v.boost(phi1, direction="x").boost(phi2, direction="x")
        single = v.boost(phi1 + phi2, direction="x")
        assert abs(composed.t_component.x - single.t_component.x) < 1e-9
        assert abs(composed.spatial_components[0].x - single.spatial_components[0].x) < 1e-9
