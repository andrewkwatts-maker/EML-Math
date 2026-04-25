"""Tests for Sprint 3 FourMomentum class."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.momentum import FourMomentum


class TestFourMomentumBasic:
    def test_energy_is_exp_x(self):
        p = EMLPoint(math.log(5.0), 1.0)
        fm = FourMomentum(p)
        assert abs(fm.energy - 5.0) < 1e-9

    def test_momentum_is_ln_y_over_c(self):
        # y = e^2 → ln(y) = 2 → p = 2/1 = 2
        p = EMLPoint(0.0, math.exp(2.0))
        fm = FourMomentum(p)
        assert abs(fm.momentum - 2.0) < 1e-9

    def test_mass_invariance_under_boost(self):
        p = EMLPoint(1.0, math.e)
        fm = FourMomentum(p)
        m0 = fm.mass
        for phi in [0.1, 0.5, 1.0, -0.3]:
            fm_b = fm.boost(phi)
            assert abs(fm_b.mass - m0) < 1e-9, f"mass changed at phi={phi}"

    def test_gamma_at_rest(self):
        # v=0 → gamma=1 → E = mc² (c=1) → E = m
        p = EMLPoint(1.0, math.e)
        fm = FourMomentum.from_mass_velocity(mass=2.0, v=0.0)
        # from_mass_velocity at v=0: gamma=1, E=m*c^2=2, p=0
        assert abs(fm.energy - 2.0) < 1e-9

    def test_gamma_massless_is_inf(self):
        # Lightlike point: minkowski_delta ≈ 0
        p = EMLPoint(0.0, 1.0)  # E=1, p=0 → m=1 (not massless for this coord)
        # Force a "massless" by making delta=0: x=0, y=e → E=1, p·c=1 → lightlike
        p2 = EMLPoint(0.0, math.e)  # E=exp(0)=1, s=c*ln(e)=1, delta=|1-1|=0
        fm = FourMomentum(p2)
        g = fm.gamma()
        assert math.isinf(g) or g > 1e10

    def test_from_mass_velocity_superluminal_raises(self):
        with pytest.raises(ValueError, match="velocity"):
            FourMomentum.from_mass_velocity(mass=1.0, v=1.0, c=1.0)

    def test_from_mass_velocity_negative_mass_raises(self):
        with pytest.raises(ValueError, match="mass"):
            FourMomentum.from_mass_velocity(mass=-1.0, v=0.5)

    def test_boost_zero_phi_identity(self):
        p = EMLPoint(1.0, math.e)
        fm = FourMomentum(p)
        fm_b = fm.boost(0.0)
        assert abs(fm_b.energy - fm.energy) < 1e-9
        assert abs(fm_b.momentum - fm.momentum) < 1e-9

    def test_repr(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert "FourMomentum" in repr(fm)

    def test_from_mass_velocity_energy_recovery(self):
        # At v=0: E = m*c^2
        fm = FourMomentum.from_mass_velocity(mass=3.0, v=0.0, c=2.0)
        assert abs(fm.energy - 3.0 * 4.0) < 1e-9  # m * c^2 = 3 * 4


# ── new expanded tests ────────────────────────────────────────────────────────

class TestMassInvarianceUnderBoost:
    @pytest.mark.parametrize("mass", [0.1, 1.0, 10.0])
    @pytest.mark.parametrize("phi", [-1.0, 0.0, 1.0, 2.0])
    def test_mass_conserved(self, mass, phi):
        # Construct FourMomentum at rest, then boost
        fm = FourMomentum.from_mass_velocity(mass=mass, v=0.0, c=1.0)
        m0 = fm.mass
        fm_b = fm.boost(phi)
        assert abs(fm_b.mass - m0) < 1e-8, \
            f"mass changed: mass={mass}, phi={phi}: {fm_b.mass} vs {m0}"


class TestGammaFactor:
    def test_gamma_at_rest_is_one(self):
        fm = FourMomentum.from_mass_velocity(mass=1.0, v=0.0, c=1.0)
        assert abs(fm.gamma() - 1.0) < 1e-9

    def test_gamma_increases_with_velocity(self):
        c = 1.0
        g_slow = FourMomentum.from_mass_velocity(mass=1.0, v=0.1 * c, c=c).gamma()
        g_fast = FourMomentum.from_mass_velocity(mass=1.0, v=0.9 * c, c=c).gamma()
        assert g_fast > g_slow

    def test_gamma_equals_E_over_mc2(self):
        c = 1.0
        mass = 2.0
        v = 0.6 * c
        fm = FourMomentum.from_mass_velocity(mass=mass, v=v, c=c)
        gamma_direct = fm.gamma()
        gamma_formula = fm.energy / (mass * c * c)
        assert abs(gamma_direct - gamma_formula) < 1e-8

    def test_gamma_lightlike_is_inf(self):
        # Lightlike point: mass ≈ 0 → gamma = inf
        p = EMLPoint(0.0, math.e)  # E=1, p·c=1, mass=0
        fm = FourMomentum(p)
        g = fm.gamma()
        assert math.isinf(g) or g > 1e10


class TestEnergyMomentumRelation:
    @pytest.mark.parametrize("mass,v", [
        (1.0, 0.0),
        (1.0, 0.5),
        (2.0, 0.8),
        (0.5, 0.3),
        (10.0, 0.1),
    ])
    def test_e2_minus_pc2_equals_mc2_sq(self, mass, v):
        c = 1.0
        fm = FourMomentum.from_mass_velocity(mass=mass, v=v, c=c)
        E = fm.energy
        p = fm.momentum
        lhs = E * E - (p * c) ** 2
        rhs = (mass * c * c) ** 2
        assert abs(lhs - rhs) < 1e-8, \
            f"E²-(pc)²={lhs:.9g} != (mc²)²={rhs:.9g} for mass={mass}, v={v}"
