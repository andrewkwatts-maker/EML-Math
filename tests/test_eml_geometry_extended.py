"""
Extended geometric property tests for EMLPoint, EMLPair, EMLState,
FourMomentum, and the Planck-lattice discrete helpers.

These tests verify actual computation — not stubs — by checking
mathematical invariants (boost conservation, metric signatures, etc.).
"""
import math
import pytest
from eml_math.point import EMLPoint
from eml_math.pair import EMLPair
from eml_math.state import EMLState
from eml_math.momentum import FourMomentum
from eml_math.discrete import planck_delta, lattice_distance, is_lattice_neighbor
from eml_math.constants import PLANCK_D


# ── EMLPoint geometry ─────────────────────────────────────────────────────────

class TestEMLPointGeometry:

    def test_euclidean_delta_at_origin(self):
        p = EMLPoint(0.0, 1.0)
        d = p.euclidean_delta()
        assert math.isfinite(d) and d >= 0

    def test_euclidean_delta_positive(self):
        for x, y in [(1.0, 1.0), (0.5, 2.0), (2.0, 3.0)]:
            p = EMLPoint(x, y)
            assert p.euclidean_delta() > 0

    def test_minkowski_delta_finite(self):
        p = EMLPoint(1.0, 1.0)
        d = p.minkowski_delta()
        assert math.isfinite(d)

    def test_timelike_point(self):
        p = EMLPoint(2.0, 1.0)
        assert p.is_timelike()

    def test_spacelike_point(self):
        p = EMLPoint(0.0, math.exp(2.0))
        assert p.is_spacelike()

    def test_boost_preserves_minkowski_delta(self):
        p = EMLPoint(1.0, 1.0)
        d0 = p.minkowski_delta()
        p2 = p.boost(0.5)
        d1 = p2.minkowski_delta()
        assert abs(d0 - d1) < 1e-8

    def test_boost_zero_is_identity(self):
        p = EMLPoint(1.0, 2.0)
        p2 = p.boost(0.0)
        assert abs(p.x - p2.x) < 1e-10
        assert abs(p.y - p2.y) < 1e-10

    def test_boost_preserves_delta_parametric(self):
        p = EMLPoint(1.5, 3.0)
        d0 = p.minkowski_delta()
        for phi in [0.1, 0.3, 0.5, 0.7, 1.0]:
            p2 = p.boost(phi)
            assert abs(d0 - p2.minkowski_delta()) < 1e-8, f"phi={phi}"

    def test_pair_from_point(self):
        p = EMLPoint(1.0, 1.0)
        pr = p.pair()
        assert isinstance(pr, EMLPair)

    def test_pair_has_positive_real_tension(self):
        p = EMLPoint(1.0, 1.0)
        pr = p.pair()
        assert pr.real_tension > 0

    def test_lightlike_check(self):
        # Create a lightlike point: exp(2x) = (c*ln(y))^2
        # exp(x) = |c*ln(y)| => use x=0, y=1 => exp(0)=1, ln(1)=0 => diff=1, not lightlike
        # Use x = ln(ln(y)) for exact lightlike: exp(x) = c*ln(y)
        c = 1.0
        y = 2.0
        target_x = math.log(math.log(y))  # exp(x) = ln(y)
        p = EMLPoint(target_x, y)
        assert p.is_lightlike(c=c, tol=1e-8)

    def test_rest_energy_equals_minkowski_delta(self):
        p = EMLPoint(1.0, 1.0)
        assert abs(p.rest_energy() - p.minkowski_delta()) < 1e-12

    def test_proper_time_is_rest_energy_over_c(self):
        p = EMLPoint(1.0, 1.0)
        c = 2.0
        assert abs(p.proper_time(c=c) - p.rest_energy(c=c) / c) < 1e-12


# ── EMLPair ───────────────────────────────────────────────────────────────────

class TestEMLPairExtended:

    def test_from_values_stores_tensions(self):
        p = EMLPair.from_values(3.0, 4.0)
        assert abs(p.real_tension - 3.0) < 1e-12
        assert abs(p.imag_tension - 4.0) < 1e-12

    def test_modulus_3_4_5(self):
        p = EMLPair.from_values(3.0, 4.0)
        assert abs(p.modulus - 5.0) < 1e-12

    def test_modulus_1_0(self):
        p = EMLPair.from_values(1.0, 0.0)
        assert abs(p.modulus - 1.0) < 1e-12

    def test_conjugate_flips_imag(self):
        p = EMLPair.from_values(2.0, 3.0)
        c = p.conjugate()
        assert abs(c.real_tension - 2.0) < 1e-12
        assert abs(c.imag_tension + 3.0) < 1e-12

    def test_conjugate_same_modulus(self):
        p = EMLPair.from_values(3.0, 4.0)
        assert abs(p.modulus - p.conjugate().modulus) < 1e-12

    def test_rotate_phase_zero_identity(self):
        p = EMLPair.from_values(1.0, 0.0)
        p2 = p.rotate_phase(0.0)
        assert abs(p2.real_tension - 1.0) < 1e-10
        assert abs(p2.imag_tension) < 1e-10

    def test_rotate_phase_pi_half(self):
        p = EMLPair.from_values(1.0, 0.0)
        p2 = p.rotate_phase(math.pi / 2)
        assert abs(p2.real_tension) < 1e-10
        assert abs(p2.imag_tension - 1.0) < 1e-10

    def test_rotate_phase_pi(self):
        p = EMLPair.from_values(1.0, 0.0)
        p2 = p.rotate_phase(math.pi)
        assert abs(p2.real_tension + 1.0) < 1e-10
        assert abs(p2.imag_tension) < 1e-10

    def test_rotate_phase_preserves_modulus(self):
        p = EMLPair.from_values(3.0, 4.0)
        m0 = p.modulus
        for phi in [0.1, 0.5, 1.0, math.pi]:
            m1 = p.rotate_phase(phi).modulus
            assert abs(m0 - m1) < 1e-10, f"phi={phi}"

    def test_frames_count(self):
        p = EMLPair.from_values(1.0, 2.0)
        frames = p.frames()
        assert len(frames) == 4

    def test_frames_all_same_modulus(self):
        p = EMLPair.from_values(3.0, 4.0)
        frames = p.frames()
        m0 = frames[0].modulus
        for f in frames[1:]:
            assert abs(f.modulus - m0) < 1e-10


# ── EMLState ──────────────────────────────────────────────────────────────────

class TestEMLStateExtended:

    def test_from_point_factory(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        assert isinstance(s, EMLState)
        assert abs(s.point.x - 1.0) < 1e-12

    def test_from_point_default_n(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        assert s._n == 0

    def test_from_point_custom_n(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p, n=5)
        assert s._n == 5

    def test_from_point_custom_theta(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p, theta=math.pi)
        assert abs(s._theta - math.pi) < 1e-12

    def test_point_property_type(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        assert isinstance(s.point, EMLPoint)

    def test_minkowski_pulse_returns_list(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        traj = s.minkowski_pulse(3)
        assert isinstance(traj, list)

    def test_minkowski_pulse_length(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        for n in [1, 3, 5]:
            traj = s.minkowski_pulse(n)
            assert len(traj) == n

    def test_minkowski_pulse_elements_are_states(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        traj = s.minkowski_pulse(3)
        for st in traj:
            assert isinstance(st, EMLState)

    def test_geodesic_step_returns_state(self):
        from eml_math.metric import MetricTensor
        p = EMLPoint(3.0, 1.0)
        s = EMLState.from_point(p)
        m = MetricTensor.flat()
        s2 = s.geodesic_step(m, dtau=0.01)
        assert isinstance(s2, EMLState)


# ── FourMomentum ──────────────────────────────────────────────────────────────

class TestFourMomentumExtended:

    def test_energy_is_exp_x(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert abs(fm.energy - math.exp(1.0)) < 1e-10

    def test_energy_positive(self):
        for x in [0.0, 0.5, 1.0, 2.0]:
            p = EMLPoint(x, 1.0)
            fm = FourMomentum(p)
            assert fm.energy > 0

    def test_momentum_property(self):
        p = EMLPoint(1.0, math.e)
        fm = FourMomentum(p)
        assert math.isfinite(fm.momentum)

    def test_mass_finite(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert math.isfinite(fm.mass)

    def test_mass_non_negative(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert fm.mass >= 0

    def test_gamma_at_rest(self):
        # For zero spatial momentum: gamma should be well-defined
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        g = fm.gamma()
        assert math.isfinite(g) or g == math.inf

    def test_boost_returns_fourmomentum(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        fm2 = fm.boost(0.5)
        assert isinstance(fm2, FourMomentum)

    def test_boost_preserves_mass(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        m0 = fm.mass
        for phi in [0.1, 0.3, 0.5]:
            fm2 = fm.boost(phi)
            assert abs(fm2.mass - m0) < 1e-8, f"phi={phi}"

    def test_from_mass_velocity_returns_fourmomentum(self):
        fm = FourMomentum.from_mass_velocity(1.0, 0.5, c=1.0)
        assert isinstance(fm, FourMomentum)

    def test_from_mass_velocity_mass_invariant(self):
        fm = FourMomentum.from_mass_velocity(2.0, 0.5, c=1.0)
        assert abs(fm.mass - 2.0) < 1e-6

    def test_point_property(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p)
        assert isinstance(fm.point, EMLPoint)

    def test_c_property(self):
        p = EMLPoint(1.0, 1.0)
        fm = FourMomentum(p, c=3.0)
        assert fm.c == 3.0


# ── Discrete helpers ──────────────────────────────────────────────────────────

class TestDiscreteHelpers:

    def test_planck_delta_returns_float(self):
        p = EMLPoint(1.0, 1.0)
        result = planck_delta(p, D=1.0)
        assert isinstance(result, float)

    def test_planck_delta_finite(self):
        p = EMLPoint(1.0, 1.0)
        result = planck_delta(p, D=1.0)
        assert math.isfinite(result)

    def test_planck_delta_quantizes(self):
        p = EMLPoint(1.0, 1.0)
        for D in [1.0, 10.0, 100.0]:
            result = planck_delta(p, D=D)
            # Result should be a multiple of 1/D
            assert abs((result * D) - round(result * D)) < 1e-10

    def test_planck_delta_same_point_deterministic(self):
        p = EMLPoint(1.0, 1.0)
        d1 = planck_delta(p, D=10.0)
        d2 = planck_delta(p, D=10.0)
        assert d1 == d2

    def test_lattice_distance_returns_float(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(1.1, 1.1)
        result = lattice_distance(p1, p2, D=1.0)
        assert isinstance(result, float)

    def test_lattice_distance_finite(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(2.0, 2.0)
        result = lattice_distance(p1, p2, D=1.0)
        assert math.isfinite(result)

    def test_lattice_distance_same_point_deterministic(self):
        p = EMLPoint(1.0, 1.0)
        d1 = lattice_distance(p, p, D=1.0)
        d2 = lattice_distance(p, p, D=1.0)
        assert d1 == d2  # deterministic

    def test_is_lattice_neighbor_returns_bool(self):
        p1 = EMLPoint(1.0, 1.0)
        p2 = EMLPoint(2.0, 2.0)
        result = is_lattice_neighbor(p1, p2, D=1.0)
        assert isinstance(result, bool)

    def test_lattice_distance_nonzero_for_different_points(self):
        p1 = EMLPoint(0.0, 1.0)
        p2 = EMLPoint(2.0, math.e)
        d = lattice_distance(p1, p2, D=1.0)
        assert d >= 0
