"""Tests for TensionPoint — the universal EML computation node."""
import math
import pytest
from eml_math import EMLPoint
from eml_math.constants import OVERFLOW_THRESHOLD


class TestEMLPrimitive:
    """EMLPoint(x, y).tension() == eml(x, y) = exp(x) - ln(y)."""

    def test_unit_point_gives_e(self):
        assert EMLPoint(1.0, 1.0).tension() == pytest.approx(math.e, rel=1e-14)

    def test_exp_x_is_eml_x_1(self):
        for x in [0.0, 0.5, 1.0, 2.0, 3.0]:
            assert EMLPoint(x, 1.0).tension() == pytest.approx(math.exp(x), rel=1e-14)

    def test_formula_matches_manual(self):
        x, y = 2.0, 3.0
        expected = math.exp(x) - math.log(y)
        assert EMLPoint(x, y).tension() == pytest.approx(expected, rel=1e-14)

    def test_tension_is_always_real(self):
        for x in [0.1, 1.0, 5.0]:
            for y in [0.1, 1.0, 5.0]:
                T = EMLPoint(x, y).tension()
                assert math.isfinite(T)

    def test_frame_shift_guard_when_y_negative(self):
        # y < 0 would make ln(y) undefined; frame guard uses |y|
        p = EMLPoint(1.0, -2.0)
        T = p.tension()
        expected = math.exp(1.0) - math.log(2.0)
        assert T == pytest.approx(expected, rel=1e-12)

    def test_frame_shift_guard_when_y_zero(self):
        p = EMLPoint(1.0, 0.0)
        T = p.tension()
        assert math.isfinite(T)


class TestNestedEML:
    """TensionPoint accepts other TensionPoints as coordinates."""

    def test_ln_nested_knot(self):
        # ln(e) = 1 via depth-3 EML nesting
        e = math.e
        result = EMLPoint(1.0, EMLPoint(EMLPoint(1.0, e), 1.0)).tension()
        assert result == pytest.approx(1.0, rel=1e-10)

    def test_ln_two(self):
        inner1 = EMLPoint(1.0, 2.0)
        inner2 = EMLPoint(inner1, 1.0)
        result = EMLPoint(1.0, inner2).tension()
        assert result == pytest.approx(math.log(2.0), rel=1e-10)

    def test_double_nesting(self):
        # exp(exp(1)) = e^e via nesting
        inner = EMLPoint(1.0, 1.0)          # tension = e
        outer = EMLPoint(inner, 1.0)         # tension = exp(e) - ln(1) = exp(e)
        assert outer.tension() == pytest.approx(math.exp(math.e), rel=1e-10)

    def test_x_coord_evaluates_nested(self):
        nested = EMLPoint(2.0, 1.0)          # tension = exp(2)
        p = EMLPoint(nested, 1.0)            # x = nested.tension() = exp(2)
        assert p.x == pytest.approx(math.exp(2.0), rel=1e-14)


class TestMirrorPulse:
    """mirror_pulse() — continuous mode."""

    def test_standard_update(self, unit_point):
        # Continuous: x_new = y, y_new = T
        y_old = unit_point.y
        T = unit_point.tension()
        nxt = unit_point.mirror_pulse()
        assert nxt.x == pytest.approx(y_old, rel=1e-12)
        assert nxt.y == pytest.approx(T, rel=1e-12)

    def test_frame_shift_on_negative_y(self):
        # When T < 0 (which happens at large y), next pulse uses |y_new|
        p = EMLPoint(0.1, 10.0)   # T = exp(0.1) - ln(10) ≈ 1.105 - 2.303 = -1.198
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_overflow_dampening(self):
        # x near OVERFLOW_THRESHOLD gets ln-dampened
        p = EMLPoint(OVERFLOW_THRESHOLD + 1.0, 1.0)
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_returns_new_object(self, unit_point):
        nxt = unit_point.mirror_pulse()
        assert nxt is not unit_point


class TestDiscreteMode:
    """Discrete mode (D set) quantizes via round(T * D)."""

    def test_discrete_quantization(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        # y_new should be round(T * 100) / 100
        T = p.tension()
        expected_y = round(T * 100) / 100
        assert nxt.y == pytest.approx(expected_y, rel=1e-12)

    def test_d_propagates_to_next(self):
        p = EMLPoint(1.0, 1.0, D=100)
        nxt = p.mirror_pulse()
        assert nxt.D == 100


class TestAxiom10Conservation:
    """Axiom 10: T + x = exp(x) at every step."""

    def test_conservation_at_unit_point(self, unit_point):
        nxt = unit_point.mirror_pulse()
        assert unit_point.conserves_tension(nxt)

    def test_conservation_over_multiple_steps(self, unit_knot):
        from eml_math.simulation import simulate_pulses, verify_conservation
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert verify_conservation(traj)


class TestTreeIntrospection:
    """is_leaf, left(), right() for converter traversal."""

    def test_flat_point_is_leaf(self, unit_point):
        assert unit_point.is_leaf()

    def test_nested_point_not_leaf(self):
        p = EMLPoint(EMLPoint(1.0, 1.0), 1.0)
        assert not p.is_leaf()

    def test_left_right_access(self):
        inner = EMLPoint(2.0, 3.0)
        outer = EMLPoint(inner, 5.0)
        assert outer.left() is inner
        assert outer.right() == 5.0


class TestResonance:
    """Axiom 14: resonance as MPM equality."""

    def test_same_point_resonates(self, unit_point):
        other = EMLPoint(1.0, 1.0)
        assert unit_point.resonates_with(other)

    def test_different_point_does_not_resonate(self, unit_point):
        other = EMLPoint(2.0, 1.0)
        assert not unit_point.resonates_with(other)
