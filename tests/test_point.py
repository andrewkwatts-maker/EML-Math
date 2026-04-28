"""Core EMLPoint tests — v1.2.0 slim surface.

Spacetime / Lorentz / discrete-iteration tests live in eml-spectral
(see EML-Spectral/tests/test_point_full.py).
"""
import math
import pytest
from eml_math import EMLPoint
from eml_math.constants import OVERFLOW_THRESHOLD


class TestConstruction:
    def test_basic_construction(self):
        p = EMLPoint(1.0, 1.0)
        assert p.x == 1.0
        assert p.y == 1.0

    def test_with_quantization(self):
        p = EMLPoint(1.0, 1.0, D=100.0)
        assert p._D == 100.0

    def test_safe_y_when_negative(self):
        # iterate() should silently use |y| when y < 0
        p = EMLPoint(1.0, -2.0)
        nxt = p.iterate()
        assert nxt.x > 0   # x_new = |y| = 2.0

    def test_safe_y_when_zero(self):
        p = EMLPoint(1.0, 0.0)
        nxt = p.iterate()
        # x_new uses 1e-300 sentinel; should still be positive
        assert nxt.x > 0


class TestTension:
    def test_tension_basic(self, unit_point):
        # eml(1, 1) = exp(1) - ln(1) = e
        assert unit_point.tension() == pytest.approx(math.e, rel=1e-12)

    def test_tension_alias_eml(self, unit_point):
        assert unit_point.tension() == unit_point.eml()

    def test_tension_with_negative_y(self):
        p = EMLPoint(0.0, -math.e)
        # eml(0, -e) safe → eml(0, e) = 1 - 1 = 0
        assert p.tension() == pytest.approx(0.0, abs=1e-12)


class TestIterate:
    def test_iterate_basic(self, unit_point):
        # iter(1, 1) → (1, e - 0) = (1, e)
        nxt = unit_point.iterate()
        assert nxt.x == pytest.approx(1.0)
        assert nxt.y == pytest.approx(math.e, rel=1e-12)

    def test_iterate_chain(self, unit_point):
        p = unit_point
        for _ in range(5):
            p = p.iterate()
        # convergence test omitted; just check it runs without error
        assert p.x is not None and p.y is not None

    def test_overflow_dampening(self):
        big_x = OVERFLOW_THRESHOLD * 2
        p = EMLPoint(big_x, 1.0)
        nxt = p.iterate()
        # x must have been clamped to ln(big_x) before exp
        assert math.isfinite(nxt.y)

    def test_mirror_pulse_alias(self, unit_point):
        # backwards-compat alias
        a = unit_point.iterate()
        b = unit_point.mirror_pulse()
        assert a == b

    def test_pulse_alias(self, unit_point):
        a = unit_point.iterate()
        b = unit_point.pulse()
        assert a == b


class TestTreeStructure:
    def test_is_leaf_true_when_floats(self):
        p = EMLPoint(1.0, 2.0)
        assert p.is_leaf() is True

    def test_is_leaf_false_when_nested(self):
        inner = EMLPoint(1.0, 1.0)
        outer = EMLPoint(inner, 2.0)
        assert outer.is_leaf() is False

    def test_left_right_accessors(self):
        inner = EMLPoint(1.0, 1.0)
        p = EMLPoint(inner, 2.0)
        assert p.left() is inner
        assert p.right() == 2.0


class TestEquality:
    def test_equal_when_tension_matches(self):
        a = EMLPoint(1.0, 1.0)
        b = EMLPoint(1.0, 1.0)
        assert a == b

    def test_not_equal_when_tension_differs(self):
        a = EMLPoint(1.0, 1.0)
        b = EMLPoint(2.0, 1.0)
        assert a != b

    def test_hash_stable(self):
        a = EMLPoint(1.0, 1.0)
        b = EMLPoint(1.0, 1.0)
        assert hash(a) == hash(b)


class TestDifferentiation:
    def test_diff_returns_emlpoint(self):
        from eml_math.point import _VarNode
        x = _VarNode("x")
        # Build a simple expression: eml(x, 1)
        expr = EMLPoint(x, 1.0)
        # Just check diff() doesn't raise
        result = expr.diff("x")
        assert isinstance(result, EMLPoint)


class TestRepr:
    def test_repr_basic(self):
        p = EMLPoint(1.0, 2.0)
        assert "EMLPoint" in repr(p)

    def test_repr_with_D(self):
        p = EMLPoint(1.0, 2.0, D=100.0)
        assert "D=100" in repr(p)
