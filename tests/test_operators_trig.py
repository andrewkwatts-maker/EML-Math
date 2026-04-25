"""
Comprehensive tests for EML trigonometric and hyperbolic operators.

Tests both the mathematical correctness of each function and EML-specific
identities (e.g. sin²+cos²=1, cosh²-sinh²=1, inverse cancellations).
"""
import math
import pytest
from eml_math.operators import (
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh, arsinh, arcosh, artanh,
    exp, ln, add, sub, mul, div, neg, inv, sqr, sqrt,
    half, logistic, hypot,
)
from eml_math.point import _LitNode

TOL = 1e-9


# ── sin() ─────────────────────────────────────────────────────────────────────

class TestSin:

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi])
    def test_sin_matches_stdlib(self, x):
        assert abs(sin(x).tension() - math.sin(x)) < TOL

    def test_sin_zero(self):
        assert abs(sin(0.0).tension()) < TOL

    def test_sin_pi_over_2(self):
        assert abs(sin(math.pi / 2).tension() - 1.0) < TOL

    def test_sin_pi(self):
        assert abs(sin(math.pi).tension()) < 1e-9

    def test_sin_negative(self):
        assert abs(sin(-1.0).tension() - math.sin(-1.0)) < TOL

    def test_sin_symmetry_odd(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(sin(-x).tension() + sin(x).tension()) < TOL

    def test_sin_2pi_period(self):
        for x in [0.3, 0.7, 1.2]:
            assert abs(sin(x).tension() - sin(x + 2 * math.pi).tension()) < TOL


# ── cos() ─────────────────────────────────────────────────────────────────────

class TestCos:

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2, math.pi])
    def test_cos_matches_stdlib(self, x):
        assert abs(cos(x).tension() - math.cos(x)) < TOL

    def test_cos_zero(self):
        assert abs(cos(0.0).tension() - 1.0) < TOL

    def test_cos_pi_over_2(self):
        assert abs(cos(math.pi / 2).tension()) < TOL

    def test_cos_pi(self):
        assert abs(cos(math.pi).tension() + 1.0) < TOL

    def test_cos_even_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(cos(-x).tension() - cos(x).tension()) < TOL

    def test_cos_2pi_period(self):
        for x in [0.3, 0.7, 1.2]:
            assert abs(cos(x).tension() - cos(x + 2 * math.pi).tension()) < TOL


# ── Pythagorean identity ──────────────────────────────────────────────────────

class TestPythagoreanIdentity:

    @pytest.mark.parametrize("x", [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 3.0, -1.0, -2.0])
    def test_sin_sq_plus_cos_sq_equals_1(self, x):
        s = sin(x).tension()
        c = cos(x).tension()
        assert abs(s**2 + c**2 - 1.0) < TOL

    def test_via_operator_composition(self):
        """Verify via add(sqr(sin), sqr(cos)) tree."""
        for x in [0.5, 1.0, 2.0]:
            lnx = _LitNode(x)
            result = add(sqr(sin(lnx)), sqr(cos(lnx))).tension()
            assert abs(result - 1.0) < TOL


# ── tan() ─────────────────────────────────────────────────────────────────────

class TestTan:

    @pytest.mark.parametrize("x", [0.0, 0.3, 0.7, 1.0, -0.5, -1.0])
    def test_tan_matches_stdlib(self, x):
        assert abs(tan(x).tension() - math.tan(x)) < TOL

    def test_tan_zero(self):
        assert abs(tan(0.0).tension()) < TOL

    def test_tan_equals_sin_over_cos(self):
        for x in [0.5, 1.0, 2.0]:
            t = tan(x).tension()
            s_over_c = sin(x).tension() / cos(x).tension()
            assert abs(t - s_over_c) < TOL

    def test_tan_near_pi_over_4(self):
        assert abs(tan(math.pi / 4).tension() - 1.0) < TOL


# ── Inverse trig ──────────────────────────────────────────────────────────────

class TestInverseTrig:

    @pytest.mark.parametrize("x", [-0.9, -0.5, 0.0, 0.5, 0.9])
    def test_arcsin_sin_roundtrip(self, x):
        assert abs(arcsin(sin(x).tension()).tension() - x) < TOL

    @pytest.mark.parametrize("x", [0.1, 0.5, 0.9])
    def test_arccos_cos_roundtrip(self, x):
        assert abs(arccos(cos(x).tension()).tension() - x) < TOL

    @pytest.mark.parametrize("x", [-1.0, -0.5, 0.0, 0.5, 1.0])
    def test_arctan_tan_roundtrip(self, x):
        # arctan domain is (-pi/2, pi/2) so x must be in that range
        assert abs(arctan(tan(x).tension()).tension() - x) < TOL

    def test_arcsin_0(self):
        assert abs(arcsin(0.0).tension()) < TOL

    def test_arccos_1(self):
        assert abs(arccos(1.0).tension()) < TOL

    def test_arctan_1(self):
        assert abs(arctan(1.0).tension() - math.pi / 4) < TOL

    def test_arcsin_1(self):
        assert abs(arcsin(1.0).tension() - math.pi / 2) < TOL

    def test_arcsin_minus_1(self):
        assert abs(arcsin(-1.0).tension() + math.pi / 2) < TOL

    def test_arctan_complementary(self):
        """arctan(x) + arctan(1/x) = pi/2 for x > 0."""
        for x in [0.5, 1.0, 2.0]:
            s = arctan(x).tension() + arctan(1.0 / x).tension()
            assert abs(s - math.pi / 2) < TOL


# ── sinh() / cosh() / tanh() ──────────────────────────────────────────────────

class TestHyperbolicFunctions:

    @pytest.mark.parametrize("x", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    def test_sinh_matches_stdlib(self, x):
        assert abs(sinh(x).tension() - math.sinh(x)) < TOL

    @pytest.mark.parametrize("x", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    def test_cosh_matches_stdlib(self, x):
        assert abs(cosh(x).tension() - math.cosh(x)) < TOL

    @pytest.mark.parametrize("x", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0])
    def test_tanh_matches_stdlib(self, x):
        assert abs(tanh(x).tension() - math.tanh(x)) < TOL

    def test_sinh_zero(self):
        assert abs(sinh(0.0).tension()) < TOL

    def test_cosh_zero_is_one(self):
        assert abs(cosh(0.0).tension() - 1.0) < TOL

    def test_tanh_zero(self):
        assert abs(tanh(0.0).tension()) < TOL

    def test_sinh_odd_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(sinh(-x).tension() + sinh(x).tension()) < TOL

    def test_cosh_even_symmetry(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(cosh(-x).tension() - cosh(x).tension()) < TOL


# ── Hyperbolic identity: cosh²(x) - sinh²(x) = 1 ────────────────────────────

class TestHyperbolicIdentity:

    @pytest.mark.parametrize("x", [-2.0, -1.0, 0.0, 0.5, 1.0, 2.0, 3.0])
    def test_cosh_sq_minus_sinh_sq_equals_1(self, x):
        c = cosh(x).tension()
        s = sinh(x).tension()
        assert abs(c**2 - s**2 - 1.0) < 1e-8

    def test_via_operator_composition(self):
        for x in [0.5, 1.0, 2.0]:
            lx = _LitNode(x)
            result = sub(sqr(cosh(lx)), sqr(sinh(lx))).tension()
            assert abs(result - 1.0) < 1e-8


# ── arsinh() / artanh() ───────────────────────────────────────────────────────

class TestInverseHyperbolic:

    @pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 1.0, 3.0])
    def test_arsinh_sinh_roundtrip(self, x):
        got = arsinh(sinh(x).tension()).tension()
        assert abs(got - x) < TOL

    @pytest.mark.parametrize("x", [-0.9, -0.5, 0.0, 0.5, 0.9])
    def test_artanh_tanh_roundtrip(self, x):
        got = artanh(tanh(x).tension()).tension()
        assert abs(got - x) < 1e-8

    def test_arsinh_0(self):
        assert abs(arsinh(0.0).tension()) < TOL

    def test_artanh_0(self):
        assert abs(artanh(0.0).tension()) < TOL

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0])
    def test_arsinh_matches_stdlib(self, x):
        assert abs(arsinh(x).tension() - math.asinh(x)) < TOL

    @pytest.mark.parametrize("x", [-0.9, 0.0, 0.5, 0.9])
    def test_artanh_matches_stdlib(self, x):
        assert abs(artanh(x).tension() - math.atanh(x)) < TOL


# ── tanh/logistic/half ───────────────────────────────────────────────────────

class TestUtilityFunctions:

    @pytest.mark.parametrize("x", [-2.0, 0.0, 1.0, 2.0])
    def test_half_is_x_div_2(self, x):
        assert abs(half(x).tension() - x / 2) < TOL

    def test_logistic_0_is_half(self):
        assert abs(logistic(0.0).tension() - 0.5) < TOL

    def test_logistic_positive_above_half(self):
        assert logistic(1.0).tension() > 0.5

    def test_logistic_negative_below_half(self):
        assert logistic(-1.0).tension() < 0.5

    @pytest.mark.parametrize("x", [-2.0, -0.5, 0.0, 0.5, 2.0])
    def test_logistic_matches_formula(self, x):
        got = logistic(x).tension()
        want = 1.0 / (1.0 + math.exp(-x))
        assert abs(got - want) < TOL

    def test_logistic_output_in_0_1(self):
        for x in [-10.0, -1.0, 0.0, 1.0, 10.0]:
            v = logistic(x).tension()
            assert 0.0 < v < 1.0

    @pytest.mark.parametrize("a,b", [(3.0, 4.0), (5.0, 12.0), (1.0, 1.0), (0.1, 0.1)])
    def test_hypot_matches_stdlib(self, a, b):
        got = hypot(a, b).tension()
        want = math.hypot(a, b)
        assert abs(got - want) < 1e-8


# ── Composition identities ────────────────────────────────────────────────────

class TestTrigCompositions:

    def test_sin_arcsin_roundtrip(self):
        for x in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            assert abs(sin(arcsin(x).tension()).tension() - x) < TOL

    def test_cos_arccos_roundtrip(self):
        for x in [0.1, 0.5, 1.0]:
            assert abs(cos(arccos(x).tension()).tension() - x) < TOL

    def test_tan_arctan_roundtrip(self):
        for x in [-3.0, -1.0, 0.0, 1.0, 3.0]:
            assert abs(tan(arctan(x).tension()).tension() - x) < TOL

    def test_sin_2x_double_angle(self):
        """sin(2x) = 2 sin(x) cos(x)"""
        for x in [0.3, 0.7, 1.2]:
            lhs = sin(2 * x).tension()
            rhs = 2 * sin(x).tension() * cos(x).tension()
            assert abs(lhs - rhs) < TOL

    def test_cos_2x_double_angle(self):
        """cos(2x) = cos²(x) - sin²(x)"""
        for x in [0.3, 0.7, 1.2]:
            lhs = cos(2 * x).tension()
            rhs = cos(x).tension()**2 - sin(x).tension()**2
            assert abs(lhs - rhs) < TOL

    def test_sinh_plus_cosh_is_exp(self):
        """sinh(x) + cosh(x) = exp(x)"""
        for x in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            lhs = sinh(x).tension() + cosh(x).tension()
            rhs = math.exp(x)
            assert abs(lhs - rhs) < TOL

    def test_cosh_minus_sinh_is_exp_neg(self):
        """cosh(x) - sinh(x) = exp(-x)"""
        for x in [-1.0, 0.0, 0.5, 1.0, 2.0]:
            lhs = cosh(x).tension() - sinh(x).tension()
            rhs = math.exp(-x)
            assert abs(lhs - rhs) < TOL
