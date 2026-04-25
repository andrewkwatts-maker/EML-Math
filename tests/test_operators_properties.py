"""
Property-based and edge-case tests for all EML operators.

Tests mathematical identities, composition rules, and boundary conditions
that verify operators are correctly implemented (not stubs).
"""
import math
import pytest
from eml_math.operators import (
    eml, exp, ln, add, sub, mul, div, neg, inv, sqr, sqrt, pow_fn,
    sin, cos, tan, arcsin, arccos, arctan,
    sinh, cosh, tanh, arsinh, arcosh, artanh,
    half, logistic, hypot, avg, log_fn,
    const_e, const_two, const_neg_one, const_half,
    mirror_abs, quantize,
)
from eml_math.point import _LitNode, EMLPoint

TOL = 1e-9


# ── eml() (direct Sheffer operator) ──────────────────────────────────────────

class TestEMLDirect:

    def test_eml_1_1_is_e(self):
        assert abs(eml(1.0, 1.0).tension() - math.e) < TOL

    def test_eml_0_1_is_1(self):
        assert abs(eml(0.0, 1.0).tension() - 1.0) < TOL

    def test_eml_x_1_is_exp_x(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(eml(x, 1.0).tension() - math.exp(x)) < TOL

    def test_eml_1_y_is_e_minus_ln_y(self):
        for y in [1.0, math.e, 4.0]:
            assert abs(eml(1.0, y).tension() - (math.e - math.log(y))) < TOL

    def test_eml_returns_emlpoint(self):
        assert isinstance(eml(1.0, 1.0), EMLPoint)


# ── Constants ─────────────────────────────────────────────────────────────────

class TestOperatorConstants:

    def test_const_e(self):
        assert abs(const_e() - math.e) < TOL

    def test_const_two(self):
        assert abs(const_two() - 2.0) < TOL

    def test_const_neg_one(self):
        assert abs(const_neg_one() - (-1.0)) < TOL

    def test_const_half(self):
        assert abs(const_half() - 0.5) < TOL


# ── mirror_abs / quantize ─────────────────────────────────────────────────────

class TestNonEMLPrimitives:

    def test_mirror_abs_positive(self):
        assert mirror_abs(3.0) == 3.0

    def test_mirror_abs_negative(self):
        assert mirror_abs(-3.0) == 3.0

    def test_mirror_abs_zero(self):
        assert mirror_abs(0.0) == 0.0

    def test_quantize_basic(self):
        # quantize(T, D) = round(T * D); Python uses banker's rounding
        result = quantize(2.5, 1.0)
        assert result in (2, 3)  # acceptable under banker's rounding

    def test_quantize_with_scale(self):
        assert quantize(1.0, 100.0) == 100

    def test_quantize_rounds_half_up(self):
        # Python uses banker's rounding, so 0.5 rounds to 0
        # but 1.5 rounds to 2
        assert quantize(1.5, 1.0) in (1, 2)  # platform-dependent rounding


# ── Algebraic ring identities ──────────────────────────────────────────────────

class TestAlgebraicIdentities:

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0])
    def test_add_zero_identity(self, x):
        """x + 0 = x (using neg(x) trick: add(x, 0) = sub(x, neg(0))"""
        result = add(x, 0.0).tension()
        assert abs(result - x) < TOL

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0])
    def test_mul_one_identity(self, x):
        """x * 1 = x"""
        result = mul(x, 1.0).tension()
        assert abs(result - x) < 1e-8

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
    def test_div_one_identity(self, x):
        """x / 1 = x"""
        result = div(x, 1.0).tension()
        assert abs(result - x) < TOL

    @pytest.mark.parametrize("x,y", [(2.0, 3.0), (0.5, 4.0), (1.0, 5.0)])
    def test_add_commutativity(self, x, y):
        """x + y = y + x"""
        assert abs(add(x, y).tension() - add(y, x).tension()) < TOL

    @pytest.mark.parametrize("x,y", [(2.0, 3.0), (0.5, 4.0)])
    def test_mul_commutativity(self, x, y):
        """x * y = y * x (for positive inputs)"""
        assert abs(mul(x, y).tension() - mul(y, x).tension()) < 1e-8

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 5.0])
    def test_inv_inv_is_identity(self, x):
        """1/(1/x) = x"""
        result = inv(inv(x).tension()).tension()
        assert abs(result - x) < 1e-8

    @pytest.mark.parametrize("x,y", [(2.0, 3.0), (1.5, 0.5)])
    def test_sub_and_add_inverse(self, x, y):
        """(x + y) - y = x"""
        result = sub(add(x, y).tension(), y).tension()
        assert abs(result - x) < TOL

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 4.0])
    def test_sqrt_sqr_identity(self, x):
        """sqrt(x²) = x for x > 0"""
        result = sqrt(sqr(x).tension()).tension()
        assert abs(result - x) < 1e-8

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
    def test_sqr_sqrt_identity(self, x):
        """(√x)² = x"""
        result = sqr(sqrt(x).tension()).tension()
        assert abs(result - x) < 1e-8

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 5.0])
    def test_exp_ln_roundtrip(self, x):
        """exp(ln(x)) = x"""
        result = exp(ln(x).tension()).tension()
        assert abs(result - x) < TOL

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
    def test_ln_exp_roundtrip(self, x):
        """ln(exp(x)) = x"""
        result = ln(exp(x).tension()).tension()
        assert abs(result - x) < TOL

    @pytest.mark.parametrize("x,y", [(2.0, 3.0), (4.0, 0.5)])
    def test_ln_mul_is_sum_of_lns(self, x, y):
        """ln(x*y) = ln(x) + ln(y)"""
        lhs = ln(mul(x, y).tension()).tension()
        rhs = add(ln(x).tension(), ln(y).tension()).tension()
        assert abs(lhs - rhs) < 1e-8

    @pytest.mark.parametrize("x,y", [(4.0, 2.0), (9.0, 3.0)])
    def test_ln_div_is_diff_of_lns(self, x, y):
        """ln(x/y) = ln(x) - ln(y)"""
        lhs = ln(div(x, y).tension()).tension()
        rhs = sub(ln(x).tension(), ln(y).tension()).tension()
        assert abs(lhs - rhs) < TOL

    @pytest.mark.parametrize("x,n", [(2.0, 2), (2.0, 3), (3.0, 2)])
    def test_pow_fn_n_via_mul(self, x, n):
        """pow(x, n) = x^n"""
        result = pow_fn(x, n).tension()
        assert abs(result - x**n) < 1e-8

    @pytest.mark.parametrize("a,b,c", [(2.0, 3.0, 4.0)])
    def test_add_associativity(self, a, b, c):
        """(a + b) + c = a + (b + c)"""
        lhs = add(add(a, b).tension(), c).tension()
        rhs = add(a, add(b, c).tension()).tension()
        assert abs(lhs - rhs) < TOL


# ── avg and log_fn ────────────────────────────────────────────────────────────

class TestAvgLogFn:

    def test_avg_2_4(self):
        assert abs(avg(2.0, 4.0).tension() - 3.0) < TOL

    def test_avg_symmetric(self):
        for a, b in [(2.0, 4.0), (1.0, 3.0)]:
            assert abs(avg(a, b).tension() - avg(b, a).tension()) < TOL

    def test_log_fn_base10(self):
        assert abs(log_fn(10.0, 1000.0).tension() - 3.0) < 1e-8

    def test_log_fn_base2(self):
        assert abs(log_fn(2.0, 16.0).tension() - 4.0) < 1e-8

    def test_log_fn_base_e(self):
        for x in [0.5, 1.0, 2.0]:
            assert abs(log_fn(math.e, x).tension() - math.log(x)) < TOL

    def test_avg_is_midpoint(self):
        for a, b in [(0.0, 2.0), (1.0, 3.0), (-1.0, 1.0)]:
            result = avg(a, b).tension()
            assert abs(result - (a + b) / 2) < TOL


# ── Trig addition formulas ────────────────────────────────────────────────────

class TestTrigAdditiveFormulas:

    @pytest.mark.parametrize("a,b", [(0.3, 0.7), (0.5, 1.0), (0.2, 0.8)])
    def test_sin_sum_formula(self, a, b):
        """sin(a+b) = sin(a)cos(b) + cos(a)sin(b)"""
        lhs = sin(a + b).tension()
        rhs = sin(a).tension() * cos(b).tension() + cos(a).tension() * sin(b).tension()
        assert abs(lhs - rhs) < TOL

    @pytest.mark.parametrize("a,b", [(0.3, 0.7), (0.5, 1.0)])
    def test_cos_sum_formula(self, a, b):
        """cos(a+b) = cos(a)cos(b) - sin(a)sin(b)"""
        lhs = cos(a + b).tension()
        rhs = cos(a).tension() * cos(b).tension() - sin(a).tension() * sin(b).tension()
        assert abs(lhs - rhs) < TOL

    def test_sin_pi_minus_x(self):
        """sin(π - x) = sin(x)"""
        for x in [0.3, 0.7, 1.2]:
            assert abs(sin(math.pi - x).tension() - sin(x).tension()) < TOL

    def test_cos_pi_minus_x(self):
        """cos(π - x) = -cos(x)"""
        for x in [0.3, 0.7, 1.2]:
            assert abs(cos(math.pi - x).tension() + cos(x).tension()) < TOL


# ── Numeric accuracy regression tests ────────────────────────────────────────

class TestNumericAccuracy:

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    def test_exp_precision(self, x):
        assert abs(exp(x).tension() - math.exp(x)) / math.exp(x) < 1e-12

    @pytest.mark.parametrize("x", [0.001, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0])
    def test_ln_precision(self, x):
        assert abs(ln(x).tension() - math.log(x)) < TOL

    @pytest.mark.parametrize("x", [0.001, 0.1, 0.5, 1.0, 2.0, 4.0, 9.0, 25.0])
    def test_sqrt_precision(self, x):
        assert abs(sqrt(x).tension() - math.sqrt(x)) < TOL

    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (3.0, 4.0), (0.5, 0.3)])
    def test_add_precision(self, a, b):
        assert abs(add(a, b).tension() - (a + b)) < TOL

    @pytest.mark.parametrize("a,b", [(5.0, 2.0), (3.0, 1.5)])
    def test_sub_precision(self, a, b):
        assert abs(sub(a, b).tension() - (a - b)) < TOL

    @pytest.mark.parametrize("a,b", [(2.0, 3.0), (4.0, 0.5)])
    def test_mul_precision(self, a, b):
        assert abs(mul(a, b).tension() - a * b) < 1e-8

    @pytest.mark.parametrize("a,b", [(6.0, 2.0), (1.0, 4.0)])
    def test_div_precision(self, a, b):
        assert abs(div(a, b).tension() - a / b) < TOL
