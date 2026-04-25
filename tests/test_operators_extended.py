"""
Extended tests for EML operators — focusing on correctness of sqrt, pow_fn,
and the _ScaleNode fix for negative-intermediate multiplications.

These tests are all pure Python (no Rust extension needed).
"""
import math
import pytest
from eml_math.operators import (
    sqrt, pow_fn, exp, ln, add, sub, mul, div, neg, inv, sqr, hypot,
    avg, log_fn,
)
from eml_math.operators import _ScaleNode
from eml_math.point import _LitNode


# ── _ScaleNode ────────────────────────────────────────────────────────────────

class TestScaleNode:

    def test_scale_half(self):
        node = _ScaleNode(_LitNode(4.0), 0.5)
        assert abs(node.tension() - 2.0) < 1e-12

    def test_scale_double(self):
        node = _ScaleNode(_LitNode(3.0), 2.0)
        assert abs(node.tension() - 6.0) < 1e-12

    def test_scale_negative_input(self):
        node = _ScaleNode(_LitNode(-5.0), 0.5)
        assert abs(node.tension() - (-2.5)) < 1e-12

    def test_scale_negative_scale(self):
        node = _ScaleNode(_LitNode(4.0), -1.0)
        assert abs(node.tension() - (-4.0)) < 1e-12

    def test_scale_zero_input(self):
        node = _ScaleNode(_LitNode(0.0), 0.5)
        assert abs(node.tension()) < 1e-12

    def test_scale_one(self):
        node = _ScaleNode(_LitNode(7.0), 1.0)
        assert abs(node.tension() - 7.0) < 1e-12

    def test_not_leaf(self):
        node = _ScaleNode(_LitNode(1.0), 0.5)
        assert not node.is_leaf()

    def test_repr_contains_scale(self):
        node = _ScaleNode(_LitNode(1.0), 0.5)
        assert '0.5' in repr(node)

    def test_scale_by_zero(self):
        node = _ScaleNode(_LitNode(99.0), 0.0)
        assert abs(node.tension()) < 1e-12


# ── sqrt() ────────────────────────────────────────────────────────────────────

class TestSqrt:

    def test_sqrt_of_1(self):
        assert abs(sqrt(1.0).tension() - 1.0) < 1e-10

    def test_sqrt_of_4(self):
        assert abs(sqrt(4.0).tension() - 2.0) < 1e-10

    def test_sqrt_of_9(self):
        assert abs(sqrt(9.0).tension() - 3.0) < 1e-10

    def test_sqrt_of_0_point_25(self):
        assert abs(sqrt(0.25).tension() - 0.5) < 1e-10

    def test_sqrt_of_0_point_1(self):
        """Key regression: was returning ~3.16 instead of ~0.316 (x < 1 bug)."""
        got = sqrt(0.1).tension()
        want = math.sqrt(0.1)
        assert abs(got - want) < 1e-10, f"sqrt(0.1) = {got}, expected {want}"

    def test_sqrt_of_0_point_5(self):
        got = sqrt(0.5).tension()
        want = math.sqrt(0.5)
        assert abs(got - want) < 1e-10

    def test_sqrt_of_0_point_01(self):
        got = sqrt(0.01).tension()
        want = math.sqrt(0.01)
        assert abs(got - want) < 1e-10

    def test_sqrt_of_100(self):
        assert abs(sqrt(100.0).tension() - 10.0) < 1e-10

    def test_sqrt_of_2(self):
        got = sqrt(2.0).tension()
        want = math.sqrt(2)
        assert abs(got - want) < 1e-10

    def test_sqrt_of_e(self):
        got = sqrt(math.e).tension()
        want = math.sqrt(math.e)
        assert abs(got - want) < 1e-10

    def test_sqrt_of_pi(self):
        got = sqrt(math.pi).tension()
        want = math.sqrt(math.pi)
        assert abs(got - want) < 1e-10

    def test_sqrt_squared_is_identity(self):
        """sqrt(x)² = x"""
        for x in [0.1, 0.5, 1.0, 2.0, 4.0]:
            got = sqr(sqrt(x)).tension()
            assert abs(got - x) < 1e-8, f"sqrt({x})^2 = {got}"

    def test_sqrt_of_sqrt(self):
        """sqrt(sqrt(x)) = x^(1/4)"""
        for x in [0.1, 1.0, 16.0]:
            got = sqrt(sqrt(x)).tension()
            want = x ** 0.25
            assert abs(got - want) < 1e-8, f"sqrt(sqrt({x})) = {got}"

    @pytest.mark.parametrize("x", [0.01, 0.1, 0.2, 0.5, 0.9, 1.0, 1.5, 2.0, 4.0, 9.0, 25.0, 100.0])
    def test_sqrt_parametric(self, x):
        got = sqrt(x).tension()
        want = math.sqrt(x)
        assert abs(got - want) < 1e-9, f"sqrt({x}) = {got}, want {want}"


# ── pow_fn() ─────────────────────────────────────────────────────────────────

class TestPowFn:

    def test_pow_2_squared(self):
        assert abs(pow_fn(2.0, 2).tension() - 4.0) < 1e-10

    def test_pow_2_cubed(self):
        assert abs(pow_fn(2.0, 3).tension() - 8.0) < 1e-10

    def test_pow_x_to_half_is_sqrt(self):
        """pow(x, 0.5) = sqrt(x)"""
        for x in [0.1, 0.5, 1.0, 4.0, 9.0]:
            got = pow_fn(x, 0.5).tension()
            want = math.sqrt(x)
            assert abs(got - want) < 1e-10, f"pow({x}, 0.5) = {got}"

    def test_pow_x_to_1(self):
        for x in [0.5, 1.0, 2.0, 3.0]:
            got = pow_fn(x, 1).tension()
            assert abs(got - x) < 1e-9

    def test_pow_x_to_0(self):
        for x in [0.5, 1.0, 2.0, 10.0]:
            got = pow_fn(x, 0).tension()
            assert abs(got - 1.0) < 1e-9

    def test_pow_x_to_minus_1_is_inv(self):
        """pow(x, -1) = 1/x"""
        for x in [0.5, 1.0, 2.0, 4.0]:
            got = pow_fn(x, -1).tension()
            want = 1.0 / x
            assert abs(got - want) < 1e-10, f"pow({x}, -1) = {got}"

    def test_pow_x_to_minus_half(self):
        """pow(x, -0.5) = 1/sqrt(x)"""
        for x in [0.25, 1.0, 4.0, 9.0]:
            got = pow_fn(x, -0.5).tension()
            want = 1.0 / math.sqrt(x)
            assert abs(got - want) < 1e-10, f"pow({x}, -0.5) = {got}"

    def test_pow_0_point_5_squared(self):
        """(0.5)^2 = 0.25"""
        assert abs(pow_fn(0.5, 2).tension() - 0.25) < 1e-10

    def test_pow_0_point_1_squared(self):
        """(0.1)^2 = 0.01"""
        assert abs(pow_fn(0.1, 2).tension() - 0.01) < 1e-10

    @pytest.mark.parametrize("base,exp_val", [
        (2.0, 2), (3.0, 3), (0.5, 2), (0.1, 2), (4.0, 0.5), (8.0, 1/3),
    ])
    def test_pow_parametric(self, base, exp_val):
        got = pow_fn(base, exp_val).tension()
        want = base ** exp_val
        assert abs(got - want) < 1e-8, f"pow({base}, {exp_val}) = {got}, want {want}"


# ── Other operators (regression coverage) ────────────────────────────────────

class TestOperatorsRegression:

    def test_exp_1(self):
        assert abs(exp(1.0).tension() - math.e) < 1e-12

    def test_exp_0(self):
        assert abs(exp(0.0).tension() - 1.0) < 1e-12

    def test_ln_e(self):
        assert abs(ln(math.e).tension() - 1.0) < 1e-10

    def test_ln_1(self):
        assert abs(ln(1.0).tension()) < 1e-10

    def test_add_integers(self):
        assert abs(add(3.0, 4.0).tension() - 7.0) < 1e-9

    def test_sub_integers(self):
        assert abs(sub(5.0, 3.0).tension() - 2.0) < 1e-9

    def test_mul_integers(self):
        assert abs(mul(3.0, 4.0).tension() - 12.0) < 1e-9

    def test_div_integers(self):
        assert abs(div(10.0, 2.0).tension() - 5.0) < 1e-9

    def test_neg_positive(self):
        assert abs(neg(3.0).tension() - (-3.0)) < 1e-12

    def test_neg_negative(self):
        assert abs(neg(-3.0).tension() - 3.0) < 1e-12

    def test_inv_2(self):
        assert abs(inv(2.0).tension() - 0.5) < 1e-10

    def test_inv_0_5(self):
        assert abs(inv(0.5).tension() - 2.0) < 1e-10

    def test_sqr_3(self):
        assert abs(sqr(3.0).tension() - 9.0) < 1e-9

    def test_sqr_small(self):
        assert abs(sqr(0.5).tension() - 0.25) < 1e-9

    def test_hypot_3_4(self):
        assert abs(hypot(3.0, 4.0).tension() - 5.0) < 1e-8

    def test_avg_2_4(self):
        assert abs(avg(2.0, 4.0).tension() - 3.0) < 1e-9

    def test_log_base_10(self):
        got = log_fn(10.0, 100.0).tension()
        assert abs(got - 2.0) < 1e-8

    def test_log_base_2(self):
        got = log_fn(2.0, 8.0).tension()
        assert abs(got - 3.0) < 1e-8

    def test_mul_with_fraction(self):
        """mul(0.5, 4.0) = 2.0 — positive operands work via mul identity"""
        assert abs(mul(0.5, 4.0).tension() - 2.0) < 1e-9

    def test_sqrt_times_sqrt_equals_identity(self):
        """sqrt(x) * sqrt(x) = x"""
        for x in [0.25, 1.0, 4.0]:
            got = mul(sqrt(x), sqrt(x)).tension()
            assert abs(got - x) < 1e-8

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, math.e, math.pi])
    def test_exp_ln_roundtrip(self, x):
        got = exp(ln(x)).tension()
        assert abs(got - x) < 1e-9

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
    def test_ln_exp_roundtrip(self, x):
        got = ln(exp(x)).tension()
        assert abs(got - x) < 1e-9

    def test_div_fractions(self):
        assert abs(div(1.0, 3.0).tension() - (1/3)) < 1e-9

    def test_add_negatives(self):
        assert abs(add(-2.0, 5.0).tension() - 3.0) < 1e-9

    def test_sub_to_negative(self):
        assert abs(sub(2.0, 5.0).tension() - (-3.0)) < 1e-9
