"""
Tests for newly added EML operators: pow, asin, mod, id, eq, apply, sum_n.

All numeric operators are verified against Python stdlib to ≥10 decimal places.
"""
import math
import pytest
import eml_math.operators as ops
from eml_math.point import _LitNode


# ── ops.pow (alias for pow_fn) ────────────────────────────────────────────────

class TestLog:
    """ops.log — alias for ln (natural logarithm)."""

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, math.e, 2.0, 10.0, 100.0])
    def test_log_matches_math_log(self, x):
        assert ops.log(x).tension() == pytest.approx(math.log(x), rel=1e-10)

    def test_log_is_alias_for_ln(self):
        assert ops.log is ops.ln

    def test_log_of_e_is_one(self):
        assert ops.log(math.e).tension() == pytest.approx(1.0, rel=1e-10)

    def test_log_of_one_is_zero(self):
        assert ops.log(1.0).tension() == pytest.approx(0.0, abs=1e-12)

    def test_log_inverse_of_exp(self):
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert ops.log(ops.exp(x)).tension() == pytest.approx(x, rel=1e-10)


class TestPow:
    @pytest.mark.parametrize("base, exp, expected", [
        (2.0, 3.0, 8.0),
        (3.0, 2.0, 9.0),
        (4.0, 0.5, 2.0),
        (math.e, 1.0, math.e),
        (10.0, 0.0, 1.0),
        (2.0, -1.0, 0.5),
    ])
    def test_pow_matches_stdlib(self, base, exp, expected):
        assert ops.pow(base, exp).tension() == pytest.approx(expected, rel=1e-10)

    def test_pow_is_alias_for_pow_fn(self):
        assert ops.pow is ops.pow_fn

    def test_pow_eml_input(self):
        base = ops.add(1.0, 1.0)  # 2.0
        assert ops.pow(base, 3.0).tension() == pytest.approx(8.0, rel=1e-10)


# ── ops.asin (alias for arcsin) ───────────────────────────────────────────────

class TestAsin:
    @pytest.mark.parametrize("x", [0.0, 0.5, -0.5, 0.866, -0.866, 1.0, -1.0])
    def test_asin_matches_stdlib(self, x):
        assert ops.asin(x).tension() == pytest.approx(math.asin(x), rel=1e-10)

    def test_asin_is_alias_for_arcsin(self):
        assert ops.asin is ops.arcsin

    def test_asin_zero(self):
        assert ops.asin(0.0).tension() == pytest.approx(0.0, abs=1e-12)

    def test_asin_one(self):
        assert ops.asin(1.0).tension() == pytest.approx(math.pi / 2, rel=1e-10)

    def test_asin_neg_one(self):
        assert ops.asin(-1.0).tension() == pytest.approx(-math.pi / 2, rel=1e-10)

    def test_asin_roundtrip(self):
        for x in [-0.8, -0.3, 0.0, 0.3, 0.8]:
            assert ops.sin(ops.asin(x)).tension() == pytest.approx(x, abs=1e-10)


# ── ops.mod ───────────────────────────────────────────────────────────────────

class TestMod:
    @pytest.mark.parametrize("a, b, expected", [
        (10.0, 3.0, 1.0),
        (9.0, 3.0, 0.0),
        (7.5, 2.5, 0.0),
        (5.0, 2.0, 1.0),
        (24.0, 24.0, 0.0),
        (1.0, 24.0, 1.0),
    ])
    def test_mod_positive(self, a, b, expected):
        assert ops.mod(a, b).tension() == pytest.approx(expected, abs=1e-12)

    def test_mod_negative_numerator(self):
        # math.fmod(-10, 3) = -1.0 (sign follows dividend)
        assert ops.mod(-10.0, 3.0).tension() == pytest.approx(math.fmod(-10.0, 3.0), abs=1e-12)

    def test_mod_negative_divisor(self):
        assert ops.mod(10.0, -3.0).tension() == pytest.approx(math.fmod(10.0, -3.0), abs=1e-12)

    def test_mod_matches_stdlib(self):
        for a in [7.3, 15.7, 100.1]:
            for b in [2.1, 3.5, 7.0]:
                assert ops.mod(a, b).tension() == pytest.approx(math.fmod(a, b), abs=1e-12)

    def test_mod_eml_inputs(self):
        a = ops.mul(4.0, 6.0)   # 24.0
        b = _LitNode(5.0)
        assert ops.mod(a, b).tension() == pytest.approx(4.0, abs=1e-12)


# ── ops.id ────────────────────────────────────────────────────────────────────

class TestId:
    @pytest.mark.parametrize("x", [0.0, 1.0, -1.0, 3.14, 100.0, -42.5])
    def test_id_is_identity(self, x):
        assert ops.id(x).tension() == pytest.approx(x, rel=1e-12)

    def test_id_eml_input(self):
        p = ops.add(2.0, 3.0)   # 5.0
        assert ops.id(p).tension() == pytest.approx(5.0, rel=1e-12)

    def test_id_preserves_value_exactly(self):
        assert ops.id(math.pi).tension() == pytest.approx(math.pi, rel=1e-14)

    def test_id_zero(self):
        assert ops.id(0.0).tension() == pytest.approx(0.0, abs=1e-15)


# ── ops.eq ────────────────────────────────────────────────────────────────────

class TestEq:
    def test_equal_values(self):
        assert ops.eq(3.0, 3.0).tension() == pytest.approx(1.0)

    def test_unequal_values(self):
        assert ops.eq(3.0, 4.0).tension() == pytest.approx(0.0)

    def test_within_tolerance(self):
        assert ops.eq(1.0, 1.0 + 1e-11).tension() == pytest.approx(1.0)

    def test_outside_tolerance(self):
        assert ops.eq(1.0, 1.0 + 1e-9).tension() == pytest.approx(0.0)

    def test_eq_zero_zero(self):
        assert ops.eq(0.0, 0.0).tension() == pytest.approx(1.0)

    def test_eq_negative_equal(self):
        assert ops.eq(-5.0, -5.0).tension() == pytest.approx(1.0)

    def test_eq_negative_unequal(self):
        assert ops.eq(-5.0, 5.0).tension() == pytest.approx(0.0)

    def test_eq_modular_identity(self):
        # 24 mod 24 == 0
        result = ops.eq(ops.mod(24.0, 24.0), 0.0)
        assert result.tension() == pytest.approx(1.0)

    def test_eq_eml_inputs(self):
        a = ops.mul(2.0, 3.0)  # 6.0
        b = ops.add(4.0, 2.0)  # 6.0
        assert ops.eq(a, b).tension() == pytest.approx(1.0)


# ── ops.apply ─────────────────────────────────────────────────────────────────

class TestApply:
    def test_callable_lambda(self):
        assert ops.apply(lambda x: x * 2, 3.0).tension() == pytest.approx(6.0, rel=1e-12)

    def test_callable_math_function(self):
        assert ops.apply(math.sqrt, 4.0).tension() == pytest.approx(2.0, rel=1e-12)

    def test_callable_identity(self):
        assert ops.apply(lambda x: x, 7.5).tension() == pytest.approx(7.5, rel=1e-12)

    def test_eml_point_as_f(self):
        # When f is an EMLPoint, return f (x is ignored)
        f = _LitNode(42.0)
        result = ops.apply(f, 99.0)
        assert result.tension() == pytest.approx(42.0, rel=1e-12)

    def test_eml_expression_as_f(self):
        f = ops.mul(2.0, 3.0)  # fixed expression = 6.0
        result = ops.apply(f, 0.0)
        assert result.tension() == pytest.approx(6.0, rel=1e-12)

    def test_apply_with_eml_arg(self):
        x = ops.add(1.0, 2.0)  # 3.0
        assert ops.apply(lambda v: v ** 2, x).tension() == pytest.approx(9.0, rel=1e-12)


# ── ops.sum_n ─────────────────────────────────────────────────────────────────

class TestSumN:
    def test_constant_term_simple(self):
        # sum(2, n=1..5) = 5 * 2 = 10
        result = ops.sum_n(2.0, 1.0, 5.0)
        assert result.tension() == pytest.approx(10.0, rel=1e-12)

    def test_single_term(self):
        # sum(7, n=1..1) = 1 * 7 = 7
        result = ops.sum_n(7.0, 1.0, 1.0)
        assert result.tension() == pytest.approx(7.0, rel=1e-12)

    def test_large_range(self):
        # sum(1, n=1..125) = 125
        result = ops.sum_n(1.0, 1.0, 125.0)
        assert result.tension() == pytest.approx(125.0, rel=1e-12)

    def test_zero_term(self):
        result = ops.sum_n(0.0, 1.0, 10.0)
        assert result.tension() == pytest.approx(0.0, abs=1e-15)

    def test_sum_over_3_8_equals_24(self):
        # sum(1, n=1..24) = 24 — leech/E8 check
        result = ops.sum_n(1.0, 1.0, 24.0)
        assert result.tension() == pytest.approx(24.0, rel=1e-12)

    def test_eml_term_input(self):
        term = ops.mul(2.0, 3.0)  # 6.0
        result = ops.sum_n(term, 1.0, 4.0)
        assert result.tension() == pytest.approx(24.0, rel=1e-12)  # 4 * 6

    def test_eml_bounds(self):
        n_start = _LitNode(1.0)
        n_end = _LitNode(10.0)
        result = ops.sum_n(1.0, n_start, n_end)
        assert result.tension() == pytest.approx(10.0, rel=1e-12)


# ── Composition tests ─────────────────────────────────────────────────────────

class TestNewOperatorCompositions:
    def test_pow_in_sum(self):
        # sum(2^n evaluated as constant 8) from n=1..3 = 24
        term = ops.pow(2.0, 3.0)  # 8.0 fixed
        result = ops.sum_n(term, 1.0, 3.0)
        assert result.tension() == pytest.approx(24.0, rel=1e-12)

    def test_id_in_eq(self):
        # id(x) == x
        x = 5.0
        assert ops.eq(ops.id(x), x).tension() == pytest.approx(1.0)

    def test_asin_of_sin(self):
        for x in [-0.7, 0.0, 0.3, 0.7]:
            assert ops.asin(ops.sin(x)).tension() == pytest.approx(x, abs=1e-10)

    def test_mod_eq_zero(self):
        # 24 mod 24 == 0 (using literal to avoid float accumulation from mul)
        remainder = ops.mod(24.0, 24.0)
        check = ops.eq(remainder, 0.0)
        assert check.tension() == pytest.approx(1.0)
