"""
Tests for eml.operators — all EML compositions vs math stdlib.

Every operator is verified against Python's math module to 10+ decimal places.
"""
import math
import pytest
import eml_math.operators as ops


class TestConstants:
    def test_const_e(self):
        assert ops.const_e() == pytest.approx(math.e, rel=1e-14)

    def test_const_neg_one(self):
        assert ops.const_neg_one() == pytest.approx(-1.0, abs=1e-10)

    def test_const_half(self):
        assert ops.const_half() == pytest.approx(0.5, rel=1e-10)


class TestExpLn:
    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0, 5.0])
    def test_exp_matches_stdlib(self, x):
        assert ops.exp(x).tension() == pytest.approx(math.exp(x), rel=1e-12)

    @pytest.mark.parametrize("x", [0.1, 0.5, 1.0, math.e, 10.0])
    def test_ln_matches_stdlib(self, x):
        assert ops.ln(x).tension() == pytest.approx(math.log(x), rel=1e-10)

    def test_exp_ln_inverse(self):
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert ops.exp(ops.ln(x)).tension() == pytest.approx(x, rel=1e-10)

    def test_ln_exp_inverse(self):
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert ops.ln(ops.exp(x)).tension() == pytest.approx(x, rel=1e-10)


class TestArithmetic:
    @pytest.mark.parametrize("a,b", [(1, 2), (3, 4), (0.5, 0.7), (10, 0.1)])
    def test_add(self, a, b):
        assert ops.add(a, b).tension() == pytest.approx(a + b, rel=1e-10)

    @pytest.mark.parametrize("a,b", [(5, 2), (3, 1), (2.5, 0.5)])
    def test_sub(self, a, b):
        assert ops.sub(a, b).tension() == pytest.approx(a - b, rel=1e-10)

    @pytest.mark.parametrize("x", [1.0, 2.0, 3.0, 0.5])
    def test_neg(self, x):
        assert ops.neg(x).tension() == pytest.approx(-x, rel=1e-10)

    @pytest.mark.parametrize("x", [0.5, 1.0, 2.0, 4.0])
    def test_inv(self, x):
        assert ops.inv(x).tension() == pytest.approx(1.0 / x, rel=1e-10)

    @pytest.mark.parametrize("a,b", [(2, 3), (4, 5), (1.5, 2.5)])
    def test_mul(self, a, b):
        assert ops.mul(a, b).tension() == pytest.approx(a * b, rel=1e-10)

    @pytest.mark.parametrize("a,b", [(6, 2), (10, 4), (3, 1.5)])
    def test_div(self, a, b):
        assert ops.div(a, b).tension() == pytest.approx(a / b, rel=1e-10)

    @pytest.mark.parametrize("x", [1.0, 2.0, 3.0, 0.5])
    def test_sqr(self, x):
        assert ops.sqr(x).tension() == pytest.approx(x ** 2, rel=1e-10)

    @pytest.mark.parametrize("x", [1.0, 2.0, 4.0, 9.0])
    def test_sqrt(self, x):
        assert ops.sqrt(x).tension() == pytest.approx(math.sqrt(x), rel=1e-10)

    @pytest.mark.parametrize("base,exp", [(2, 3), (3, 2), (2, 0.5), (math.e, 1)])
    def test_pow(self, base, exp):
        assert ops.pow_fn(base, exp).tension() == pytest.approx(base ** exp, rel=1e-10)

    def test_chained_add_mul(self):
        # exp(ln(2) + ln(3)) = exp(ln(6)) = 6
        result = ops.exp(ops.add(ops.ln(2), ops.ln(3))).tension()
        assert result == pytest.approx(6.0, rel=1e-10)

    def test_log_base_2(self):
        assert ops.log_fn(2.0, 8.0).tension() == pytest.approx(3.0, rel=1e-10)

    def test_avg(self):
        assert ops.avg(3.0, 7.0).tension() == pytest.approx(5.0, rel=1e-10)

    def test_hypot(self):
        assert ops.hypot(3.0, 4.0).tension() == pytest.approx(5.0, rel=1e-10)


class TestHyperbolic:
    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0])
    def test_sinh(self, x):
        assert ops.sinh(x).tension() == pytest.approx(math.sinh(x), rel=1e-10)

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0])
    def test_cosh(self, x):
        assert ops.cosh(x).tension() == pytest.approx(math.cosh(x), rel=1e-10)

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, 2.0])
    def test_tanh(self, x):
        assert ops.tanh(x).tension() == pytest.approx(math.tanh(x), rel=1e-10)

    def test_arsinh(self):
        for x in [0.5, 1.0, 2.0]:
            assert ops.arsinh(x).tension() == pytest.approx(math.asinh(x), rel=1e-10)

    def test_arcosh(self):
        for x in [1.0, 2.0, 3.0]:
            assert ops.arcosh(x).tension() == pytest.approx(math.acosh(x), rel=1e-10)

    def test_artanh(self):
        for x in [0.1, 0.5, 0.9]:
            assert ops.artanh(x).tension() == pytest.approx(math.atanh(x), rel=1e-10)


class TestTrigonometric:
    @pytest.mark.parametrize("x", [0.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2])
    def test_sin(self, x):
        assert ops.sin(x).tension() == pytest.approx(math.sin(x), abs=1e-10)

    @pytest.mark.parametrize("x", [0.0, math.pi/6, math.pi/4, math.pi/3, math.pi/2])
    def test_cos(self, x):
        assert ops.cos(x).tension() == pytest.approx(math.cos(x), abs=1e-10)

    @pytest.mark.parametrize("x", [0.0, math.pi/6, math.pi/4, math.pi/3])
    def test_tan(self, x):
        assert ops.tan(x).tension() == pytest.approx(math.tan(x), rel=1e-10)

    def test_pythagorean_identity(self):
        for x in [0.1, 0.5, 1.0, 1.5]:
            s = ops.sin(x).tension()
            c = ops.cos(x).tension()
            assert s ** 2 + c ** 2 == pytest.approx(1.0, abs=1e-10)

    def test_arcsin(self):
        for x in [0.0, 0.5, 1.0]:
            assert ops.arcsin(x).tension() == pytest.approx(math.asin(x), rel=1e-10)

    def test_arccos(self):
        for x in [0.0, 0.5, 1.0]:
            assert ops.arccos(x).tension() == pytest.approx(math.acos(x), rel=1e-10)

    def test_arctan(self):
        for x in [0.0, 0.5, 1.0, 2.0]:
            assert ops.arctan(x).tension() == pytest.approx(math.atan(x), rel=1e-10)

    def test_logistic(self):
        for x in [-2.0, 0.0, 2.0]:
            expected = 1.0 / (1.0 + math.exp(-x))
            assert ops.logistic(x).tension() == pytest.approx(expected, rel=1e-10)


class TestNonEML:
    def test_mirror_abs_positive(self):
        assert ops.mirror_abs(3.5) == 3.5

    def test_mirror_abs_negative(self):
        assert ops.mirror_abs(-3.5) == 3.5

    def test_quantize(self):
        assert ops.quantize(2.718, 100) == 272
        assert ops.quantize(1.0, 100) == 100
