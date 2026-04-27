"""
Tests for eml_math.evaluator — EMLEvaluator and eml_eval.

Verifies that eml_description strings are parsed correctly and that
evaluated results match expected values from the parameter context.
"""
import math
import pytest

from eml_math.evaluator import EMLEvaluator, eml_eval, ParseError
from eml_math.operators import eml_scalar, eml_pi, eml_vec
import eml_math.operators as ops


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CONTEXT = {
    "alpha_inv": 137.035999177,
    "b3": 24.0,
    "phi": (1 + math.sqrt(5)) / 2,
    "pi": math.pi,
    "mass.electron": 0.51099895,
    "mass.muon": 105.6583755,
}


# ---------------------------------------------------------------------------
# ParseError
# ---------------------------------------------------------------------------

class TestParseError:
    def test_missing_prefix(self):
        ev = EMLEvaluator(CONTEXT)
        with pytest.raises(ParseError, match="must start with"):
            ev.eval("ops.mul(eml_scalar(2), eml_scalar(3)) — two times three")

    def test_empty_prefix(self):
        ev = EMLEvaluator(CONTEXT)
        with pytest.raises(ParseError):
            ev.eval("")

    def test_prefix_only(self):
        ev = EMLEvaluator(CONTEXT)
        # "EML: " with no expression — evaluates to empty string → ParseError
        with pytest.raises(Exception):
            ev.eval("EML: ")


# ---------------------------------------------------------------------------
# Parsing: strip prefix and description tail
# ---------------------------------------------------------------------------

class TestParsing:
    def test_em_dash_separator(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(42.0) — the answer to everything")
        assert result == pytest.approx(42.0, rel=1e-12)

    def test_en_dash_separator(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(3.14) – approximate pi")
        assert result == pytest.approx(3.14, rel=1e-6)

    def test_double_hyphen_separator(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(1.0) -- one")
        assert result == pytest.approx(1.0, rel=1e-12)

    def test_no_description(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(7.0)")
        assert result == pytest.approx(7.0, rel=1e-12)

    def test_whitespace_stripped(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("  EML:   eml_scalar(5.0)   — five  ")
        assert result == pytest.approx(5.0, rel=1e-12)


# ---------------------------------------------------------------------------
# Literal constructors: eml_scalar, eml_pi, eml_vec
# ---------------------------------------------------------------------------

class TestLiterals:
    def test_eml_scalar_int(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(137) — fine structure inverse")
        assert result == pytest.approx(137.0, rel=1e-12)

    def test_eml_scalar_float(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_scalar(2.718281828) — approx e")
        assert result == pytest.approx(math.e, rel=1e-9)

    def test_eml_pi(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_pi() — pi")
        assert result == pytest.approx(math.pi, rel=1e-14)

    def test_eml_vec_known_key(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_vec('alpha_inv') — alpha-inv")
        assert result == pytest.approx(137.035999177, rel=1e-10)

    def test_eml_vec_dotted_key(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: eml_vec('mass.electron') — electron mass MeV")
        assert result == pytest.approx(0.51099895, rel=1e-10)


# ---------------------------------------------------------------------------
# Arithmetic operators
# ---------------------------------------------------------------------------

class TestArithmetic:
    def test_mul_scalars(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.mul(eml_scalar(3.0), eml_scalar(4.0)) — 3×4")
        assert result == pytest.approx(12.0, rel=1e-12)

    def test_div_scalars(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.div(eml_scalar(10.0), eml_scalar(4.0)) — 10/4")
        assert result == pytest.approx(2.5, rel=1e-12)

    def test_add_scalars(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.add(eml_scalar(1.5), eml_scalar(2.5)) — 1.5+2.5")
        assert result == pytest.approx(4.0, rel=1e-12)

    def test_sub_scalars(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.sub(eml_scalar(7.0), eml_scalar(3.0)) — 7-3")
        assert result == pytest.approx(4.0, rel=1e-12)

    def test_neg(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.neg(eml_scalar(5.0)) — -5")
        assert result == pytest.approx(-5.0, rel=1e-12)

    def test_inv(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.inv(eml_scalar(4.0)) — 1/4")
        assert result == pytest.approx(0.25, rel=1e-12)

    def test_sqrt(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.sqrt(eml_scalar(9.0)) — √9")
        assert result == pytest.approx(3.0, rel=1e-12)

    def test_pow(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.pow(eml_scalar(2.0), eml_scalar(10.0)) — 2^10")
        assert result == pytest.approx(1024.0, rel=1e-10)

    def test_exp(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.exp(eml_scalar(1.0)) — e^1")
        assert result == pytest.approx(math.e, rel=1e-12)

    def test_ln(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.ln(eml_scalar(1.0)) — ln(1)")
        assert result == pytest.approx(0.0, abs=1e-14)


# ---------------------------------------------------------------------------
# Trig operators
# ---------------------------------------------------------------------------

class TestTrig:
    def test_sin(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.sin(eml_pi()) — sin(π)")
        assert result == pytest.approx(0.0, abs=1e-14)

    def test_cos(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.cos(eml_scalar(0.0)) — cos(0)")
        assert result == pytest.approx(1.0, rel=1e-12)

    def test_sin_pi_over_6(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.sin(ops.div(eml_pi(), eml_scalar(6.0))) — sin(π/6)")
        assert result == pytest.approx(0.5, rel=1e-12)


# ---------------------------------------------------------------------------
# Context-bound eml_vec in compound expressions
# ---------------------------------------------------------------------------

class TestContextBound:
    def test_vec_in_mul(self):
        ev = EMLEvaluator(CONTEXT)
        # alpha_inv / 2
        result = ev.eval(
            "EML: ops.div(eml_vec('alpha_inv'), eml_scalar(2.0)) — α⁻¹/2"
        )
        assert result == pytest.approx(137.035999177 / 2, rel=1e-10)

    def test_vec_ratio(self):
        ev = EMLEvaluator(CONTEXT)
        # muon/electron mass ratio ≈ 206.77
        result = ev.eval(
            "EML: ops.div(eml_vec('mass.muon'), eml_vec('mass.electron')) — muon/electron"
        )
        assert result == pytest.approx(105.6583755 / 0.51099895, rel=1e-8)

    def test_vec_sqrt(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.sqrt(eml_vec('b3')) — √b₃")
        assert result == pytest.approx(math.sqrt(24.0), rel=1e-12)

    def test_vec_b3_over_pi(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.eval("EML: ops.div(eml_vec('b3'), eml_pi()) — b₃/π")
        assert result == pytest.approx(24.0 / math.pi, rel=1e-12)


# ---------------------------------------------------------------------------
# Strict vs non-strict mode
# ---------------------------------------------------------------------------

class TestStrictMode:
    def test_strict_raises_on_unknown_vec(self):
        ev = EMLEvaluator(CONTEXT, strict=True)
        with pytest.raises(KeyError, match="eml_vec"):
            ev.eval("EML: eml_vec('does.not.exist') — missing")

    def test_non_strict_returns_zero_for_unknown_vec(self):
        ev = EMLEvaluator(CONTEXT, strict=False)
        result = ev.eval("EML: eml_vec('does.not.exist') — missing")
        assert result == pytest.approx(0.0, abs=1e-12)
        assert "does.not.exist" in ev.missing_refs

    def test_non_strict_records_all_missing(self):
        ev = EMLEvaluator(CONTEXT, strict=False)
        ev.eval("EML: ops.add(eml_vec('x.missing'), eml_vec('y.missing')) — two missing")
        assert "x.missing" in ev.missing_refs
        assert "y.missing" in ev.missing_refs


# ---------------------------------------------------------------------------
# try_eval
# ---------------------------------------------------------------------------

class TestTryEval:
    def test_try_eval_success(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.try_eval("EML: eml_scalar(42.0) — the answer")
        assert result == pytest.approx(42.0, rel=1e-12)

    def test_try_eval_returns_none_on_error(self):
        ev = EMLEvaluator(CONTEXT)
        result = ev.try_eval("not an eml string")
        assert result is None

    def test_try_eval_returns_none_on_unknown_vec(self):
        ev = EMLEvaluator(CONTEXT, strict=True)
        result = ev.try_eval("EML: eml_vec('no.such.param') — missing")
        assert result is None


# ---------------------------------------------------------------------------
# eml_eval convenience function
# ---------------------------------------------------------------------------

class TestEmlEval:
    def test_eml_eval_basic(self):
        result = eml_eval("EML: eml_scalar(2.0) — two", {})
        assert result == pytest.approx(2.0, rel=1e-12)

    def test_eml_eval_with_context(self):
        ctx = {"x": 10.0}
        result = eml_eval("EML: ops.mul(eml_vec('x'), eml_scalar(3.0)) — x*3", ctx)
        assert result == pytest.approx(30.0, rel=1e-12)

    def test_eml_eval_strict_default(self):
        with pytest.raises(KeyError):
            eml_eval("EML: eml_vec('missing') — missing", {})

    def test_eml_eval_non_strict(self):
        result = eml_eval("EML: eml_vec('missing') — missing", {}, strict=False)
        assert result == pytest.approx(0.0, abs=1e-12)


# ---------------------------------------------------------------------------
# Standalone eml_scalar / eml_pi / eml_vec from operators module
# ---------------------------------------------------------------------------

class TestOperatorsLiterals:
    def test_eml_scalar_standalone(self):
        assert eml_scalar(5.0).tension() == pytest.approx(5.0, rel=1e-12)

    def test_eml_pi_standalone(self):
        assert eml_pi().tension() == pytest.approx(math.pi, rel=1e-14)

    def test_eml_vec_raises_without_context(self):
        with pytest.raises(KeyError, match="no value context bound"):
            eml_vec("some.param")

    def test_eml_scalar_in_ops_expression(self):
        result = ops.mul(eml_scalar(3.0), eml_scalar(4.0)).tension()
        assert result == pytest.approx(12.0, rel=1e-12)

    def test_eml_pi_in_ops_expression(self):
        result = ops.sin(eml_pi()).tension()
        assert result == pytest.approx(0.0, abs=1e-14)
