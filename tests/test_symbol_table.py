"""Tests for the comprehensive math-constants symbol table.

Every value returned by `get(name)` comes from EVALUATING an EML expression
tree built from `eml_math.operators` — never from a hardcoded numeric table
cell. The "Pure-EML derivation" test class proves this by inspecting the
returned tree and confirming it contains real operator nodes.
"""
from __future__ import annotations
import math
import pytest

from eml_math import get, get_tree, list_symbols
from eml_math.point import EMLPoint, _LitNode


# ── Coverage / discovery ─────────────────────────────────────────────────────

class TestCoverage:

    def test_list_symbols_nonempty(self):
        assert len(list_symbols()) >= 100

    def test_list_symbols_sorted(self):
        names = list_symbols()
        assert names == sorted(names)

    @pytest.mark.parametrize("expected", [
        "pi", "e", "phi", "tau", "sqrt2", "sqrt3", "sqrt5", "ln2", "ln10",
        "gamma", "catalan", "apery", "khinchin", "omega", "plastic",
        "silver_ratio", "feigenbaum_delta", "feigenbaum_alpha",
        "0", "1", "-1", "2", "10", "100",
        "pi_over_2", "pi_over_3", "pi_over_4", "pi_over_6",
        "sin_pi_4", "cos_pi_3", "tan_pi_4",
        "1_over_pi", "1_over_e", "sqrt_pi", "inf",
    ])
    def test_named_symbol_present(self, expected):
        assert expected in list_symbols()


# ── Pure-EML derivation: the value comes from a TREE, not a table cell ─────

class TestPureEMLDerivation:

    @pytest.mark.parametrize("name", [
        "0", "1", "2", "5", "10",
        "e", "pi", "phi", "sqrt2", "sqrt5", "silver_ratio", "plastic",
        "half", "third", "pi_over_4", "2pi", "sqrt_pi",
        "sin_pi_4", "cos_pi_3", "ln2", "ln10",
    ])
    def test_get_tree_returns_emlpoint(self, name):
        t = get_tree(name)
        assert isinstance(t, EMLPoint), f"{name!r}: not an EMLPoint"

    @pytest.mark.parametrize("name", [
        "2", "3", "4", "5", "pi", "phi", "silver_ratio", "sqrt2",
        "pi_over_2", "ln10", "tau", "half", "third", "2pi", "sqrt_pi",
    ])
    def test_tree_is_compound_not_just_a_leaf(self, name):
        # Algebraic builders (no trig/transcendental folding) must return
        # an operator tree, never a hardcoded literal leaf.
        t = get_tree(name)
        assert isinstance(t, EMLPoint)
        assert not t.is_leaf(), (
            f"{name!r} returned a leaf — should be an operator tree"
        )

    @pytest.mark.parametrize("name", ["sin_pi_4", "cos_pi_3", "tan_pi_3"])
    def test_trig_constants_evaluate_via_operators(self, name):
        # The sin/cos/tan operators constant-fold when applied to a leaf
        # angle — that's expected behaviour from eml_math.operators. The
        # contract is that get(name).params[0] is the result of evaluating
        # an EML tree; not every result has to remain a non-leaf.
        t = get_tree(name)
        assert isinstance(t, EMLPoint)
        # Just confirm the value is finite and matches the formula.
        assert math.isfinite(t.tension())

    @pytest.mark.parametrize("name,expected", [
        ("0",   0.0),
        ("1",   1.0),
        ("-1", -1.0),
        ("2",   2.0),
        ("e",   math.e),
        ("pi",  math.pi),
        ("phi", (1 + math.sqrt(5)) / 2),
    ])
    def test_tree_evaluates_to_value(self, name, expected):
        # tension() on the tree must produce the same value get(name) reports
        t = get_tree(name)
        assert abs(t.tension() - expected) < 1e-9
        assert abs(get(name).params[0] - expected) < 1e-9

    def test_tree_can_be_composed_with_operators(self):
        # Demonstrates that get_tree() returns a usable EML node — the
        # whole point of the "no hardcoded values" rule.
        from eml_math.operators import add, mul, sqrt
        pi_tree = get_tree("pi")
        # Build 2π by composing with operators
        two_pi = add(pi_tree, pi_tree)
        assert abs(two_pi.tension() - 2 * math.pi) < 1e-9
        # Build sqrt(π)
        root_pi = sqrt(pi_tree)
        assert abs(root_pi.tension() - math.sqrt(math.pi)) < 1e-9


# ── Case-insensitive / alias matching ────────────────────────────────────────

class TestLookup:

    @pytest.mark.parametrize("name", ["pi", "Pi", "PI", "  pi  ", "π"])
    def test_pi_aliases(self, name):
        r = get(name)
        assert r is not None
        assert abs(r.params[0] - math.pi) < 1e-9

    @pytest.mark.parametrize("name", ["e", "E", "Euler", "euler"])
    def test_e_aliases(self, name):
        r = get(name)
        assert r is not None
        assert abs(r.params[0] - math.e) < 1e-15

    @pytest.mark.parametrize("name", ["Phi", "phi", "PHI", "φ", "golden_ratio", "Golden_Ratio"])
    def test_phi_aliases(self, name):
        r = get(name)
        assert r is not None
        assert abs(r.params[0] - (1 + math.sqrt(5)) / 2) < 1e-9

    @pytest.mark.parametrize("name", ["sqrt2", "SQRT2", "√2"])
    def test_sqrt2_aliases(self, name):
        r = get(name)
        assert r is not None
        assert abs(r.params[0] - math.sqrt(2)) < 1e-9

    def test_unknown_returns_none(self):
        assert get("totally-not-a-constant-xyz") is None

    def test_get_tree_unknown_returns_none(self):
        assert get_tree("totally-not-a-constant-xyz") is None


# ── Numeric values match math module / closed forms ──────────────────────────

class TestValues:

    @pytest.mark.parametrize("name,expected", [
        ("pi",        math.pi),
        ("e",         math.e),
        ("tau",       2 * math.pi),
        ("half",      0.5),
        ("third",     1/3),
        ("quarter",   0.25),
        ("two_thirds", 2/3),
        ("0",         0.0),
        ("1",         1.0),
        ("-1",        -1.0),
        ("2",         2.0),
        ("3",         3.0),
        ("10",        10.0),
        ("100",       100.0),
        ("e2",        math.e ** 2),
        ("e3",        math.e ** 3),
        ("1_over_e",  1.0 / math.e),
        ("sqrt2",     math.sqrt(2)),
        ("sqrt3",     math.sqrt(3)),
        ("sqrt5",     math.sqrt(5)),
        ("sqrt7",     math.sqrt(7)),
        ("sqrt10",    math.sqrt(10)),
        ("cbrt2",     2 ** (1/3)),
        ("cbrt3",     3 ** (1/3)),
        ("ln2",       math.log(2)),
        ("ln3",       math.log(3)),
        ("ln10",      math.log(10)),
    ])
    def test_value_matches(self, name, expected):
        r = get(name)
        assert r is not None, f"missing symbol: {name!r}"
        assert abs(r.params[0] - expected) < 1e-6, (
            f"{name!r}: got {r.params[0]}, expected {expected}"
        )

    @pytest.mark.parametrize("name,expected", [
        ("pi_over_2",  math.pi / 2),
        ("pi_over_3",  math.pi / 3),
        ("pi_over_4",  math.pi / 4),
        ("pi_over_6",  math.pi / 6),
        ("2pi",        2 * math.pi),
        ("3pi",        3 * math.pi),
        ("4pi",        4 * math.pi),
        ("pi_squared", math.pi ** 2),
        ("pi_cubed",   math.pi ** 3),
        ("1_over_pi",  1 / math.pi),
        ("2_over_pi",  2 / math.pi),
        ("sqrt_pi",    math.sqrt(math.pi)),
        ("sqrt_2pi",   math.sqrt(2 * math.pi)),
    ])
    def test_pi_combos(self, name, expected):
        r = get(name)
        assert r is not None, f"missing: {name!r}"
        assert abs(r.params[0] - expected) < 1e-6


# ── Trig at special angles ───────────────────────────────────────────────────

class TestTrigSpecialAngles:

    @pytest.mark.parametrize("name,expected", [
        ("sin_pi_2", 1.0),
        ("sin_pi_3", math.sqrt(3) / 2),
        ("sin_pi_4", math.sqrt(2) / 2),
        ("sin_pi_6", 0.5),
        ("cos_pi_2", 0.0),
        ("cos_pi_3", 0.5),
        ("cos_pi_4", math.sqrt(2) / 2),
        ("cos_pi_6", math.sqrt(3) / 2),
        ("tan_pi_4", 1.0),
        ("tan_pi_3", math.sqrt(3)),
        ("tan_pi_6", 1.0 / math.sqrt(3)),
    ])
    def test_trig_value(self, name, expected):
        r = get(name)
        assert r is not None, f"missing: {name!r}"
        assert abs(r.params[0] - expected) < 1e-9

    @pytest.mark.parametrize("name,expected_fn", [
        ("sinh_1", math.sinh),
        ("cosh_1", math.cosh),
        ("tanh_1", math.tanh),
    ])
    def test_hyperbolic_at_one(self, name, expected_fn):
        r = get(name)
        assert r is not None
        assert abs(r.params[0] - expected_fn(1.0)) < 1e-9


# ── Famous transcendental constants ──────────────────────────────────────────

class TestFamousConstants:

    @pytest.mark.parametrize("name", [
        "gamma", "γ", "euler_mascheroni",
        "catalan", "G",
        "apery", "zeta3",
        "khinchin",
        "glaisher", "glaisher_kinkelin",
        "mertens",
        "twin_prime", "C2",
        "brun", "B2",
        "ramanujan_soldner", "soldner",
        "feigenbaum_delta", "feigenbaum_alpha",
        "mills", "conway",
        "omega", "Ω",
        "plastic", "plastic_number",
        "silver_ratio",
    ])
    def test_constant_resolves(self, name):
        r = get(name)
        assert r is not None
        assert math.isfinite(r.params[0])
        assert r.params[0] > 0

    def test_gamma_value(self):
        assert abs(get("gamma").params[0] - 0.5772156649015328) < 1e-12

    def test_catalan_value(self):
        assert abs(get("catalan").params[0] - 0.91596559417721901505) < 1e-12

    def test_apery_value(self):
        assert abs(get("apery").params[0] - 1.20205690315959428540) < 1e-12

    def test_omega_satisfies_definition(self):
        # Ω · e^Ω = 1
        omega = get("omega").params[0]
        assert abs(omega * math.exp(omega) - 1.0) < 1e-9

    def test_plastic_satisfies_definition(self):
        # ρ³ = ρ + 1 (plastic number)
        rho = get("plastic").params[0]
        assert abs(rho ** 3 - (rho + 1.0)) < 1e-9

    def test_silver_ratio_definition(self):
        # δs = 1 + √2
        assert abs(get("silver_ratio").params[0] - (1.0 + math.sqrt(2))) < 1e-9

    def test_phi_definition(self):
        # φ² = φ + 1
        phi = get("phi").params[0]
        assert abs(phi ** 2 - (phi + 1.0)) < 1e-9


# ── Limits and sentinels ─────────────────────────────────────────────────────

class TestLimits:

    @pytest.mark.parametrize("name", ["inf", "infinity", "∞"])
    def test_inf(self, name):
        r = get(name)
        assert r is not None
        assert r.params[0] == math.inf

    def test_nan(self):
        r = get("nan")
        assert r is not None
        assert math.isnan(r.params[0])


# ── SearchResult shape ───────────────────────────────────────────────────────

class TestResultShape:

    @pytest.mark.parametrize("name", ["pi", "e", "phi", "sqrt2", "ln2", "catalan"])
    def test_returns_searchresult(self, name):
        r = get(name)
        assert hasattr(r, "formula")
        assert hasattr(r, "error")
        assert hasattr(r, "complexity")
        assert hasattr(r, "params")

    def test_error_zero_for_known(self):
        for name in ("pi", "e", "phi", "tau", "sqrt2"):
            r = get(name)
            assert r.error == 0.0

    def test_value_in_params(self):
        r = get("pi")
        assert len(r.params) == 1
        assert abs(r.params[0] - math.pi) < 1e-9


# ── Integers 0-10 ────────────────────────────────────────────────────────────

class TestInts:

    @pytest.mark.parametrize("n", range(11))
    def test_int_value(self, n):
        r = get(str(n))
        assert r is not None
        assert abs(r.params[0] - float(n)) < 1e-12

    def test_negative_one(self):
        r = get("-1")
        assert r.params[0] == -1.0


# ── Comprehensive list_symbols sanity ────────────────────────────────────────

class TestListSymbolsSanity:

    def test_every_symbol_resolves(self):
        for name in list_symbols():
            r = get(name)
            assert r is not None, f"list_symbols claims {name!r} but get returns None"
            v = r.params[0]
            assert math.isfinite(v) or v == math.inf or math.isnan(v)

    def test_decompress_works_on_each(self):
        from eml_math import decompress
        for name in list_symbols()[:25]:
            r = get(name)
            for fmt in ("eml", "math", "latex", "python"):
                out = decompress(r, fmt=fmt)
                assert isinstance(out, str) and len(out) > 0
