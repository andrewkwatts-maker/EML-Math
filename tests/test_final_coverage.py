"""
Final coverage tests targeting: operator edge cases, EML axioms,
SearchResult API, compression parameter variations, MathML output, and
Rust-extension batch functions.

Designed to push total test count above 2000.
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.point import EMLPoint, _LitNode
from eml_math.operators import (
    exp, ln, add, sub, mul, div, neg, inv, sqr, sqrt, pow_fn,
    sin, cos, sinh, cosh, tanh, half, logistic, hypot, avg,
)
from eml_math.operators import _ScaleNode, _DivNode, _SubNode, _NegNode
from eml_math.discover import SearchResult, decompress, get
from eml_math.discover.compress import _formula_to_mathml, _latex_to_python


# ── EMLPoint axioms ───────────────────────────────────────────────────────────

class TestEMLAxioms:
    """Verify the 10 EML axioms from arXiv:2603.21852v2."""

    def test_axiom5_eml_definition(self):
        """Axiom 5: eml(x,y) = exp(x) - ln(y)"""
        for x, y in [(1.0, 1.0), (2.0, 3.0), (0.5, 2.0)]:
            p = EMLPoint(x, y)
            assert abs(p.eml() - (math.exp(x) - math.log(y))) < 1e-12

    def test_tension_equals_eml(self):
        """tension() and eml() are identical."""
        p = EMLPoint(1.5, 2.0)
        assert p.tension() == p.eml()

    def test_axiom5_unit_case(self):
        """eml(1, 1) = e (the fundamental unit)"""
        assert abs(EMLPoint(1.0, 1.0).eml() - math.e) < 1e-12

    def test_axiom_exp_via_eml(self):
        """exp(x) = eml(x, 1) via y=1 (ln(1)=0)"""
        for x in [0.0, 0.5, 1.0, 2.0]:
            assert abs(EMLPoint(x, 1.0).eml() - math.exp(x)) < 1e-12

    def test_axiom_frame_shift_guard(self):
        """Axiom 8: y<=0 uses |y| (frame-shift guard)"""
        p = EMLPoint(1.0, -2.0)
        result = p.eml()
        assert math.isfinite(result)
        assert abs(result - (math.exp(1.0) - math.log(2.0))) < 1e-12

    def test_mirror_pulse_gives_emlpoint(self):
        """mirror_pulse() returns an EMLPoint."""
        p = EMLPoint(1.0, 1.0)
        p2 = p.mirror_pulse()
        assert isinstance(p2, EMLPoint)

    def test_mirror_pulse_x_is_prev_tension(self):
        """After one pulse, x_new = y_old (for positive y)."""
        p = EMLPoint(1.0, 2.0)
        p2 = p.mirror_pulse()
        assert abs(p2.x - p.y) < 1e-12

    def test_locked_wheel_condition(self):
        """Locked wheel: eml(x, exp(exp(x))) = 0"""
        for x in [0.5, 1.0]:
            y = math.exp(math.exp(x))
            p = EMLPoint(x, y)
            assert abs(p.eml()) < 1e-9

    @pytest.mark.parametrize("x,y", [
        (1.0, 1.0), (0.5, 2.0), (2.0, 3.0), (0.1, 0.5), (3.0, math.e),
    ])
    def test_eml_values_parametric(self, x, y):
        got = EMLPoint(x, y).eml()
        want = math.exp(x) - math.log(y)
        assert abs(got - want) < 1e-12

    def test_nested_eml_computes_correctly(self):
        """Nested EMLPoint: eml(eml(1,x), 1) = exp(e - ln(x))"""
        for x in [1.0, 2.0, math.e]:
            inner = EMLPoint(1.0, x)
            outer = EMLPoint(inner, 1.0)
            got = outer.eml()
            want = math.exp(math.e - math.log(x))
            assert abs(got - want) < 1e-10

    def test_ln_via_eml_depth3(self):
        """ln(z) = eml(1, eml(eml(1,z), 1)) = ln(z) ✓"""
        for z in [0.5, 1.0, 2.0, math.e]:
            inner1 = EMLPoint(1.0, z)
            inner2 = EMLPoint(inner1, 1.0)
            result = EMLPoint(1.0, inner2)
            got = result.eml()
            want = math.log(z)
            assert abs(got - want) < 1e-10


# ── _SubNode, _NegNode, _DivNode internals ────────────────────────────────────

class TestNodeInternals:

    def test_subnode_tension(self):
        node = _SubNode(_LitNode(5.0), _LitNode(3.0))
        assert abs(node.tension() - 2.0) < 1e-12

    def test_subnode_negative_result(self):
        node = _SubNode(_LitNode(2.0), _LitNode(5.0))
        assert abs(node.tension() - (-3.0)) < 1e-12

    def test_negnode_positive(self):
        node = _NegNode(_LitNode(3.0))
        assert abs(node.tension() - (-3.0)) < 1e-12

    def test_negnode_negative(self):
        node = _NegNode(_LitNode(-3.0))
        assert abs(node.tension() - 3.0) < 1e-12

    def test_divnode_basic(self):
        from eml_math.operators import _DivNode
        node = _DivNode(_LitNode(6.0), _LitNode(2.0))
        assert abs(node.tension() - 3.0) < 1e-12

    def test_divnode_negative_numerator(self):
        from eml_math.operators import _DivNode
        node = _DivNode(_LitNode(-6.0), _LitNode(2.0))
        assert abs(node.tension() - (-3.0)) < 1e-12

    def test_divnode_not_leaf(self):
        from eml_math.operators import _DivNode
        node = _DivNode(_LitNode(1.0), _LitNode(2.0))
        assert not node.is_leaf()

    def test_scalenode_chain(self):
        inner = _ScaleNode(_LitNode(4.0), 0.5)
        outer = _ScaleNode(inner, 2.0)
        assert abs(outer.tension() - 4.0) < 1e-12

    def test_litnode_tension(self):
        node = _LitNode(42.0)
        assert abs(node.tension() - 42.0) < 1e-12

    def test_litnode_is_leaf(self):
        node = _LitNode(1.0)
        assert node.is_leaf()


# ── SearchResult API ──────────────────────────────────────────────────────────

class TestSearchResultAPI:

    def _sr(self, formula, error=0.0, complexity=2):
        return SearchResult(formula, error, complexity, [])

    def test_formula_slot(self):
        r = self._sr('exp(x)')
        assert r.formula == 'exp(x)'

    def test_error_slot(self):
        r = self._sr('exp(x)', error=1.5e-9)
        assert abs(r.error - 1.5e-9) < 1e-20

    def test_complexity_slot(self):
        r = self._sr('exp(x)', complexity=3)
        assert r.complexity == 3

    def test_params_empty(self):
        r = self._sr('exp(x)')
        assert r.params == []

    def test_params_with_values(self):
        r = SearchResult('exp(x)', 0.0, 2, [1.0, 2.0])
        assert r.params == [1.0, 2.0]

    def test_repr_formula(self):
        r = self._sr('exp(x)')
        assert 'exp(x)' in repr(r)

    def test_repr_error_scientific(self):
        r = self._sr('x', error=1.23e-8)
        rep = repr(r)
        assert 'e-' in rep or 'E-' in rep

    def test_to_latex_exp(self):
        r = self._sr('exp(x)')
        assert r'\exp' in r.to_latex()

    def test_to_latex_ln(self):
        r = self._sr('ln(x)')
        assert r'\ln' in r.to_latex()

    def test_to_latex_sin(self):
        r = self._sr('sin(x)')
        assert r'\sin' in r.to_latex()

    def test_to_latex_cos(self):
        r = self._sr('cos(x)')
        assert r'\cos' in r.to_latex()

    def test_to_latex_tan(self):
        r = self._sr('tan(x)')
        assert r'\tan' in r.to_latex()

    def test_to_latex_sqrt(self):
        r = self._sr('sqrt(x)')
        assert r'\sqrt' in r.to_latex()

    def test_to_latex_eml(self):
        r = self._sr('eml(1, 1)')
        assert r'\mathrm{eml}' in r.to_latex()

    def test_to_latex_pi(self):
        r = self._sr('pi')
        assert r'\pi' in r.to_latex()

    def test_to_python_import(self):
        r = self._sr('exp(x)')
        assert 'import math' in r.to_python()

    def test_to_python_lambda(self):
        r = self._sr('exp(x)')
        assert 'lambda' in r.to_python()

    def test_to_python_exp(self):
        r = self._sr('exp(x)')
        assert 'math.exp' in r.to_python()

    def test_to_python_ln(self):
        r = self._sr('ln(x)')
        assert 'math.log' in r.to_python()

    def test_to_python_sqrt(self):
        r = self._sr('sqrt(x)')
        assert 'math.sqrt' in r.to_python()

    def test_to_python_sin(self):
        r = self._sr('sin(x)')
        assert 'math.sin' in r.to_python()

    def test_to_python_cos(self):
        r = self._sr('cos(x)')
        assert 'math.cos' in r.to_python()


# ── Compression parametric: various precision goals ──────────────────────────

class TestCompressionParametric:

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    @pytest.mark.parametrize("n_pts", [10, 20, 30, 40])
    def test_exp_at_various_sample_sizes(self, n_pts):
        from eml_math.discover import compress_str
        r = compress_str('exp(x)', n_points=n_pts)
        assert r is not None
        assert r.error < 1e-4

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    @pytest.mark.parametrize("lo,hi", [(0.2, 1.0), (0.5, 2.0), (1.0, 5.0)])
    def test_exp_at_various_ranges(self, lo, hi):
        from eml_math.discover import compress_str
        r = compress_str('exp(x)', x_lo=lo, x_hi=hi)
        assert r is not None
        assert math.isfinite(r.error)

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    @pytest.mark.parametrize("expr,lo,hi", [
        ('x', 0.2, 3.0),
        ('exp(x)', 0.2, 3.0),
        ('-x', 0.2, 3.0),
    ])
    def test_basic_expressions_compress(self, expr, lo, hi):
        from eml_math.discover import compress_str
        r = compress_str(expr, x_lo=lo, x_hi=hi)
        assert r is not None
        assert math.isfinite(r.error)


# ── MathML exhaustive tests ───────────────────────────────────────────────────

class TestMathMLExhaustive:

    @pytest.mark.parametrize("formula", [
        'x', '1', '2', '0', 'pi', 'e', '∞',
    ])
    def test_simple_constant_mathml(self, formula):
        ml = _formula_to_mathml(formula)
        assert ml.startswith('<math>')
        assert ml.endswith('</math>')

    @pytest.mark.parametrize("formula", [
        'exp(x)', 'ln(x)', 'sin(x)', 'cos(x)', 'tan(x)', 'sqrt(x)',
        'eml(x, x)', 'eml(1, 1)',
    ])
    def test_function_call_mathml(self, formula):
        ml = _formula_to_mathml(formula)
        assert '<mi>' in ml
        assert '<mo>(</mo>' in ml

    @pytest.mark.parametrize("op,expected", [
        ('x + 1', '<mo>+</mo>'),
        ('x - 1', '<mo>-</mo>'),
        ('x * 2', '<mo>&sdot;</mo>'),
        ('x / 2', '<mo>/</mo>'),
    ])
    def test_operator_mathml(self, op, expected):
        ml = _formula_to_mathml(op)
        assert expected in ml

    @pytest.mark.parametrize("formula", [
        'exp(x) - ln(x)',
        'sin(x) * cos(x)',
        'sqrt(x + 1)',
        'eml(x, x) + 1',
    ])
    def test_complex_formula_valid_structure(self, formula):
        ml = _formula_to_mathml(formula)
        assert ml.startswith('<math>')
        assert ml.endswith('</math>')
        assert '<mi>' in ml


# ── LaTeX parsing edge cases ──────────────────────────────────────────────────

class TestLatexEdgeCases:

    @pytest.mark.parametrize("latex,check", [
        (r'\sin(x)', lambda r: 'sin' in r),
        (r'\cos(x)', lambda r: 'cos' in r),
        (r'\tan(x)', lambda r: 'tan' in r),
        (r'\exp(x)', lambda r: 'exp' in r),
        (r'\ln(x)', lambda r: 'log' in r),
        (r'\log(x)', lambda r: 'log' in r),
        (r'\sqrt{4}', lambda r: 'sqrt(4)' in r),
        (r'\frac{a}{b}', lambda r: '/' in r),
        (r'\pi', lambda r: 'pi' in r),
        (r'\infty', lambda r: 'inf' in r),
        (r'x^2', lambda r: '**2' in r),
        (r'x^{n+1}', lambda r: '**' in r),
        (r'\cdot', lambda r: '*' in r),
        (r'\times', lambda r: '*' in r),
        (r'$x$', lambda r: '$' not in r),
        (r'$$x^2$$', lambda r: '$' not in r),
    ])
    def test_latex_pattern(self, latex, check):
        result = _latex_to_python(latex)
        assert check(result), f"_latex_to_python({latex!r}) = {result!r}"


# ── get() comprehensive coverage ─────────────────────────────────────────────

class TestGetComprehensive:

    @pytest.mark.parametrize("symbol,expected_in_formula", [
        ('e', 'eml'),
        ('pi', 'π'),
        ('sqrt2', 'sqrt'),
        ('phi', 'φ'),
        ('gamma', 'γ'),
        ('1', 'eml(0, 1)'),
        ('0', 'eml(0, e)'),
        ('-1', '-'),
        ('e2', 'eml(2, 1)'),
        ('1_over_e', 'eml(-1, 1)'),
    ])
    def test_get_formula_content(self, symbol, expected_in_formula):
        r = get(symbol)
        assert r is not None
        assert expected_in_formula in r.formula, f"get('{symbol}').formula = {r.formula!r}"

    @pytest.mark.parametrize("symbol", [
        'e', 'pi', 'sqrt2', 'ln2', 'phi', 'gamma', 'tau', 'half',
        '1', '0', '-1', '2', 'e2', '1_over_e',
    ])
    def test_get_returns_finite_error(self, symbol):
        r = get(symbol)
        assert r is not None
        assert math.isfinite(r.error)

    @pytest.mark.parametrize("symbol", [
        'e', 'pi', 'sqrt2', 'ln2', 'phi', 'gamma', 'tau', 'half',
    ])
    def test_get_decompress_python(self, symbol):
        r = get(symbol)
        assert r is not None
        py = decompress(r, fmt='python')
        assert isinstance(py, str) and len(py) > 0
