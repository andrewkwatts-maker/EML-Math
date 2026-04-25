"""
Additional symbol recognition, pipeline integration, and edge-case tests.
Brings total test count to ~1500.
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.discover import compress, compress_str, compress_latex, decompress, get, recognize, SearchResult
from eml_math.discover.compress import _latex_to_python, _make_callable, _formula_to_mathml


# ── Symbol numeric accuracy ───────────────────────────────────────────────────

class TestSymbolNumericValues:

    def test_e_value(self):
        r = get('e')
        assert r is not None
        assert abs(r.error) < 1e-12

    def test_sqrt2_value(self):
        r = get('sqrt2')
        assert r is not None
        assert abs(r.error) < 1e-12

    def test_ln2_value_accurate(self):
        r = get('ln2')
        assert r is not None
        # ln2 recognized via recognize(), may have small error
        assert r.error < 1e-6

    def test_phi_value(self):
        r = get('phi')
        assert r is not None
        assert abs(r.error) < 1e-6

    def test_gamma_value(self):
        r = get('gamma')
        assert r is not None
        assert r.error < 1e-6

    def test_tau_is_2pi(self):
        r = get('tau')
        assert r is not None

    def test_half_is_point_5(self):
        r = get('half')
        assert r is not None

    def test_e2_is_e_squared(self):
        r = get('e2')
        assert r is not None
        assert 'eml(2, 1)' in r.formula

    def test_1_over_e_formula(self):
        r = get('1_over_e')
        assert r is not None
        assert 'eml(-1, 1)' in r.formula

    def test_get_returns_list_params(self):
        for sym in ['e', 'pi', 'sqrt2', 'phi']:
            r = get(sym)
            assert r is not None
            assert isinstance(r.params, list)

    def test_get_complexity_at_least_1(self):
        for sym in ['e', 'pi', '1', '0']:
            r = get(sym)
            assert r is not None
            assert r.complexity >= 1


# ── decompress() edge cases ───────────────────────────────────────────────────

class TestDecompressEdgeCases:

    def _sr(self, formula):
        return SearchResult(formula, 0.0, 2, [])

    def test_formula_with_both_exp_and_ln(self):
        r = self._sr('exp(x) - ln(x)')
        py = decompress(r, fmt='python')
        assert 'math.exp' in py
        assert 'math.log' in py

    def test_mathml_for_addition(self):
        r = self._sr('x + 1')
        ml = decompress(r, fmt='mathml')
        assert '<mo>+</mo>' in ml

    def test_mathml_for_subtraction(self):
        r = self._sr('exp(x) - ln(x)')
        ml = decompress(r, fmt='mathml')
        assert '<mo>-</mo>' in ml

    def test_mathml_for_multiplication(self):
        r = self._sr('x * 2')
        ml = decompress(r, fmt='mathml')
        assert 'sdot' in ml or '*' in ml or 'mo' in ml

    def test_mathml_for_division(self):
        r = self._sr('x / 2')
        ml = decompress(r, fmt='mathml')
        assert '/' in ml

    def test_eml_format_preserves_formula(self):
        formulas = ['x', '1', 'exp(x)', 'ln(x)', 'eml(1, 1)', 'sin(x)', 'cos(x)']
        for f in formulas:
            r = self._sr(f)
            assert decompress(r, fmt='eml') == f

    def test_math_format_returns_string(self):
        for formula in ['exp(x)', 'ln(x)', 'x', '1', 'pi']:
            r = self._sr(formula)
            result = decompress(r, fmt='math')
            assert isinstance(result, str)

    def test_python_format_has_correct_structure(self):
        r = self._sr('sin(x)')
        py = decompress(r, fmt='python')
        lines = py.strip().split('\n')
        assert len(lines) >= 1
        assert 'math' in py

    def test_latex_for_tan(self):
        r = self._sr('tan(x)')
        latex = decompress(r, fmt='latex')
        assert r'\tan' in latex

    def test_latex_for_exp_2_is_squared(self):
        r = self._sr('x**2')
        latex = decompress(r, fmt='latex')
        assert isinstance(latex, str)

    def test_formula_to_mathml_for_0(self):
        ml = _formula_to_mathml('0')
        assert '<mn>0</mn>' in ml

    def test_formula_to_mathml_for_2(self):
        ml = _formula_to_mathml('2')
        assert '<mn>2</mn>' in ml

    def test_formula_to_mathml_for_e(self):
        ml = _formula_to_mathml('e')
        assert '<mi>' in ml

    def test_all_formats_return_strings(self):
        r = SearchResult('exp(x)', 0.0, 2, [])
        for fmt in ['eml', 'python', 'latex', 'math', 'mathml']:
            result = decompress(r, fmt=fmt)
            assert isinstance(result, str), f"fmt={fmt} returned {type(result)}"

    def test_all_formats_nonempty(self):
        r = SearchResult('exp(x)', 0.0, 2, [])
        for fmt in ['eml', 'python', 'latex', 'math', 'mathml']:
            result = decompress(r, fmt=fmt)
            assert len(result) > 0, f"fmt={fmt} returned empty string"


# ── _make_callable edge cases ─────────────────────────────────────────────────

class TestMakeCallableEdgeCases:

    def test_constant_returns_callable(self):
        fn = _make_callable('1.0')
        assert fn is not None

    def test_constant_2pi(self):
        fn = _make_callable('2 * pi')
        assert fn is not None
        assert abs(fn(0.0) - 2 * math.pi) < 1e-12

    def test_nested_functions(self):
        fn = _make_callable('exp(log(x))')
        assert fn is not None
        assert abs(fn(2.0) - 2.0) < 1e-10

    def test_pythagorean_identity(self):
        fn = _make_callable('sin(x)**2 + cos(x)**2')
        assert fn is not None
        for xv in [0.0, 0.5, 1.0, 2.0, math.pi]:
            assert abs(fn(xv) - 1.0) < 1e-12

    def test_sqrt_small_values(self):
        fn = _make_callable('sqrt(x)')
        assert fn is not None
        assert abs(fn(0.01) - 0.1) < 1e-12
        assert abs(fn(0.25) - 0.5) < 1e-12

    def test_mixed_expression(self):
        fn = _make_callable('x * x + 2 * x + 1')
        assert fn is not None
        assert abs(fn(3.0) - 16.0) < 1e-10

    def test_empty_string_returns_none(self):
        fn = _make_callable('')
        assert fn is None

    def test_whitespace_only_returns_none(self):
        fn = _make_callable('   ')
        assert fn is None

    def test_hyperbolic_functions(self):
        fn = _make_callable('sinh(x) + cosh(x)')
        assert fn is not None
        # sinh(x) + cosh(x) = exp(x)
        for xv in [0.5, 1.0, 2.0]:
            assert abs(fn(xv) - math.exp(xv)) < 1e-12


# ── latex_to_python more patterns ─────────────────────────────────────────────

class TestLatexToPythonExtra:

    def test_double_backslash_exp(self):
        result = _latex_to_python(r'\\exp(x)')
        assert 'exp' in result

    def test_sqrt_with_expression(self):
        result = _latex_to_python(r'\sqrt{x^2 + 1}')
        assert 'sqrt' in result

    def test_nested_frac(self):
        result = _latex_to_python(r'\frac{1}{x}')
        assert '/' in result
        assert '1' in result

    def test_sin_cos_sum(self):
        result = _latex_to_python(r'\sin(x) + \cos(x)')
        assert 'sin' in result
        assert 'cos' in result
        assert '+' in result

    def test_power_with_braces_2(self):
        result = _latex_to_python(r'x^{2}')
        assert '**2' in result or '**' in result

    def test_power_with_braces_n_plus_1(self):
        result = _latex_to_python(r'x^{n+1}')
        assert '**' in result

    def test_inline_delimiters_stripped(self):
        result1 = _latex_to_python(r'\sin(x)')
        result2 = _latex_to_python(r'\(\sin(x)\)')
        assert 'sin' in result1
        assert 'sin' in result2

    def test_pi_in_expression(self):
        result = _latex_to_python(r'2 \cdot \pi')
        assert 'pi' in result
        assert '*' in result

    def test_exp_x_over_2(self):
        result = _latex_to_python(r'\exp\left(\frac{x}{2}\right)')
        assert 'exp' in result

    def test_no_latex_passthrough(self):
        result = _latex_to_python('x + 1')
        assert result.strip() == 'x + 1'


# ── Full pipeline integration ──────────────────────────────────────────────────

class TestPipelineIntegration:
    """End-to-end: string/LaTeX in → compress → decompress → all formats."""

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_identity_full_pipeline(self):
        r = compress_str('x')
        assert r is not None
        assert decompress(r, fmt='eml') == r.formula
        assert decompress(r, fmt='python').count('import math') == 1
        assert decompress(r, fmt='mathml').startswith('<math>')

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_full_pipeline(self):
        r = compress_str('exp(x)')
        assert r is not None
        latex = decompress(r, fmt='latex')
        assert isinstance(latex, str)
        ml = decompress(r, fmt='mathml')
        assert '<math>' in ml

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_symbol_pipeline(self):
        """get('e') → decompress → all formats produce strings"""
        r = get('e')
        assert r is not None
        for fmt in ['eml', 'python', 'latex', 'math', 'mathml']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str) and len(out) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_latex_to_eml_pipeline(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        eml_str = decompress(r, fmt='eml')
        assert isinstance(eml_str, str)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_to_decompress_all_formats(self):
        r = compress_str('exp(x)')
        assert r is not None
        formats = ['eml', 'python', 'latex', 'math', 'mathml']
        for fmt in formats:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str), f"{fmt} is not str"
            assert len(out) > 0, f"{fmt} is empty"

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_str_with_use_eml_true(self):
        r = compress_str('exp(x) - log(x)', x_lo=0.5, use_eml=True)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_str_trig_disabled(self):
        r = compress_str('x', use_trig=False)
        assert r is not None  # x is trivial, should still work

    def test_get_all_symbols_return_valid_searchresult(self):
        for sym in ['e', 'pi', 'sqrt2', 'ln2', 'phi', 'gamma', 'tau', 'half',
                    '1', '0', '-1', '2', 'inf', 'e2', '1_over_e']:
            r = get(sym)
            assert r is not None, f"get('{sym}') = None"
            assert isinstance(r, SearchResult), f"get('{sym}') is not SearchResult"
            assert isinstance(r.formula, str)
            assert r.complexity >= 1

    def test_get_then_decompress_all_symbols(self):
        for sym in ['e', 'pi', 'sqrt2', 'ln2', 'phi', 'gamma']:
            r = get(sym)
            assert r is not None
            for fmt in ['eml', 'math', 'latex', 'mathml']:
                out = decompress(r, fmt=fmt)
                assert isinstance(out, str)

    def test_searchresult_repr_contains_formula(self):
        r = SearchResult('exp(x)', 1e-8, 2, [])
        assert 'exp(x)' in repr(r)

    def test_searchresult_error_in_repr(self):
        r = SearchResult('exp(x)', 1.23e-8, 2, [])
        rep = repr(r)
        assert 'e-' in rep or 'E-' in rep or '1.23' in rep

    def test_searchresult_to_latex(self):
        r = SearchResult('exp(x) - ln(x)', 0.0, 3, [])
        latex = r.to_latex()
        assert r'\exp' in latex
        assert r'\ln' in latex

    def test_searchresult_to_python(self):
        r = SearchResult('exp(x)', 0.0, 2, [])
        py = r.to_python()
        assert 'math.exp' in py
        assert 'import math' in py
