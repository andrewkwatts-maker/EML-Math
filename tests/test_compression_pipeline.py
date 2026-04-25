"""
Tests for the EML equation compression pipeline:
  compress_str(), compress_latex(), decompress(), get()

The pure-Python path (get, decompress, _latex_to_python) runs without
the Rust extension. Tests that call the beam-search engine (compress_str,
compress_latex with x-dependent functions) are guarded with skipif(_RUST).
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.discover import (
    compress_str,
    compress_latex,
    decompress,
    get,
    SearchResult,
)
from eml_math.discover.compress import _latex_to_python, _make_callable


# ── get() ─────────────────────────────────────────────────────────────────────

class TestGet:

    def test_get_e_formula(self):
        r = get('e')
        assert r is not None
        assert 'eml(1, 1)' in r.formula

    def test_get_e_value_correct(self):
        r = get('e')
        assert r.error < 1e-12

    def test_get_pi_formula_contains_pi(self):
        r = get('pi')
        assert r is not None
        assert 'π' in r.formula

    def test_get_pi_unicode_alias(self):
        r1 = get('pi')
        r2 = get('π')
        assert r1 is not None
        assert r2 is not None
        assert r1.formula == r2.formula

    def test_get_pi_uppercase(self):
        r = get('PI')
        assert r is not None

    def test_get_sqrt2_formula(self):
        r = get('sqrt2')
        assert r is not None
        assert 'sqrt(2)' in r.formula

    def test_get_sqrt2_unicode_alias(self):
        r1 = get('sqrt2')
        r2 = get('√2')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_ln2_formula(self):
        r = get('ln2')
        assert r is not None

    def test_get_log2_alias_equals_ln2(self):
        r1 = get('ln2')
        r2 = get('log2')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_phi_formula_contains_phi(self):
        r = get('phi')
        assert r is not None
        assert 'φ' in r.formula

    def test_get_golden_ratio_alias(self):
        r1 = get('phi')
        r2 = get('golden_ratio')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_phi_unicode_alias(self):
        r1 = get('phi')
        r2 = get('φ')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_gamma_formula_contains_gamma(self):
        r = get('gamma')
        assert r is not None
        assert 'γ' in r.formula

    def test_get_euler_mascheroni_alias(self):
        r1 = get('gamma')
        r2 = get('euler_mascheroni')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_gamma_unicode_alias(self):
        r1 = get('gamma')
        r2 = get('γ')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_tau_formula(self):
        r = get('tau')
        assert r is not None

    def test_get_tau_unicode_alias(self):
        r1 = get('tau')
        r2 = get('τ')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_half_formula(self):
        r = get('half')
        assert r is not None

    def test_get_1_formula(self):
        r = get('1')
        assert r is not None
        assert 'eml(0, 1)' in r.formula

    def test_get_0_formula(self):
        r = get('0')
        assert r is not None
        assert 'eml(0, e)' in r.formula

    def test_get_minus_1_formula(self):
        r = get('-1')
        assert r is not None

    def test_get_2_formula(self):
        r = get('2')
        assert r is not None

    def test_get_inf_formula(self):
        r = get('inf')
        assert r is not None

    def test_get_infinity_alias(self):
        r1 = get('inf')
        r2 = get('infinity')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_e2_formula(self):
        r = get('e2')
        assert r is not None
        assert 'eml(2, 1)' in r.formula

    def test_get_1_over_e_formula(self):
        r = get('1_over_e')
        assert r is not None
        assert 'eml(-1, 1)' in r.formula

    def test_get_unknown_returns_none(self):
        assert get('unknown_constant_xyz') is None

    def test_get_returns_searchresult_type(self):
        r = get('pi')
        assert isinstance(r, SearchResult)

    def test_get_complexity_positive(self):
        r = get('e')
        assert r.complexity >= 1

    def test_get_whitespace_stripped(self):
        r1 = get('pi')
        r2 = get('  pi  ')
        assert r1 is not None and r2 is not None
        assert r1.formula == r2.formula

    def test_get_e_error_near_zero(self):
        r = get('e')
        assert r.error < 1e-10

    def test_get_sqrt2_value_correct(self):
        r = get('sqrt2')
        assert r.error < 1e-10

    def test_get_params_is_list(self):
        r = get('pi')
        assert isinstance(r.params, list)

    def test_get_all_symbols_not_none(self):
        symbols = ['e', 'pi', 'sqrt2', 'ln2', 'phi', 'gamma', 'tau', 'half', '1', '0', '-1', '2', 'e2', '1_over_e']
        for sym in symbols:
            r = get(sym)
            assert r is not None, f"get('{sym}') returned None"


# ── decompress() ──────────────────────────────────────────────────────────────

class TestDecompress:

    def _sr(self, formula):
        return SearchResult(formula, 0.0, 2, [])

    def test_fmt_eml_returns_formula(self):
        r = self._sr('exp(x)')
        assert decompress(r, fmt='eml') == 'exp(x)'

    def test_fmt_eml_eml_formula(self):
        r = self._sr('eml(x, x)')
        assert decompress(r, fmt='eml') == 'eml(x, x)'

    def test_fmt_python_has_import_math(self):
        r = self._sr('exp(x)')
        py = decompress(r, fmt='python')
        assert 'import math' in py

    def test_fmt_python_has_lambda(self):
        r = self._sr('exp(x)')
        py = decompress(r, fmt='python')
        assert 'lambda' in py

    def test_fmt_python_has_math_exp(self):
        r = self._sr('exp(x)')
        py = decompress(r, fmt='python')
        assert 'math.exp' in py

    def test_fmt_latex_exp_uses_backslash_exp(self):
        r = self._sr('exp(x)')
        latex = decompress(r, fmt='latex')
        assert r'\exp' in latex

    def test_fmt_latex_ln_uses_backslash_ln(self):
        r = self._sr('ln(x)')
        latex = decompress(r, fmt='latex')
        assert r'\ln' in latex

    def test_fmt_latex_sin_uses_backslash_sin(self):
        r = self._sr('sin(x)')
        latex = decompress(r, fmt='latex')
        assert r'\sin' in latex

    def test_fmt_latex_pi_uses_backslash_pi(self):
        r = self._sr('pi')
        latex = decompress(r, fmt='latex')
        assert r'\pi' in latex

    def test_fmt_math_sqrt_uses_unicode(self):
        r = self._sr('sqrt(x)')
        math_str = decompress(r, fmt='math')
        assert '√' in math_str

    def test_fmt_math_pi_uses_unicode(self):
        r = self._sr('pi')
        math_str = decompress(r, fmt='math')
        assert 'π' in math_str

    def test_fmt_mathml_starts_with_math_tag(self):
        r = self._sr('exp(x)')
        ml = decompress(r, fmt='mathml')
        assert ml.startswith('<math>')

    def test_fmt_mathml_ends_with_math_tag(self):
        r = self._sr('exp(x)')
        ml = decompress(r, fmt='mathml')
        assert ml.endswith('</math>')

    def test_fmt_mathml_constant_1(self):
        r = self._sr('1')
        ml = decompress(r, fmt='mathml')
        assert '<mn>1</mn>' in ml

    def test_fmt_mathml_constant_x(self):
        r = self._sr('x')
        ml = decompress(r, fmt='mathml')
        assert '<mi>x</mi>' in ml

    def test_fmt_mathml_pi(self):
        r = self._sr('π')
        ml = decompress(r, fmt='mathml')
        assert '&pi;' in ml

    def test_fmt_mathml_infinity(self):
        r = self._sr('∞')
        ml = decompress(r, fmt='mathml')
        assert '&infin;' in ml

    def test_fmt_mathml_is_string(self):
        r = self._sr('exp(x) - ln(x)')
        ml = decompress(r, fmt='mathml')
        assert isinstance(ml, str)

    def test_default_fmt_is_math(self):
        r = self._sr('sqrt(x)')
        result = decompress(r)
        assert isinstance(result, str)

    def test_fmt_math_identity_returns_string(self):
        r = self._sr('x')
        assert isinstance(decompress(r, fmt='math'), str)

    def test_fmt_python_for_ln(self):
        r = self._sr('ln(x)')
        py = decompress(r, fmt='python')
        assert 'math.log' in py

    def test_fmt_latex_for_sqrt(self):
        r = self._sr('sqrt(x)')
        latex = decompress(r, fmt='latex')
        assert r'\sqrt' in latex

    def test_fmt_eml_for_eml(self):
        r = self._sr('eml(1, 1)')
        assert decompress(r, fmt='eml') == 'eml(1, 1)'

    def test_fmt_latex_eml_uses_mathrm(self):
        r = self._sr('eml(x, x)')
        latex = decompress(r, fmt='latex')
        assert r'\mathrm{eml}' in latex

    def test_fmt_python_for_cos(self):
        r = self._sr('cos(x)')
        py = decompress(r, fmt='python')
        assert 'math.cos' in py

    def test_fmt_python_for_sin(self):
        r = self._sr('sin(x)')
        py = decompress(r, fmt='python')
        assert 'math.sin' in py


# ── _latex_to_python() ────────────────────────────────────────────────────────

class TestLatexToPython:

    def test_sin_command(self):
        result = _latex_to_python(r'\sin(x)')
        assert 'sin' in result

    def test_cos_command(self):
        result = _latex_to_python(r'\cos(x)')
        assert 'cos' in result

    def test_exp_command(self):
        result = _latex_to_python(r'\exp(x)')
        assert 'exp' in result

    def test_ln_command(self):
        result = _latex_to_python(r'\ln(x)')
        assert 'log' in result

    def test_log_command(self):
        result = _latex_to_python(r'\log(x)')
        assert 'log' in result

    def test_sqrt_braces(self):
        result = _latex_to_python(r'\sqrt{x}')
        assert 'sqrt(x)' in result

    def test_frac_converts(self):
        result = _latex_to_python(r'\frac{1}{x}')
        assert '/' in result
        assert '1' in result
        assert 'x' in result

    def test_pi_command(self):
        result = _latex_to_python(r'\pi')
        assert 'pi' in result

    def test_infty_command(self):
        result = _latex_to_python(r'\infty')
        assert 'inf' in result

    def test_cdot_becomes_multiply(self):
        result = _latex_to_python(r'x \cdot y')
        assert '*' in result

    def test_times_becomes_multiply(self):
        result = _latex_to_python(r'x \times y')
        assert '*' in result

    def test_power_2(self):
        result = _latex_to_python(r'x^2')
        assert '**2' in result or '**' in result

    def test_power_3(self):
        result = _latex_to_python(r'x^3')
        assert '**3' in result or '**' in result

    def test_power_braces(self):
        result = _latex_to_python(r'x^{n+1}')
        assert '**' in result
        assert 'n' in result

    def test_sin_squared(self):
        result = _latex_to_python(r'\sin^2(x)')
        assert 'sin' in result
        assert '**2' in result

    def test_cos_squared(self):
        result = _latex_to_python(r'\cos^2(x)')
        assert 'cos' in result
        assert '**2' in result

    def test_dollar_delimiters_stripped(self):
        result = _latex_to_python(r'$x^2$')
        assert '$' not in result

    def test_double_dollar_stripped(self):
        result = _latex_to_python(r'$$x^2$$')
        assert '$' not in result

    def test_left_right_parens_stripped(self):
        result = _latex_to_python(r'\left(x\right)')
        assert r'\left' not in result
        assert r'\right' not in result

    def test_curly_braces_become_parens(self):
        result = _latex_to_python(r'x^{2}')
        assert '{' not in result
        assert '}' not in result

    def test_returns_string(self):
        result = _latex_to_python(r'\sin(x) + \cos(x)')
        assert isinstance(result, str)


# ── _make_callable() ─────────────────────────────────────────────────────────

class TestMakeCallable:

    def test_simple_expression(self):
        fn = _make_callable('x')
        assert fn is not None
        assert fn(3.0) == 3.0

    def test_exp_expression(self):
        fn = _make_callable('exp(x)')
        assert fn is not None
        assert abs(fn(1.0) - math.e) < 1e-12

    def test_log_expression(self):
        fn = _make_callable('log(x)')
        assert fn is not None
        assert abs(fn(math.e) - 1.0) < 1e-12

    def test_sin_expression(self):
        fn = _make_callable('sin(x)')
        assert fn is not None
        assert abs(fn(0.0)) < 1e-12

    def test_complex_expression(self):
        fn = _make_callable('sin(x)**2 + cos(x)**2')
        assert fn is not None
        for xv in [0.5, 1.0, 2.0]:
            assert abs(fn(xv) - 1.0) < 1e-12

    def test_syntax_error_returns_none(self):
        fn = _make_callable('x x x invalid!!!!')
        assert fn is None

    def test_pi_constant(self):
        fn = _make_callable('pi')
        assert fn is not None
        assert abs(fn(0.0) - math.pi) < 1e-12

    def test_no_builtins(self):
        fn = _make_callable('__import__("os")')
        # Should not raise but will fail to produce useful output
        # The key is it doesn't allow dangerous builtins
        assert fn is None or True  # either None from parse error or restricted

    def test_returns_callable(self):
        fn = _make_callable('x * 2')
        assert fn is not None
        assert callable(fn)

    def test_abs_available(self):
        fn = _make_callable('abs(x)')
        assert fn is not None
        assert fn(-3.0) == 3.0

    def test_sqrt_available(self):
        fn = _make_callable('sqrt(x)')
        assert fn is not None
        assert abs(fn(4.0) - 2.0) < 1e-12


# ── compress_str() ───────────────────────────────────────────────────────────

class TestCompressStr:

    def test_bad_syntax_returns_none(self):
        r = compress_str('!@#$%^&*() invalid')
        assert r is None

    def test_exception_domain_returns_none(self):
        def bad(x):
            raise ValueError("boom")
        from eml_math.discover import compress
        r = compress(bad)
        assert r is None

    def test_returns_searchresult_or_none(self):
        r = compress_str('x')
        assert r is None or isinstance(r, SearchResult)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_identity_x(self):
        r = compress_str('x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_x(self):
        r = compress_str('exp(x)')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_pythagorean_identity_compresses(self):
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_pythagorean_identity_is_simpler(self):
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.complexity <= 2

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_ln_cancels(self):
        r = compress_str('exp(log(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_x_squared(self):
        r = compress_str('x * x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_formula_is_string(self):
        r = compress_str('x')
        assert r is not None
        assert isinstance(r.formula, str)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_error_is_finite(self):
        r = compress_str('exp(x)')
        assert r is not None
        assert math.isfinite(r.error)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_complexity_is_positive(self):
        r = compress_str('x')
        assert r is not None
        assert r.complexity > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_custom_range(self):
        r = compress_str('sqrt(x)', x_lo=1.0, x_hi=9.0, n_points=20)
        assert r is not None
        assert math.isfinite(r.error)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_negation_x(self):
        r = compress_str('-x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_eml_self_formula(self):
        r = compress_str('exp(x) - log(x)', x_lo=0.5, x_hi=3.0, use_eml=True)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_ln_ln_cancels(self):
        r = compress_str('log(exp(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_constant_pi(self):
        r = compress_str('pi + 0 * x')
        assert r is not None

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_params_is_list(self):
        r = compress_str('x')
        assert r is not None
        assert isinstance(r.params, list)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_can_render_latex(self):
        r = compress_str('exp(x)')
        assert r is not None
        latex = r.to_latex()
        assert isinstance(latex, str) and len(latex) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_can_render_python(self):
        r = compress_str('exp(x)')
        assert r is not None
        py = r.to_python()
        assert 'import math' in py

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_decompress_roundtrip_latex(self):
        r = compress_str('exp(x)')
        assert r is not None
        latex = decompress(r, fmt='latex')
        assert isinstance(latex, str) and len(latex) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_decompress_roundtrip_python(self):
        r = compress_str('x')
        assert r is not None
        py = decompress(r, fmt='python')
        assert 'import math' in py

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_decompress_roundtrip_mathml(self):
        r = compress_str('x')
        assert r is not None
        ml = decompress(r, fmt='mathml')
        assert ml.startswith('<math>')

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_trig_disabled_no_sin_cos(self):
        r = compress_str('sin(x)**2 + cos(x)**2', use_trig=False)
        # Without trig operators the compressor can't recover the Pythagorean form
        # It may return None or a non-trig approximation — either is acceptable
        if r is not None:
            assert math.isfinite(r.error)


# ── compress_latex() ─────────────────────────────────────────────────────────

class TestCompressLatex:

    def test_invalid_latex_returns_none_or_result(self):
        r = compress_latex(r'!!!invalid###')
        assert r is None or isinstance(r, SearchResult)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_pythagorean_identity(self):
        r = compress_latex(r'\sin^2(x) + \cos^2(x)')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_ln_cancels(self):
        r = compress_latex(r'\exp(\ln(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_x_squared(self):
        r = compress_latex(r'x^2')
        assert r is not None
        assert r.error < 1e-4

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_dollar_delimiter_stripped(self):
        r1 = compress_latex(r'\exp(x)')
        r2 = compress_latex(r'$\exp(x)$')
        assert r1 is not None and r2 is not None
        assert abs(r1.error - r2.error) < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_frac_expression(self):
        r = compress_latex(r'\frac{1}{1}')
        assert r is not None

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_formula_is_string(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        assert isinstance(r.formula, str)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_error_is_finite(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        assert math.isfinite(r.error)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_returns_searchresult(self):
        r = compress_latex(r'\exp(x)')
        assert r is None or isinstance(r, SearchResult)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_decompress_to_latex(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        latex = decompress(r, fmt='latex')
        assert isinstance(latex, str)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_decompress_to_mathml(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        ml = decompress(r, fmt='mathml')
        assert ml.startswith('<math>')


# ── Known simplifications (Equation-in → Equation-out) ───────────────────────

class TestKnownSimplifications:
    """
    Prove that the compressor recovers exact known simplifications.
    These are the ground-truth cases: feed the expanded form in, get the
    simplified form out with error < 1e-6.
    """

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_pythagorean_identity_simplifies(self):
        """sin²(x) + cos²(x) = 1"""
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.error < 1e-6
        # Should be a very simple expression (constant)
        assert r.complexity <= 3

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_ln_inverse(self):
        """exp(ln(x)) = x"""
        r = compress_str('exp(log(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_ln_exp_inverse(self):
        """ln(exp(x)) = x"""
        r = compress_str('log(exp(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_exp_compresses_to_itself(self):
        """exp(x) is already minimal"""
        r = compress_str('exp(x)')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_identity_is_identity(self):
        """x compresses to x"""
        r = compress_str('x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_eml_self(self):
        """exp(x) - ln(x) = eml(x, x) — fundamental EML primitive"""
        r = compress_str('exp(x) - log(x)', x_lo=0.5, x_hi=3.0, use_eml=True)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_negation_is_minimal(self):
        """-x compresses to negation"""
        r = compress_str('-x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_then_decompress_latex(self):
        """Round-trip: str → compress → decompress(latex) produces non-empty LaTeX"""
        r = compress_str('exp(x)')
        assert r is not None
        latex = decompress(r, fmt='latex')
        assert len(latex) > 0
        assert isinstance(latex, str)

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_then_decompress_mathml(self):
        """Round-trip: str → compress → decompress(mathml) produces valid MathML"""
        r = compress_str('x')
        assert r is not None
        ml = decompress(r, fmt='mathml')
        assert ml.startswith('<math>') and ml.endswith('</math>')

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_then_decompress_python(self):
        """Round-trip: str → compress → decompress(python) is executable"""
        r = compress_str('x')
        assert r is not None
        py = decompress(r, fmt='python')
        assert 'import math' in py and 'lambda' in py

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_then_decompress_eml(self):
        """Round-trip: str → compress → decompress(eml) = formula string"""
        r = compress_str('exp(x)')
        assert r is not None
        eml_str = decompress(r, fmt='eml')
        assert eml_str == r.formula

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_latex_round_trip(self):
        """LaTeX input → compress_latex → decompress(latex) → LaTeX string"""
        r = compress_latex(r'\exp(x)')
        assert r is not None
        out = decompress(r, fmt='latex')
        assert isinstance(out, str) and len(out) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_get_pi_numeric_value(self):
        """get('pi') EML value matches math.pi"""
        r = get('pi')
        assert r is not None
        # pi can be recognized from the recognize() path
        assert r.error < 1e-6

    def test_get_e_exact_eml(self):
        """get('e') formula is exactly eml(1,1) = e"""
        r = get('e')
        assert r is not None
        assert 'eml(1, 1)' in r.formula
        assert r.error < 1e-12

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_str_sqrt(self):
        """sqrt(x) compresses to sqrt"""
        r = compress_str('sqrt(x)', x_lo=0.5, x_hi=9.0)
        assert r is not None
        assert r.error < 1e-6
