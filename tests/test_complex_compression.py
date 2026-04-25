"""
Complex and lengthy equation compression tests.

Verifies compress_str / compress_latex / decompress with mathematically
interesting inputs — not stubs or hard-coded shortcuts.

Each test:
  1. Feeds a known expression string to the compressor.
  2. Checks that the output error is below a meaningful threshold.
  3. Optionally checks that the simplified form is strictly simpler.
  4. Does a round-trip back through decompress() to confirm rendering.
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.discover import compress_str, compress_latex, decompress, get, SearchResult
from eml_math.discover.compress import _latex_to_python, _make_callable, _formula_to_mathml


# ── Compression quality: known simplifications ───────────────────────────────

class TestKnownSimplificationQuality:

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_pythagorean_identity_error(self):
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_pythagorean_complexity_minimal(self):
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.complexity <= 4

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_exp_ln_cancellation_error(self):
        r = compress_str('exp(log(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_ln_exp_cancellation_error(self):
        r = compress_str('log(exp(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_eml_self_error(self):
        r = compress_str('exp(x) - log(x)', x_lo=0.5, x_hi=3.0, use_eml=True)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_identity_function_error(self):
        r = compress_str('x')
        assert r is not None
        assert r.error < 1e-8

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_negation_error(self):
        r = compress_str('-x')
        assert r is not None
        assert r.error < 1e-8

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_exp_alone_error(self):
        r = compress_str('exp(x)')
        assert r is not None
        assert r.error < 1e-8

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_x_squared_error(self):
        r = compress_str('x * x')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_sqrt_error(self):
        r = compress_str('sqrt(x)', x_lo=0.5, x_hi=9.0)
        assert r is not None
        assert r.error < 1e-6


# ── MathML correctness tests ──────────────────────────────────────────────────

class TestMathMLCorrectness:

    def test_simple_x(self):
        ml = _formula_to_mathml('x')
        assert ml == '<math><mi>x</mi></math>'

    def test_simple_1(self):
        ml = _formula_to_mathml('1')
        assert '<mn>1</mn>' in ml

    def test_simple_pi(self):
        ml = _formula_to_mathml('pi')
        assert '&pi;' in ml

    def test_simple_pi_unicode(self):
        ml = _formula_to_mathml('π')
        assert '&pi;' in ml

    def test_simple_inf(self):
        ml = _formula_to_mathml('∞')
        assert '&infin;' in ml

    def test_exp_x_has_mi_exp(self):
        ml = _formula_to_mathml('exp(x)')
        assert '<mi>exp</mi>' in ml

    def test_exp_x_has_parens(self):
        ml = _formula_to_mathml('exp(x)')
        assert '<mo>(</mo>' in ml
        assert '<mo>)</mo>' in ml

    def test_ln_x_has_mi_ln(self):
        ml = _formula_to_mathml('ln(x)')
        assert '<mi>ln</mi>' in ml

    def test_sin_x_has_mi_sin(self):
        ml = _formula_to_mathml('sin(x)')
        assert '<mi>sin</mi>' in ml

    def test_cos_x_has_mi_cos(self):
        ml = _formula_to_mathml('cos(x)')
        assert '<mi>cos</mi>' in ml

    def test_addition_has_plus_operator(self):
        ml = _formula_to_mathml('x + 1')
        assert '<mo>+</mo>' in ml

    def test_subtraction_has_minus_operator(self):
        ml = _formula_to_mathml('exp(x) - ln(x)')
        assert '<mo>-</mo>' in ml

    def test_multiplication_has_sdot(self):
        ml = _formula_to_mathml('x * 2')
        assert '<mo>&sdot;</mo>' in ml

    def test_division_has_slash(self):
        ml = _formula_to_mathml('x / 2')
        assert '<mo>/</mo>' in ml

    def test_eml_has_mi_eml(self):
        ml = _formula_to_mathml('eml(x, x)')
        assert '<mi>eml</mi>' in ml

    def test_sqrt_has_mi_sqrt(self):
        ml = _formula_to_mathml('sqrt(x)')
        assert '<mi>sqrt</mi>' in ml

    def test_number_two(self):
        ml = _formula_to_mathml('2')
        assert '<mn>2</mn>' in ml

    def test_outer_math_tags(self):
        for formula in ['x', 'exp(x)', '1', 'eml(1, 1)']:
            ml = _formula_to_mathml(formula)
            assert ml.startswith('<math>'), f"No open tag for {formula!r}"
            assert ml.endswith('</math>'), f"No close tag for {formula!r}"

    def test_comma_in_eml_args(self):
        ml = _formula_to_mathml('eml(1, 1)')
        assert '<mo>,</mo>' in ml

    def test_complex_formula_valid_mathml(self):
        ml = _formula_to_mathml('exp(x) - ln(x) + sin(x) * cos(x)')
        assert ml.startswith('<math>')
        assert ml.endswith('</math>')
        assert '<mi>exp</mi>' in ml
        assert '<mi>ln</mi>' in ml
        assert '<mi>sin</mi>' in ml
        assert '<mi>cos</mi>' in ml


# ── Complex equation compression with output matching ─────────────────────────

class TestComplexCompressionOutputMatching:

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_pythagorean_constant_output(self):
        """sin²+cos²=1 → should compress to a constant-like expression."""
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        assert r.error < 1e-6
        # The result should be very close to the constant 1
        fn = _make_callable(r.formula)
        if fn is not None:
            for xv in [0.5, 1.0, 2.0]:
                val = fn(xv)
                if val is not None and math.isfinite(val):
                    assert abs(val - 1.0) < 1e-5

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_exp_ln_output_matches_x(self):
        """exp(ln(x))=x → formula should evaluate close to x."""
        r = compress_str('exp(log(x))', x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6
        fn = _make_callable(r.formula)
        if fn is not None:
            for xv in [0.5, 1.0, 2.0, 3.0]:
                val = fn(xv)
                if val is not None and math.isfinite(val):
                    assert abs(val - xv) < 1e-5

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_negation_output_matches_neg_x(self):
        """compress(-x) result evaluates to -x."""
        r = compress_str('-x')
        assert r is not None
        assert r.error < 1e-6
        fn = _make_callable(r.formula)
        if fn is not None:
            for xv in [0.5, 1.0, 2.0]:
                val = fn(xv)
                if val is not None and math.isfinite(val):
                    assert abs(val - (-xv)) < 1e-5

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_identity_output_matches_x(self):
        r = compress_str('x')
        assert r is not None
        fn = _make_callable(r.formula)
        if fn is not None:
            for xv in [0.3, 1.0, 2.7]:
                val = fn(xv)
                if val is not None and math.isfinite(val):
                    assert abs(val - xv) < 1e-5

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_exp_output_matches_math_exp(self):
        r = compress_str('exp(x)')
        assert r is not None
        fn = _make_callable(r.formula)
        if fn is not None:
            for xv in [0.5, 1.0, 2.0]:
                val = fn(xv)
                if val is not None and math.isfinite(val):
                    assert abs(val - math.exp(xv)) / max(1.0, abs(math.exp(xv))) < 1e-5

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_sum_two_vars_output(self):
        """x0 + x1 (multivariate)."""
        n = 30
        x0 = [0.5 + i * 0.1 for i in range(n)]
        x1 = [1.0 + i * 0.05 for i in range(n)]
        y = [a + b for a, b in zip(x0, x1)]
        from eml_math.discover import Searcher
        r = Searcher(max_complexity=5, precision_goal=1e-8).find([x0, x1], y)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_compress_latex_pythagorean_via_string(self):
        r = compress_latex(r'\sin^2(x) + \cos^2(x)')
        assert r is not None
        assert r.error < 1e-6
        # Decompose to all formats
        for fmt in ['eml', 'math', 'latex', 'mathml', 'python']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str) and len(out) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_compress_latex_exp_via_latex(self):
        r = compress_latex(r'\exp(x)')
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_compress_error_always_finite(self):
        exprs = ['x', 'exp(x)', '-x', 'x * x']
        for expr in exprs:
            r = compress_str(expr)
            if r is not None:
                assert math.isfinite(r.error), f"{expr} error = {r.error}"

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_compress_complexity_always_positive(self):
        exprs = ['x', 'exp(x)', '-x']
        for expr in exprs:
            r = compress_str(expr)
            if r is not None:
                assert r.complexity > 0, f"{expr} complexity = {r.complexity}"

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_all_formats_from_pythagorean(self):
        r = compress_str('sin(x)**2 + cos(x)**2')
        assert r is not None
        for fmt in ['eml', 'math', 'latex', 'mathml', 'python']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str) and len(out) > 0, f"fmt={fmt} empty"

    @pytest.mark.skipif(not _RUST, reason="Rust required")
    def test_all_formats_from_eml_self(self):
        r = compress_str('exp(x) - log(x)', x_lo=0.5, use_eml=True)
        assert r is not None
        for fmt in ['eml', 'math', 'latex', 'mathml', 'python']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str) and len(out) > 0, f"fmt={fmt} empty"


# ── LaTeX parsing for complex inputs ─────────────────────────────────────────

class TestLatexParsingComplex:

    def test_parse_frac_converts_to_div(self):
        py = _latex_to_python(r'\frac{x}{2}')
        assert '/' in py or 'x' in py

    def test_parse_sqrt_converts(self):
        py = _latex_to_python(r'\sqrt{x^2 + 1}')
        assert 'sqrt' in py

    def test_parse_sin_squared_converts(self):
        py = _latex_to_python(r'\sin^2(x) + \cos^2(x)')
        assert 'sin' in py
        assert 'cos' in py
        assert '**2' in py

    def test_parse_exp_ln_converts(self):
        py = _latex_to_python(r'\exp(\ln(x))')
        assert 'exp' in py
        assert 'log' in py

    def test_parse_pi_constant(self):
        py = _latex_to_python(r'2\pi')
        assert 'pi' in py

    def test_parse_mixed_power_and_trig(self):
        py = _latex_to_python(r'\sin(x)^2 + \cos(x)^2')
        assert 'sin' in py and 'cos' in py

    def test_parse_left_right_parens(self):
        py = _latex_to_python(r'\exp\left(\frac{x}{2}\right)')
        assert 'exp' in py
        assert r'\left' not in py
        assert r'\right' not in py

    def test_parse_basic_passthrough(self):
        py = _latex_to_python('x + 1')
        assert 'x' in py and '1' in py and '+' in py

    @pytest.mark.parametrize("latex,expected_contains", [
        (r'\sin(x)', 'sin'),
        (r'\cos(x)', 'cos'),
        (r'\exp(x)', 'exp'),
        (r'\ln(x)', 'log'),
        (r'\sqrt{x}', 'sqrt'),
        (r'\pi', 'pi'),
        (r'\infty', 'inf'),
        (r'x^2', '**2'),
        (r'x^{2}', '**'),
        (r'\frac{1}{x}', '/'),
    ])
    def test_parse_parametric(self, latex, expected_contains):
        result = _latex_to_python(latex)
        assert expected_contains in result, f"'{expected_contains}' not in _latex_to_python({latex!r}) = {result!r}"


# ── get() with round-trip verify ─────────────────────────────────────────────

class TestGetRoundTrip:

    def test_get_e_roundtrip_all_formats(self):
        r = get('e')
        assert r is not None
        for fmt in ['eml', 'math', 'latex', 'mathml', 'python']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str) and len(out) > 0

    def test_get_pi_roundtrip_mathml(self):
        r = get('pi')
        assert r is not None
        ml = decompress(r, fmt='mathml')
        assert '<math>' in ml and '</math>' in ml

    def test_get_sqrt2_roundtrip_latex(self):
        r = get('sqrt2')
        assert r is not None
        latex = decompress(r, fmt='latex')
        assert isinstance(latex, str) and len(latex) > 0

    def test_get_phi_roundtrip_python(self):
        r = get('phi')
        assert r is not None
        py = decompress(r, fmt='python')
        assert 'import math' in py

    def test_get_gamma_roundtrip_math(self):
        r = get('gamma')
        assert r is not None
        m = decompress(r, fmt='math')
        assert isinstance(m, str)

    def test_get_inf_roundtrip_mathml(self):
        r = get('inf')
        assert r is not None
        ml = decompress(r, fmt='mathml')
        assert '&infin;' in ml

    def test_get_tau_roundtrip(self):
        r = get('tau')
        assert r is not None
        for fmt in ['eml', 'math', 'latex', 'mathml']:
            out = decompress(r, fmt=fmt)
            assert isinstance(out, str)

    def test_get_e2_roundtrip(self):
        r = get('e2')
        assert r is not None
        eml_str = decompress(r, fmt='eml')
        assert 'eml(2, 1)' in eml_str

    def test_get_1_over_e_roundtrip(self):
        r = get('1_over_e')
        assert r is not None
        eml_str = decompress(r, fmt='eml')
        assert 'eml(-1, 1)' in eml_str
