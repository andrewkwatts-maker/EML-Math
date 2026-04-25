"""
Tests for:
  - Rust batch arithmetic operators (exp_n, ln_n, add_n, mul_n, etc.)
  - SearchResult.to_mathjax() / to_mathml()
  - decompress() with fmt='mathjax' and fmt='mathml'
  - C API arithmetic function coverage (via Python validation of same math)
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.discover import get, decompress
from eml_math.discover.result import SearchResult


# ── Rust batch arithmetic ─────────────────────────────────────────────────────

@pytest.mark.skipif(not _RUST, reason="Rust required")
class TestRustBatchArithmetic:

    def test_exp_n_identity(self):
        xs = [0.0, 0.5, 1.0, 2.0, 5.0]
        out = _core.exp_n(xs)
        for x, v in zip(xs, out):
            assert abs(v - math.exp(x)) < 1e-10

    def test_exp_n_length(self):
        assert len(_core.exp_n([1.0, 2.0, 3.0])) == 3

    def test_ln_n_identity(self):
        ys = [0.5, 1.0, math.e, 4.0]
        out = _core.ln_n(ys)
        for y, v in zip(ys, out):
            assert abs(v - math.log(y)) < 1e-10

    def test_ln_n_frame_shift_guard(self):
        # y <= 0 should not raise, returns ln(|y|)
        out = _core.ln_n([-1.0])
        assert math.isfinite(out[0])

    @pytest.mark.parametrize("a,b", [(1.0, 2.0), (3.5, -1.5), (0.0, 5.0)])
    def test_add_n(self, a, b):
        assert abs(_core.add_n([a], [b])[0] - (a + b)) < 1e-12

    @pytest.mark.parametrize("a,b", [(5.0, 2.0), (3.5, 1.0)])
    def test_sub_n(self, a, b):
        assert abs(_core.sub_n([a], [b])[0] - (a - b)) < 1e-12

    @pytest.mark.parametrize("a,b", [(2.0, 3.0), (-2.0, 5.0), (0.5, 4.0)])
    def test_mul_n(self, a, b):
        assert abs(_core.mul_n([a], [b])[0] - a * b) < 1e-10

    @pytest.mark.parametrize("a,b", [(6.0, 2.0), (1.0, 4.0), (-9.0, 3.0)])
    def test_div_n(self, a, b):
        assert abs(_core.div_n([a], [b])[0] - a / b) < 1e-10

    def test_div_n_zero_denominator_returns_nan(self):
        result = _core.div_n([1.0], [0.0])
        assert math.isnan(result[0])

    @pytest.mark.parametrize("x", [0.25, 1.0, 4.0, 9.0])
    def test_sqrt_n(self, x):
        assert abs(_core.sqrt_n([x])[0] - math.sqrt(x)) < 1e-10

    def test_sqrt_n_negative_input(self):
        result = _core.sqrt_n([-4.0])
        assert abs(result[0] - 2.0) < 1e-10  # sqrt(|-4|) = 2

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, math.pi / 2])
    def test_sin_n(self, x):
        assert abs(_core.sin_n([x])[0] - math.sin(x)) < 1e-10

    @pytest.mark.parametrize("x", [0.0, 0.5, 1.0, math.pi / 3])
    def test_cos_n(self, x):
        assert abs(_core.cos_n([x])[0] - math.cos(x)) < 1e-10

    def test_tension_n_basic(self):
        xs = [0.0, 1.0, 2.0]
        ys = [1.0, 1.0, 1.0]
        out = _core.tension_n(xs, ys)
        for x, v in zip(xs, out):
            assert abs(v - math.exp(x)) < 1e-10  # ln(1) = 0

    @pytest.mark.parametrize("base,exp_", [(2.0, 3.0), (3.0, 2.0), (5.0, 0.5)])
    def test_pow_n(self, base, exp_):
        result = _core.pow_n([base], [exp_])[0]
        assert abs(result - base ** exp_) < 1e-8

    def test_batch_parallel_large(self):
        n = 1000
        xs = [i * 0.001 for i in range(n)]
        out = _core.exp_n(xs)
        assert len(out) == n
        assert all(math.isfinite(v) for v in out)

    def test_exp_ln_roundtrip_batch(self):
        xs = [0.1, 0.5, 1.0, 2.0]
        lns = _core.ln_n(_core.exp_n(xs))
        for x, v in zip(xs, lns):
            assert abs(v - x) < 1e-10

    def test_sin_cos_pythagorean_batch(self):
        xs = [0.1, 0.5, 1.0, 1.5, 2.0]
        ss = _core.sin_n(xs)
        cs = _core.cos_n(xs)
        for s, c in zip(ss, cs):
            assert abs(s * s + c * c - 1.0) < 1e-10


# ── SearchResult.to_mathjax / to_mathml ───────────────────────────────────────

class TestMathJaxOutput:

    def _sr(self, formula):
        return SearchResult(formula, 0.0, 1, [])

    def test_to_mathjax_wraps_in_delimiters(self):
        r = self._sr('exp(x)')
        mj = r.to_mathjax()
        assert mj.startswith(r'\(')
        assert mj.endswith(r'\)')

    def test_to_mathjax_contains_latex(self):
        r = self._sr('exp(x)')
        assert r'\exp' in r.to_mathjax()

    def test_to_mathjax_pi(self):
        r = self._sr('pi')
        assert r'\pi' in r.to_mathjax()

    def test_to_mathjax_ln(self):
        r = self._sr('ln(x)')
        assert r'\ln' in r.to_mathjax()

    def test_to_mathjax_sqrt(self):
        r = self._sr('sqrt(x)')
        assert r'\sqrt' in r.to_mathjax()

    def test_to_mathjax_sin(self):
        r = self._sr('sin(x)')
        assert r'\sin' in r.to_mathjax()

    def test_to_mathjax_cos(self):
        r = self._sr('cos(x)')
        assert r'\cos' in r.to_mathjax()

    def test_to_mathjax_eml(self):
        r = self._sr('eml(1, 1)')
        assert r'\mathrm{eml}' in r.to_mathjax()

    def test_to_mathml_starts_with_math_tag(self):
        r = self._sr('exp(x)')
        ml = r.to_mathml()
        assert ml.startswith('<math>')
        assert ml.endswith('</math>')

    def test_to_mathml_pi(self):
        r = self._sr('pi')
        ml = r.to_mathml()
        assert 'pi' in ml or '&pi;' in ml

    @pytest.mark.parametrize("sym", ['e', 'pi', 'sqrt2', 'phi', 'gamma'])
    def test_get_to_mathjax(self, sym):
        r = get(sym)
        mj = r.to_mathjax()
        assert mj.startswith(r'\(') and mj.endswith(r'\)')

    @pytest.mark.parametrize("sym", ['e', 'pi', 'sqrt2'])
    def test_get_to_mathml(self, sym):
        r = get(sym)
        ml = r.to_mathml()
        assert '<math>' in ml


# ── decompress with mathjax/mathml formats ────────────────────────────────────

class TestDecompressFormats:

    @pytest.mark.parametrize("sym,fmt", [
        ('e', 'mathjax'), ('pi', 'mathjax'), ('sqrt2', 'mathjax'),
        ('e', 'mathml'), ('pi', 'mathml'), ('sqrt2', 'mathml'),
        ('e', 'latex'), ('pi', 'latex'), ('sqrt2', 'python'),
    ])
    def test_decompress_sym_fmt(self, sym, fmt):
        r = get(sym)
        out = decompress(r, fmt=fmt)
        assert isinstance(out, str) and len(out) > 0

    def test_decompress_mathjax_wrapped(self):
        r = get('pi')
        out = decompress(r, fmt='mathjax')
        assert out.startswith(r'\(') and out.endswith(r'\)')

    def test_decompress_mathml_has_tag(self):
        r = get('pi')
        out = decompress(r, fmt='mathml')
        assert '<math>' in out

    def test_decompress_eml_format(self):
        r = get('e')
        out = decompress(r, fmt='eml')
        assert out == r.formula

    def test_decompress_math_format(self):
        r = get('sqrt2')
        out = decompress(r, fmt='math')
        assert isinstance(out, str) and len(out) > 0

    def test_decompress_python_has_import(self):
        r = get('e')
        out = decompress(r, fmt='python')
        assert 'import math' in out
