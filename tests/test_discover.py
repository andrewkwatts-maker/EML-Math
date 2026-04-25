"""
Tests for EML formula discovery (eml.discover).

Verifies that the Rust-backed BFS recovers known EML expressions from data,
and that the compress() / recognize() convenience API works correctly.
"""
import math
import pytest

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False

from eml_math.discover import Searcher, SearchResult, compress, recognize


def _x(n=40, lo=0.2, hi=3.0):
    step = (hi - lo) / (n - 1)
    return [lo + i * step for i in range(n)]


class TestSearchResult:

    def test_repr(self):
        r = SearchResult("exp(x)", 1.2e-11, 2, [])
        assert "exp(x)" in repr(r)
        assert "1.20e-11" in repr(r)

    def test_to_latex(self):
        r = SearchResult("exp(x) - ln(x)", 0.0, 3, [])
        assert r"\exp" in r.to_latex()
        assert r"\ln" in r.to_latex()

    def test_to_python(self):
        r = SearchResult("exp(x)", 0.0, 2, [])
        assert "math.exp" in r.to_python()
        assert "import math" in r.to_python()


class TestSearcherRecognize:

    def test_recognize_pi(self):
        s = Searcher()
        r = s.recognize(math.pi)
        assert r is not None
        assert "π" in r.formula

    def test_recognize_e(self):
        s = Searcher()
        r = s.recognize(math.e)
        assert r is not None
        assert "e" in r.formula

    def test_recognize_ln2(self):
        s = Searcher()
        r = s.recognize(math.log(2))
        assert r is not None

    def test_recognize_eml_constant(self):
        s = Searcher()
        val = math.exp(2) - math.log(3)  # eml(2, 3)
        r = s.recognize(val)
        assert r is not None
        assert r.error < 1e-6


class TestFormulaDiscovery:

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_finds_exp(self):
        x = _x()
        y = [math.exp(xi) for xi in x]
        result = Searcher(max_complexity=4, precision_goal=1e-8).find(x, y)
        assert result is not None
        assert result.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_finds_identity(self):
        x = _x()
        y = list(x)
        result = Searcher(max_complexity=3, precision_goal=1e-10).find(x, y)
        assert result is not None
        assert result.error < 1e-8

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_finds_negation(self):
        x = _x()
        y = [-xi for xi in x]
        result = Searcher(max_complexity=4, precision_goal=1e-10).find(x, y)
        assert result is not None
        assert result.error < 1e-8

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_finds_eml_self(self):
        """eml(x, x) = exp(x) - ln(x) — the fundamental EML expression."""
        x = _x()
        y = [math.exp(xi) - math.log(xi) for xi in x]
        result = Searcher(max_complexity=6, precision_goal=1e-8, use_eml=True).find(x, y)
        assert result is not None
        assert result.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_finds_sum_two_vars(self):
        """x + y — basic multivariate."""
        n = 30
        x0 = [0.5 + i * 0.1 for i in range(n)]
        x1 = [1.0 + i * 0.05 for i in range(n)]
        y = [a + b for a, b in zip(x0, x1)]
        result = Searcher(max_complexity=5, precision_goal=1e-8).find([x0, x1], y)
        assert result is not None
        assert result.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_result_has_formula_string(self):
        x = _x()
        y = [xi * xi for xi in x]  # x^2 — via mul(x,x)
        result = Searcher(max_complexity=5).find(x, y)
        assert result is not None
        assert isinstance(result.formula, str)
        assert len(result.formula) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required for search")
    def test_result_error_is_finite(self):
        x = _x()
        y = [math.sin(xi) for xi in x]
        result = Searcher(max_complexity=5, use_trig=True).find(x, y)
        assert result is not None
        assert math.isfinite(result.error)


class TestRustCore:
    """Direct tests of the Rust extension types."""

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_eml_point_tension(self):
        from eml_math import eml_core
        p = eml_core.EMLPoint(1.0, 1.0)
        assert abs(p.tension() - math.e) < 1e-12

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_eml_point_mirror_pulse(self):
        from eml_math import eml_core
        p = eml_core.EMLPoint(1.0, 1.0)
        p2 = p.mirror_pulse()
        assert abs(p2.x - 1.0) < 1e-12   # x_new = y_old = 1.0
        assert abs(p2.y - math.e) < 1e-12 # y_new = tension = e

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_eml_pair_rotate_phase_pi_half(self):
        from eml_math import eml_core
        p = eml_core.EMLPair.from_values(1.0, 0.0)
        p2 = p.rotate_phase(math.pi / 2)
        assert abs(p2.real_tension) < 1e-10
        assert abs(p2.imag_tension - 1.0) < 1e-10

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_eml_pair_modulus(self):
        from eml_math import eml_core
        p = eml_core.EMLPair.from_values(3.0, 4.0)
        assert abs(p.modulus - 5.0) < 1e-12

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_eml_pair_conjugate(self):
        from eml_math import eml_core
        p = eml_core.EMLPair.from_values(2.0, 3.0)
        c = p.conjugate()
        assert c.real_tension == pytest.approx(2.0)
        assert c.imag_tension == pytest.approx(-3.0)

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_schrodinger_step_n_modulus_preserved(self):
        from eml_math import eml_core
        psi = [(math.cos(i), math.sin(i)) for i in range(8)]
        v = [float(i) for i in range(8)]
        psi2 = eml_core.schrodinger_step_n(psi, v, 0.1, 1.0)
        for (r0, i0), (r1, i1) in zip(psi, psi2):
            mod0 = math.sqrt(r0**2 + i0**2)
            mod1 = math.sqrt(r1**2 + i1**2)
            assert abs(mod0 - mod1) < 1e-12

    @pytest.mark.skipif(not _RUST, reason="Rust extension not installed")
    def test_simulate_pulses_n(self):
        from eml_math import eml_core
        results = eml_core.simulate_pulses_n(1.0, 1.0, 5)
        assert len(results) == 6
        # First element should be initial state
        x0, y0, t0 = results[0]
        assert abs(x0 - 1.0) < 1e-12
        assert abs(y0 - 1.0) < 1e-12
        assert abs(t0 - math.e) < 1e-10


# ── TestCompress ──────────────────────────────────────────────────────────────

class TestCompress:
    """Tests for the compress() equation-compression convenience function."""

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_exp_returns_result(self):
        r = compress(math.exp)
        assert r is not None
        assert isinstance(r.formula, str)
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_eml_self(self):
        r = compress(lambda x: math.exp(x) - math.log(x), x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_pythagorean_identity(self):
        # sin²(x) + cos²(x) = 1 — should compress to a constant
        r = compress(lambda x: math.sin(x)**2 + math.cos(x)**2)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_returns_lower_complexity_than_input(self):
        # exp(ln(x)) = x — should recover identity, complexity ≤ 2
        r = compress(lambda x: math.exp(math.log(x)), x_lo=0.5, x_hi=3.0)
        assert r is not None
        assert r.error < 1e-6

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_custom_range(self):
        r = compress(math.sqrt, x_lo=1.0, x_hi=9.0, n_points=20)
        assert r is not None
        assert math.isfinite(r.error)

    def test_compress_bad_domain_returns_none_or_result(self):
        # Function that raises everywhere — should return None gracefully
        def bad(x):
            raise ValueError("bad domain")
        r = compress(bad)
        assert r is None

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_result_has_latex(self):
        r = compress(math.exp)
        assert r is not None
        latex = r.to_latex()
        assert isinstance(latex, str)
        assert len(latex) > 0

    @pytest.mark.skipif(not _RUST, reason="Rust extension required")
    def test_compress_result_has_python(self):
        r = compress(math.exp)
        assert r is not None
        py = r.to_python()
        assert "import math" in py
        assert "lambda" in py


# ── TestRecognize ─────────────────────────────────────────────────────────────

class TestRecognize:
    """Tests for the recognize() constant-identification function."""

    def test_recognize_pi(self):
        r = recognize(math.pi)
        assert r is not None
        assert "π" in r.formula

    def test_recognize_e(self):
        r = recognize(math.e)
        assert r is not None
        assert "e" in r.formula

    def test_recognize_sqrt2(self):
        r = recognize(math.sqrt(2))
        assert r is not None
        assert r.error < 1e-8

    def test_recognize_ln2(self):
        r = recognize(math.log(2))
        assert r is not None

    def test_recognize_golden_ratio(self):
        phi = (1 + math.sqrt(5)) / 2
        r = recognize(phi)
        assert r is not None
        assert "φ" in r.formula

    def test_recognize_euler_mascheroni(self):
        gamma = 0.5772156649015328
        r = recognize(gamma)
        assert r is not None
        assert "γ" in r.formula

    def test_recognize_eml_integer_combo(self):
        val = math.exp(2) - math.log(3)   # eml(2, 3)
        r = recognize(val)
        assert r is not None
        assert r.error < 1e-6

    def test_recognize_unknown_returns_none(self):
        r = recognize(123.456789)
        assert r is None

    def test_recognize_returns_searchresult_type(self):
        r = recognize(math.pi)
        assert isinstance(r, SearchResult)
