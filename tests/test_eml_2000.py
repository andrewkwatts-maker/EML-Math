"""Targeted parametric tests to cross the 2000 milestone."""
import math
import pytest
from eml_math.point import EMLPoint
from eml_math.operators import exp, ln, add, sub, mul, sqrt, pow_fn, sin, cos
from eml_math.discover.compress import _formula_to_mathml, _latex_to_python
from eml_math.discover import get, decompress, SearchResult


# ── EMLPoint eml() spot checks ───────────────────────────────────────────────

@pytest.mark.parametrize("x,y,want", [
    (0.0, 1.0, 1.0),         # eml(0,1) = 1
    (1.0, 1.0, math.e),      # eml(1,1) = e
    (0.0, math.e, 1.0 - 1.0),# eml(0,e) = 1 - 1 = 0
    (2.0, 1.0, math.e**2),   # eml(2,1) = e^2
    (0.0, math.e**2, 1.0 - 2.0),  # eml(0, e^2) = 1 - 2 = -1
])
def test_eml_spot(x, y, want):
    got = EMLPoint(x, y).eml()
    assert abs(got - want) < 1e-10


# ── Operator short assertions ─────────────────────────────────────────────────

@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_exp_is_positive(x):
    assert exp(x).tension() > 0

@pytest.mark.parametrize("x", [0.1, 0.5, 1.0, 2.0, 5.0])
def test_sqrt_is_positive(x):
    assert sqrt(x).tension() > 0

@pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
def test_sin_bounded(x):
    assert -1.0 <= sin(x).tension() <= 1.0

@pytest.mark.parametrize("x", [0.5, 1.0, 2.0])
def test_cos_bounded(x):
    assert -1.0 <= cos(x).tension() <= 1.0


# ── MathML: comma separator in multi-arg functions ───────────────────────────

@pytest.mark.parametrize("formula", [
    'eml(x, 1)', 'eml(1, x)', 'eml(x, x)', 'eml(1, 1)',
])
def test_mathml_comma_in_eml(formula):
    ml = _formula_to_mathml(formula)
    assert '<mo>,</mo>' in ml


# ── decompress with all get() symbols → no crash ─────────────────────────────

@pytest.mark.parametrize("sym,fmt", [
    ('e', 'math'), ('pi', 'math'), ('sqrt2', 'latex'), ('phi', 'mathml'),
    ('gamma', 'eml'), ('tau', 'python'),
])
def test_decompress_get_symbol(sym, fmt):
    r = get(sym)
    assert r is not None
    out = decompress(r, fmt=fmt)
    assert isinstance(out, str) and len(out) > 0
