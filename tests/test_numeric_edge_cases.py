# Numeric edge-case regressions for the EML core.
#
# Covers:
#   * eml_scalar / tension round-trip across the full double range
#   * ln() correctness for subnormal inputs (Slipping-Wheel corruption)
#   * Rust/Python agreement between the leaf and nested evaluation paths
import math

import pytest

from eml_math import operators as ops
from eml_math.constants import OVERFLOW_THRESHOLD
from eml_math.point import EMLPoint, _LitNode

# ── round-trip property: eml_scalar(v).tension() == v ────────────────────────

ROUNDTRIP_VALUES = [
    0.0,
    -0.0,
    1.0,
    -1.0,
    2.0,
    0.5,
    -3.5,
    math.pi,
    math.e,
    1e-10,
    -1e-10,
    1e100,
    -1e100,
    1e300,
    -1e300,
    1.7976931348623157e308,      # DBL_MAX
    2.2250738585072014e-308,     # smallest normal
    1e-320,                      # subnormal
    5e-324,                      # smallest subnormal
    -5e-324,
]


@pytest.mark.parametrize("v", ROUNDTRIP_VALUES)
def test_eml_scalar_tension_roundtrip(v):
    """A literal must survive the wrap/unwrap round trip bit-for-bit."""
    got = ops.eml_scalar(v).tension()
    assert got == pytest.approx(v, rel=1e-15, abs=0.0), f"{v!r} -> {got!r}"


def test_eml_scalar_zero_roundtrips_exactly():
    assert ops.eml_scalar(0.0).tension() == 0.0
    assert ops.eml_scalar(1.0).tension() == 1.0


# ── regression: ln() was corrupted for subnormal / near-subnormal inputs ─────
#
# ln(x) is the depth-3 chain eml(1, eml(eml(1, x), 1)). The middle node
# evaluates exp(e - ln(x)), which exceeds DBL_MAX once x < ~1.1e-307. The
# Slipping-Wheel clamp in EMLPoint.tension then rewrote the exponent as
# ln(e - ln(x)) and the outer e - ln(...) no longer cancelled.
#
# Before the fix: ln(1e-320) returned -3.8878 instead of -736.8272.

@pytest.mark.parametrize(
    "x",
    [1e-307, 1e-308, 1e-309, 1e-310, 1e-315, 1e-320, 1e-323, 5e-324],
)
def test_ln_correct_for_subnormal_inputs(x):
    got = ops.ln(x).tension()
    expected = math.log(x)
    assert got == pytest.approx(expected, rel=1e-12), (
        f"ln({x!r}) = {got!r}, expected {expected!r}"
    )


def test_ln_subnormal_is_not_the_slipping_wheel_artifact():
    """The old bug produced a small negative number near -3.9, not ~-736."""
    got = ops.ln(1e-320).tension()
    assert got < -700.0, f"ln(1e-320) = {got!r} — Slipping-Wheel corruption is back"


def test_ln_normal_range_unchanged():
    for x in [0.5, 1.0, 2.0, math.e, 1e-100, 1e-300, 1e100, 1e300]:
        assert ops.ln(x).tension() == pytest.approx(math.log(x), rel=1e-12)


def test_ln_downstream_ops_on_subnormals():
    """mul/div route through ln(), so they inherited the same corruption."""
    # 1e-320 is subnormal (~4 significant digits), hence the loose rtol.
    assert ops.mul(1e-320, 1e10).tension() == pytest.approx(1e-310, rel=1e-4)
    assert ops.div(1e-320, 1e-10).tension() == pytest.approx(1e-310, rel=1e-4)
    assert ops.sqrt(1e-320).tension() == pytest.approx(math.sqrt(1e-320), rel=1e-4)


# ── regression: leaf (Rust) and nested (Python) paths must agree ─────────────
#
# EMLPoint(x, y) with plain floats is a leaf and dispatches to the Rust
# extension; EMLPoint(_LitNode(x), _LitNode(y)) is the same node mathematically
# but evaluates in Python. The two guards had drifted apart:
#   * Rust's OVERFLOW_THRESHOLD was the rounded literal 709.78 rather than
#     f64::MAX.ln() == 709.782712893384.
#   * Rust's frame-shift guard used abs(y).max(1e-300), which crushed every
#     negative subnormal up to 1e-300 instead of flooring only exact zero.

LEAF_NESTED_CASES = [
    (0.0, -1e-320),
    (1.0, -1e-310),
    (0.0, -5e-324),
    (2.0, -1e-308),
    (709.781, 1.0),      # inside the old threshold gap
    (709.7827, 1.0),
    (0.0, 0.0),          # zero still floors to 1e-300 on both paths
    (0.0, -0.0),
    (1.0, -3.0),
    (800.0, 2.0),        # genuinely over threshold on both paths
]


@pytest.mark.parametrize("x,y", LEAF_NESTED_CASES)
def test_leaf_and_nested_paths_agree(x, y):
    leaf = EMLPoint(x, y).tension()
    nested = EMLPoint(_LitNode(x), _LitNode(y)).tension()
    assert leaf == pytest.approx(nested, rel=1e-12), (
        f"EMLPoint({x!r}, {y!r}): leaf(Rust)={leaf!r} nested(Python)={nested!r}"
    )


def test_overflow_threshold_gap_is_closed():
    """x just above the old rounded 709.78 literal must not be clamped."""
    x = 709.781
    assert x < OVERFLOW_THRESHOLD
    assert EMLPoint(x, 1.0).tension() == pytest.approx(math.exp(x), rel=1e-12)


def test_negative_subnormal_y_not_crushed_to_1e300():
    """abs(y) for a negative subnormal must be kept, not floored to 1e-300."""
    got = EMLPoint(0.0, -1e-320).tension()
    assert got == pytest.approx(1.0 - math.log(1e-320), rel=1e-12)


# ── regression: _fmt_num crashed on non-finite literals ─────────────────────
#
# `v == int(v)` was evaluated before the `abs(v) < 1e6` guard, so a literal
# that Python parses to inf (1e400) raised OverflowError out of
# parse_eml_tree, which only catches SyntaxError.

def test_fmt_num_handles_non_finite():
    from eml_math.tree import _fmt_num
    assert _fmt_num(float("inf")) == "inf"
    assert _fmt_num(float("-inf")) == "-inf"
    assert _fmt_num(float("nan")) == "nan"


def test_parse_eml_tree_survives_overflowing_literal():
    from eml_math.tree import parse_eml_tree
    tree = parse_eml_tree("EML: eml_scalar(1e400)")
    assert tree is not None


# ── regression: cosh_1 was labelled "sinh(1)" ───────────────────────────────

def test_cosh_1_formula_label():
    from eml_math.discover.compress import get
    r = get("cosh_1")
    assert r.formula == "cosh(1)", f"cosh_1 mislabelled as {r.formula!r}"
    assert r.params[0] == pytest.approx(math.cosh(1.0))
    assert get("sinh_1").formula == "sinh(1)"


# ── regression: the sign-aware shim skipped inv() ───────────────────────────

def test_signed_ops_inv_keeps_sign():
    from eml_math.evaluator import eml_eval
    ctx = {"a": -2.0}
    assert eml_eval("EML: ops.inv(eml_vec('a'))", ctx) == pytest.approx(-0.5)
    # must agree with div(1, a), which was already sign-correct
    assert eml_eval("EML: ops.inv(eml_vec('a'))", ctx) == pytest.approx(
        eml_eval("EML: ops.div(eml_scalar(1.0), eml_vec('a'))", ctx)
    )
    assert eml_eval("EML: ops.mul(ops.inv(eml_vec('a')), eml_scalar(3.0))", ctx) == pytest.approx(-1.5)
    # positive path unchanged
    assert eml_eval("EML: ops.inv(eml_vec('a'))", {"a": 4.0}) == pytest.approx(0.25)


# ── regression: unbounded trial division in is_prime_tension ────────────────

def test_is_prime_tension_trial_division_is_bounded():
    import eml_math.extensions.primes as primes

    class _Pt:
        D = 6.187e34

    class _Knot:
        rho = 1.0
        point = _Pt()

    # Force the sympy-free fallback.
    real_import = __builtins__["__import__"] if isinstance(__builtins__, dict) else __import__

    def no_sympy(name, *a, **kw):
        if name == "sympy":
            raise ImportError("forced")
        return real_import(name, *a, **kw)

    import builtins
    builtins.__import__ = no_sympy
    try:
        with pytest.raises(RuntimeError, match="trial-division limit"):
            primes.is_prime_tension(_Knot())
    finally:
        builtins.__import__ = real_import


def test_is_prime_tension_non_finite_returns_false():
    import eml_math.extensions.primes as primes

    class _Pt:
        D = float("inf")

    class _Knot:
        rho = 1.0
        point = _Pt()

    assert primes.is_prime_tension(_Knot()) is False

    class _PtNan:
        D = float("nan")

    class _KnotNan:
        rho = 1.0
        point = _PtNan()

    assert primes.is_prime_tension(_KnotNan()) is False
