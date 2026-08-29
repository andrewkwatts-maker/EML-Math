"""Non-EML primitives: abs, sum, max, min, log10, gt, lt, and_, or_.

WHY THESE EXIST
---------------
Expression authors reach for ``ops.abs`` and friends. They were absent, so
every expression using them raised AttributeError -- 35 registry parameters
in the downstream framework were recorded as "did not evaluate" when the
expressions were fine and only the operator was missing.

THE TRAP THESE TESTS GUARD
--------------------------
Exposing them under their natural names shadows Python builtins inside the
operators module. ``abs = abs_fn`` makes abs_fn's own ``abs(...)`` call
resolve back to abs_fn -- unbounded recursion -- because module globals
shadow builtins at CALL time, not at def time. The module captures
``_builtin_abs`` etc. before the alias block to prevent that, and the
recursion tests below are what keep it prevented.
"""
from __future__ import annotations

import math

import pytest

import eml_math.operators as ops


# ── the recursion trap ───────────────────────────────────────────────────────


@pytest.mark.parametrize("fn,arg", [
    (lambda: ops.abs(-3.0), 3.0),
    (lambda: ops.sum(1.0, 2.0), 3.0),
    (lambda: ops.max(1.0, 3.0), 3.0),
    (lambda: ops.min(1.0, 3.0), 1.0),
])
def test_shadowing_aliases_do_not_recurse(fn, arg):
    """If the builtin capture is removed these raise RecursionError."""
    assert fn() == pytest.approx(arg)


def test_mirror_abs_still_uses_the_real_builtin():
    """The pre-existing primitive must survive the shadowing."""
    assert ops.mirror_abs(-5.0) == 5.0
    assert ops.mirror_abs(5.0) == 5.0


def test_existing_eml_compositions_are_unaffected():
    assert ops.mul(3.0, 4.0).tension() == pytest.approx(12.0)
    assert ops.add(3.0, 4.0).tension() == pytest.approx(7.0)
    assert ops.div(12.0, 4.0).tension() == pytest.approx(3.0)


# ── values ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("x,expected", [(-3.0, 3.0), (3.0, 3.0), (0.0, 0.0)])
def test_abs_values(x, expected):
    assert ops.abs(x) == pytest.approx(expected)


def test_abs_accepts_an_eml_point():
    """Expressions nest operators, so a primitive must take a tree node."""
    assert ops.abs(ops.eml_scalar(-4.0)) == pytest.approx(4.0)
    assert ops.abs(ops.sub(2.0, 9.0)) == pytest.approx(7.0)


def test_sum_is_variadic_and_accepts_nodes():
    assert ops.sum(1.0, 2.0, 3.0, 4.0) == pytest.approx(10.0)
    assert ops.sum(ops.eml_scalar(1.5), 2.5) == pytest.approx(4.0)


def test_max_and_min():
    assert ops.max(1.0, 7.0, 3.0) == pytest.approx(7.0)
    assert ops.min(1.0, 7.0, 3.0) == pytest.approx(1.0)
    assert ops.max(ops.eml_scalar(2.0), 5.0) == pytest.approx(5.0)


@pytest.mark.parametrize("fn", [ops.max, ops.min])
def test_selection_requires_an_argument(fn):
    with pytest.raises(ValueError):
        fn()


@pytest.mark.parametrize("x,expected", [(1000.0, 3.0), (1.0, 0.0), (0.01, -2.0)])
def test_log10_values(x, expected):
    assert ops.log10(x) == pytest.approx(expected, abs=1e-9)


def test_log10_agrees_with_the_eml_composition():
    """log10 delegates to log_fn(10, x) so the two cannot drift apart."""
    for x in (2.0, 50.0, 1234.0):
        assert ops.log10(x) == pytest.approx(
            ops.log_fn(10.0, x).tension(), abs=1e-9
        )
        assert ops.log10(x) == pytest.approx(math.log10(x), abs=1e-9)


# ── comparison and logic ─────────────────────────────────────────────────────


def test_gt_and_lt():
    assert ops.gt(3.0, 2.0) is True
    assert ops.gt(2.0, 3.0) is False
    assert ops.lt(2.0, 3.0) is True
    assert ops.gt(ops.eml_scalar(5.0), 1.0) is True


def test_and_or():
    assert ops.and_(True, True) is True
    assert ops.and_(True, False) is False
    assert ops.or_(False, True) is True
    assert ops.or_(False, False) is False


def test_and_is_named_with_a_trailing_underscore():
    """`and` is a keyword; the underscore is the convention callers use."""
    assert hasattr(ops, "and_")
    assert not hasattr(ops, "and")


# ── context exposed as bare names (evaluator) ────────────────────────────────


def test_context_values_resolve_as_bare_names():
    """Expressions are written both ways; both must resolve.

    Only eml_vec('b3') used to work, so ops.div(b3, chi_eff) raised
    NameError and was scored as a failed expression -- misreading a naming
    convention as broken physics.
    """
    from eml_math.evaluator import EMLEvaluator

    ev = EMLEvaluator({"b3": 24.0, "chi_eff": 72.0}, strict=False)
    bare = ev.eval("EML: ops.div(b3, chi_eff)")
    explicit = ev.eval("EML: ops.div(eml_vec('b3'), eml_vec('chi_eff'))")
    assert bare == pytest.approx(explicit)
    assert bare == pytest.approx(24.0 / 72.0)


def test_context_cannot_shadow_the_dsl():
    """A parameter named 'ops' or 'math' must not break every expression."""
    from eml_math.evaluator import EMLEvaluator

    ev = EMLEvaluator(
        {"ops": 1.0, "math": 2.0, "eml_scalar": 3.0, "eml_vec": 4.0,
         "eml_pi": 5.0, "b3": 24.0},
        strict=False,
    )
    assert ev.eval("EML: ops.add(eml_scalar(1.0), eml_scalar(2.0))") == pytest.approx(3.0)
    assert ev.eval("EML: ops.mul(b3, eml_scalar(2.0))") == pytest.approx(48.0)
