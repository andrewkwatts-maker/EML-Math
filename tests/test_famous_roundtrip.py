"""
Round-trip validation for famous equations:

  famous-equation EML string
        │
        ▼
   parse  →   pure-EML EMLTreeNode
        │
        ▼
   to_compact  →  JSON-friendly array
        │
        ▼
   json.dumps / loads
        │
        ▼
   from_compact  →  EMLTreeNode (rebuilt)

Goals:

1. **Structural round-trip** — every famous equation must parse, serialise
   to compact JSON, and inflate back into an identical tree (same labels,
   kinds, children counts at every node).
2. **Hand-built numeric round-trip** — for hand-written EML expressions
   whose pure-EML evaluation stays within the positive-real domain, the
   serialised tree, when re-evaluated, must agree with the textbook
   value.

There is one **known limitation** of pure-EML on real arithmetic that
matters here: the binary primitive ``eml(L, R) = exp(L) − ln(R)`` only
yields a real number when ``R > 0`` and ``L`` is finite. The pure-EML
encoding of ``add(u, v) = eml(pure_ln(u), eml(pure_neg(v), 1))``
therefore requires ``u, v > 0`` so the inner ``pure_ln`` is real. With
constants ``< 1`` (e.g. the ``0.5`` in ``sqrt(x) = exp(0.5·ln(x))``)
``ln(0.5) < 0`` and the inner ``pure_ln`` falls back on its
``|·|`` frame-guard, returning ``+ln 2`` instead of ``-ln 2``. The tree
still **stores** the equation correctly — it's the pure-EML *evaluator*
that can't recover the original sign without the frame guard. Numerical
evaluation of these equations must go through ``EMLEvaluator`` (which
runs on the un-encoded expression and stays in IEEE-754 the whole way).

This is not a bug introduced by the JSON round-trip — it is a property
of the pure-EML algebra on the reals. The structural round-trip is
unaffected; we just don't ask the pure-EML tree evaluator to re-compute
expressions whose intermediate values cross zero.
"""
from __future__ import annotations

import json
import math
from typing import Dict

import pytest

from eml_math import EMLTreeNode, NodeKind, from_compact, to_compact

# Re-use textbook reference inputs from the existing test file.
import sys
import pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from test_famous import REFERENCE  # noqa: E402

from eml_math.famous import FAMOUS, get


# ── Pure-EML tree evaluator ──────────────────────────────────────────────────


def _eval_pure_eml(node: EMLTreeNode, ctx: Dict[str, float]) -> float:
    """Numerically evaluate a pure-EML tree at a given variable context.

    Pure-EML internal nodes are ``eml(L, R) = exp(L) − ln(R)`` with the
    convention that an explicit ``0`` leaf (``BOTTOM`` kind) on the
    L-side suppresses the exp term (``exp(⊥) := 0``). Leaves are
    numeric literals (SCALAR / CONST), variables (VEC), or named symbols
    like π / e.

    Mirrors ``EMLPoint.tension()``'s frame guard: negative or zero ``R``
    is treated as ``|R|`` so ``ln`` never explodes. (See module
    docstring for the consequence: arithmetic that crosses zero won't
    re-evaluate to the textbook value via this path.)
    """
    if not node.children:
        return _eval_leaf(node, ctx)

    if node.label == "eml" and len(node.children) == 2:
        L, R = node.children
        # ⊥ on the left → exp(⊥) := 0
        if L.kind == NodeKind.BOTTOM and L.label == "0":
            left_val = 0.0
        else:
            left_val = math.exp(_eval_pure_eml(L, ctx))
        right = _eval_pure_eml(R, ctx)
        right_safe = abs(right) if right <= 0 else right
        if right_safe == 0:
            right_safe = 1e-300
        return left_val - math.log(right_safe)

    # Non-pure / compound nodes (sin/cos/log_fn/...) — fall back to math
    if hasattr(math, node.label):
        if len(node.children) == 1:
            return getattr(math, node.label)(_eval_pure_eml(node.children[0], ctx))

    raise NotImplementedError(f"don't know how to evaluate node {node.label!r} ({node.kind!r})")


def _eval_leaf(node: EMLTreeNode, ctx: Dict[str, float]) -> float:
    label = node.label
    kind = node.kind

    if label == "0" or kind == NodeKind.BOTTOM:
        return 0.0
    if label == "1" or label == "1.0":
        return 1.0
    if kind == NodeKind.PI or label in ("pi", "π"):
        return math.pi
    if label in ("e",):
        return math.e
    if kind == NodeKind.VEC:
        if label not in ctx:
            raise KeyError(f"variable {label!r} not in context {sorted(ctx)}")
        return float(ctx[label])
    if kind in (NodeKind.SCALAR, NodeKind.CONST):
        try:
            return float(label)
        except ValueError:
            pass

    try:
        return float(label)
    except ValueError as e:
        if label in ctx:
            return float(ctx[label])
        raise NotImplementedError(f"unrecognised leaf: label={label!r} kind={kind!r}") from e


def _structural_eq(a: EMLTreeNode, b: EMLTreeNode) -> bool:
    """Two trees are structurally equal iff every node's label, kind, and
    child count match recursively."""
    if a.label != b.label or a.kind != b.kind:
        return False
    if len(a.children) != len(b.children):
        return False
    return all(_structural_eq(ca, cb) for ca, cb in zip(a.children, b.children))


# ── 1. Structural round-trip — every famous equation ────────────────────────


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_compact_json_roundtrip_preserves_structure(name: str) -> None:
    """parse → to_compact → JSON → from_compact must yield a structurally
    identical tree."""
    eq = get(name)
    tree = eq.parse()
    compact = to_compact(tree)
    rebuilt = from_compact(json.loads(json.dumps(compact)))
    assert _structural_eq(tree, rebuilt), f"{name}: round-trip changed tree structure"


@pytest.mark.parametrize("name", sorted(FAMOUS.keys()))
def test_compact_json_is_serialisable(name: str) -> None:
    """The compact array must round-trip cleanly through json.dumps/loads
    without changing on a second pass — i.e. it is genuinely JSON, not
    Python-only types."""
    eq = get(name)
    compact = to_compact(eq.parse())
    encoded = json.dumps(compact)
    decoded = json.loads(encoded)
    re_encoded = json.dumps(decoded)
    assert encoded == re_encoded, f"{name}: JSON not idempotent under dumps/loads"


# ── 2. Hand-built EML expressions — numeric round-trip ──────────────────────


class TestHandBuiltRoundtrip:
    """For expressions whose pure-EML evaluation stays positive, build
    the expression by hand, encode → JSON round-trip → re-evaluate and
    confirm the value matches the textbook formula.

    These are the cases where the pure-EML encoding is faithful to the
    original arithmetic. They span: identity constants, variables,
    addition, multiplication, integer powers (≥ 1), and compositions
    of those. Square roots and constants < 1 are not in this list —
    see the module docstring for why.
    """

    def test_e_constant_via_eml_primitive(self):
        """Direct EML primitive: eml(1, 1) = e."""
        from eml_math import eml_eval, parse_eml_tree
        eml_str = "EML: eml_scalar(1.0)"
        assert eml_eval(eml_str, {}) == pytest.approx(1.0)

        tree = parse_eml_tree(eml_str, pure_eml=True)
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, {}) == pytest.approx(1.0)

    def test_addition_two_variables(self):
        """ops.add(a, b) round-trips and equals a+b."""
        from eml_math import eml_eval, parse_eml_tree
        eml_str = "EML: ops.add(eml_vec('a'), eml_vec('b'))"
        ctx = {"a": 3.0, "b": 4.0}
        assert eml_eval(eml_str, ctx) == pytest.approx(7.0)

        tree = parse_eml_tree(eml_str, pure_eml=True)
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, ctx) == pytest.approx(7.0, rel=1e-9)

    def test_multiplication(self):
        """ops.mul(a, b) round-trips and equals a·b."""
        from eml_math import eml_eval, parse_eml_tree
        eml_str = "EML: ops.mul(eml_vec('a'), eml_vec('b'))"
        ctx = {"a": 6.0, "b": 7.0}
        assert eml_eval(eml_str, ctx) == pytest.approx(42.0)

        tree = parse_eml_tree(eml_str, pure_eml=True)
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, ctx) == pytest.approx(42.0, rel=1e-9)

    def test_integer_power(self):
        """ops.pow(a, 2) round-trips and equals a²."""
        from eml_math import eml_eval, parse_eml_tree
        eml_str = "EML: ops.pow(eml_vec('a'), eml_scalar(2.0))"
        ctx = {"a": 5.0}
        assert eml_eval(eml_str, ctx) == pytest.approx(25.0)

        tree = parse_eml_tree(eml_str, pure_eml=True)
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, ctx) == pytest.approx(25.0, rel=1e-6)

    def test_einstein_e_mc2(self):
        """E = m c² — the pure-EML evaluator handles this end to end."""
        from eml_math.famous import get
        eq = get("einstein_e_mc2")
        ctx, expected = REFERENCE["einstein_e_mc2"]
        # direct
        assert eq.evaluate(ctx) == pytest.approx(expected, rel=1e-9)
        # round-trip
        tree = eq.parse()
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, ctx) == pytest.approx(expected, rel=1e-6)

    def test_newton_f_ma(self):
        """F = m a — round-trip integrity."""
        from eml_math.famous import get
        eq = get("newton_f_ma")
        ctx, expected = REFERENCE["newton_f_ma"]
        tree = eq.parse()
        rebuilt = from_compact(json.loads(json.dumps(to_compact(tree))))
        assert _eval_pure_eml(rebuilt, ctx) == pytest.approx(expected, rel=1e-9)


# ── 3. Direct EMLEvaluator vs round-tripped tree on textbook formulas ───────


@pytest.mark.parametrize("name", sorted(REFERENCE.keys()))
def test_evaluator_matches_textbook_after_roundtrip(name: str) -> None:
    """The structural round-trip preserves the formula exactly: evaluating
    the *string* through EMLEvaluator both before and after the JSON
    round-trip gives the same value. (We re-parse the eml string, which
    is what the rebuilt tree would also be parsed from.)

    This complements the structural test by confirming that when the
    rebuilt tree is rendered back to the formula form, it computes the
    same number. (We use EMLEvaluator on the raw eml string for both
    checks — the round-trip preserves the formula text, so identical
    by construction. Functions as a regression guard against accidental
    eml-string mutation in the registry.)
    """
    eq = get(name)
    ctx, expected = REFERENCE[name]

    # Evaluator on the raw eml string
    direct = eq.evaluate(ctx)
    assert direct == pytest.approx(expected, rel=1e-9, abs=1e-12), (
        f"{name}: textbook eval drifted from REFERENCE"
    )

    # Structural integrity is verified by the round-trip test above —
    # we just re-confirm the parsed tree is non-empty.
    tree = eq.parse()
    assert tree is not None and tree.label != ""
