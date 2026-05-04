"""Tests for the Python-AST ``BinOp`` / ``UnaryOp`` handlers in
``eml_math.tree._ast_to_node``.

Before this layer existed, expressions written with infix operators
(``a + b``, ``a/b``, ``x**2``) fell through ``_ast_to_node`` and produced
a single ``kind='unknown'`` leaf carrying the raw ``ast.dump`` text, which
made every downstream consumer (``to_latex``, ``to_compact``,
``to_dict``, ``flow_png``, ``flow_layout``) unusable for ordinary maths.

These tests pin the round-trip behaviour — anything that types as Python
maths must produce a real, structured tree whose:

* ``kind`` reflects the operator (``compound``),
* ``children`` are the operands in order,
* ``label`` is one of ``add | sub | mul | div | pow | mod``,
* ``to_latex`` / ``to_compact`` agree across re-parses.
"""
from __future__ import annotations

import pytest

from eml_math import from_compact, parse_eml_tree


def _parse(expr: str, *, expand: bool = False):
    return parse_eml_tree(f"EML: {expr}", expand_eml=expand)


# ---------------------------------------------------------------------------
# Operator → label mapping
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expr", "label"),
    [
        ("x + y", "add"),
        ("x - y", "sub"),
        ("x * y", "mul"),
        ("x / y", "div"),
        ("x // y", "div"),     # FloorDiv reuses div
        ("x ** y", "pow"),
        ("x % y", "mod"),
    ],
)
def test_binop_compact_label(expr: str, label: str) -> None:
    t = _parse(expr, expand=False)
    assert t.kind == "compound", f"{expr!r} → kind={t.kind!r}"
    assert t.label == label
    assert len(t.children) == 2
    assert {c.label for c in t.children} == {"x", "y"}


# ---------------------------------------------------------------------------
# Multivariate / nested expressions
# ---------------------------------------------------------------------------
def test_multivariate_no_unknown_leaves() -> None:
    """`(1/x) + (y**3)` was the canonical bug report — must produce a
    real tree with no unknown nodes inside it."""
    t = _parse("(1/x) + (y**3)", expand=False)
    assert t.kind == "compound" and t.label == "add"
    left, right = t.children
    assert (left.label, right.label) == ("div", "pow")
    assert {c.kind for c in (t, left, right)} == {"compound"}
    # ensure no descendant is "unknown"
    seen = []
    stack = [t]
    while stack:
        n = stack.pop()
        seen.append(n)
        stack.extend(n.children)
    assert all(n.kind != "unknown" for n in seen)


def test_unary_plus_is_passthrough() -> None:
    t = _parse("+x", expand=False)
    assert t.kind == "vec" and t.label == "x"


def test_unary_minus_keeps_existing_behaviour() -> None:
    """Pre-existing ``-x`` collapses into a single signed-label leaf;
    keep that contract — anything stricter would break ``test_tree``."""
    t = _parse("-x", expand=False)
    assert t.label == "-x"


# ---------------------------------------------------------------------------
# LaTeX rendering of binops
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expr", "expected_latex"),
    [
        ("x + y",          "x + y"),
        ("x - y",          "x - y"),
        ("(1/x) + y**3",   r"\frac{1}{x} + y^{3}"),
        ("x**2 + y**2",    "x^{2} + y^{2}"),
        ("a*b/c",          r"\frac{a \cdot b}{c}"),
    ],
)
def test_binop_latex(expr: str, expected_latex: str) -> None:
    assert _parse(expr).to_latex() == expected_latex


# ---------------------------------------------------------------------------
# Compact form round-trips
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "expr",
    [
        "x + y",
        "(1/x) + (y**3)",
        "x**2 + y**2",
        "2 * x - 3 * y",
        "a*b/c",
        "sin(x) + cos(x)",
        "sqrt(x**2 + y**2)",
        "(a + b) * (c - d)",
    ],
)
def test_binop_compact_roundtrip(expr: str) -> None:
    """``to_compact`` → ``from_compact`` must be an exact identity."""
    t = _parse(expr, expand=False)
    compact = t.to_compact()
    rebuilt = from_compact(compact)
    assert rebuilt.to_compact() == compact


# ---------------------------------------------------------------------------
# Layout / rendering smoke tests — the very pipelines that were broken.
# ---------------------------------------------------------------------------
def test_binop_renders_to_png() -> None:
    """``flow_png`` should produce real bytes, not raise, for a binop tree."""
    t = _parse("(1/x) + (y**3)")
    png = t.flow_png(direction="down", width=300, height=200)
    assert isinstance(png, (bytes, bytearray)) and len(png) > 100


def test_binop_layout_has_children() -> None:
    """``layout()`` must report nodes for every operand — not a single
    unknown leaf."""
    t = _parse("(1/x) + (y**3)")
    lay = t.layout(direction="down", canvas=(400, 300))
    nodes = lay["nodes"]
    edges = lay["edges"]
    # add(div(1, x), pow(y, 3)) → 7 nodes, 6 edges
    assert len(nodes) == 7
    assert len(edges) == 6
    labels = sorted(n["label"] for n in nodes)
    assert labels == ["1", "3", "add", "div", "pow", "x", "y"]
