"""Pin the user-input helpers exposed at the package level.

``normalize_input`` is the single rewrite step that maps caret / Unicode
maths onto Python form so eml-math's parser accepts what people actually
type. ``tree_to_python`` is the inverse direction — turn a parsed tree
(usually a famous-equation tree) back into a clickable Python expression
suitable for an input field.

Both are designed to be downstream-safe: idempotent, deterministic, no
side effects.
"""
from __future__ import annotations

import pytest

from eml_math import (
    FAMOUS,
    get_famous,
    normalize_input,
    parse_eml_tree,
    tree_to_python,
)


# ---------------------------------------------------------------------------
# normalize_input
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("a^b",                "a**b"),
        ("m^c**2",             "m**c**2"),
        ("a×b",                "a*b"),
        ("π·r",                "π*r"),
        ("a÷b",                "a/b"),
        ("a−b",                "a-b"),
        ("x²",                 "x**2"),
        ("y³ + x⁵",            "y**3 + x**5"),
        ("x₁ + y₂",            "x1 + y2"),
        ("a^b + 2×c − d/e",    "a**b + 2*c - d/e"),
        ("a**b",               "a**b"),
        ("a ** b",             "a ** b"),
        ("",                   ""),
    ],
)
def test_normalize_input(raw: str, expected: str) -> None:
    assert normalize_input(raw) == expected


def test_normalize_input_is_idempotent() -> None:
    """Running twice must give the same answer — important so callers can
    apply it defensively without worrying about double-substitution."""
    samples = ["a^b", "x²+y³", "π·r²", "1 + 2", "sin(x)+cos(x)"]
    for s in samples:
        once = normalize_input(s)
        twice = normalize_input(once)
        assert once == twice, f"{s!r} not idempotent: {once!r} → {twice!r}"


def test_normalize_input_none_safe() -> None:
    assert normalize_input(None) is None    # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# tree_to_python
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("x + y",        "(x + y)"),
        ("x - y",        "(x - y)"),
        ("x * y",        "(x * y)"),
        ("x / y",        "(x / y)"),
        ("x ** 2",       "(x ** 2)"),
        ("sin(x)",       "sin(x)"),
        ("sqrt(x*y)",    "sqrt((x * y))"),
    ],
)
def test_tree_to_python_compact(expr: str, expected: str) -> None:
    """Round-trip through compact-mode parse and back."""
    t = parse_eml_tree(f"EML: {expr}", expand_eml=False)
    assert tree_to_python(t) == expected


def test_tree_to_python_compiles() -> None:
    """Output must always be a valid Python expression."""
    for name in FAMOUS:
        fe = get_famous(name)
        head = fe.eml.split(" — ", 1)[0]
        t = parse_eml_tree(head, expand_eml=False)
        py = tree_to_python(t)
        compile(py, f"<famous:{name}>", "eval")
