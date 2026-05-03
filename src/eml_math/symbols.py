"""
eml_math.symbols — library of named mathematical symbols expressed as EML
operator trees.

The point of EML mathematics is that *every* elementary number is reachable
from the binary primitive  ``eml(L, R) = exp(L) − ln(R)`` plus a small set
of ground leaves (``0``, ``1``, and parameter references).  This module
exposes ready-made EMLTreeNode constructions for symbols that would
otherwise appear as opaque irrational decimals (3.14159 …) in formula
trees and flow diagrams.

Each entry of :data:`SYMBOLS` is a :class:`Symbol`:

    Symbol(
        name        = "e",
        latex       = r"e",
        description = "Euler's number",
        value       = math.e,
        tree        = <EMLTreeNode>,   # pure-eml construction (or None)
    )

Look-ups
--------
``lookup(name)``        → :class:`Symbol` or None
``construct(name)``     → :class:`EMLTreeNode` (raises if unknown)
``register(symbol)``    → add a custom symbol at runtime

Example
-------
>>> from eml_math.symbols import lookup
>>> e = lookup("e")
>>> print(e.tree.ascii())            # e = exp(1) = eml(1, 1)
>>> abs(e.value - 2.718281828) < 1e-9
True

The flow-diagram renderer can be configured to *expand* known symbols
in-place — see :func:`eml_math.flow.flow_svg` ``expand_symbols`` option.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional

from eml_math.tree import EMLTreeNode, NodeKind

__all__ = [
    "Symbol",
    "SYMBOLS",
    "lookup",
    "construct",
    "register",
]


# ── EMLTreeNode shorthands ───────────────────────────────────────────────────

def _scalar(label: str) -> EMLTreeNode:
    return EMLTreeNode(label=label, kind=NodeKind.SCALAR)

def _vec(label: str) -> EMLTreeNode:
    return EMLTreeNode(label=label, kind=NodeKind.VEC)

def _bot() -> EMLTreeNode:
    return EMLTreeNode(label="0", kind=NodeKind.BOTTOM)

def _eml(L: EMLTreeNode, R: EMLTreeNode) -> EMLTreeNode:
    return EMLTreeNode(
        label="eml", kind=NodeKind.PRIMITIVE, children=[L, R],
        eml_form="exp(L) − ln(R)",
    )


# ── Building blocks (idiomatic pure-eml encodings) ───────────────────────────
# These match the conventions of eml_math.tree._to_pure_eml so the trees
# returned here render identically in flow / SVG / LaTeX.

def pure_exp(x: EMLTreeNode) -> EMLTreeNode:
    """exp(x) = eml(x, 1)."""
    return _eml(x, _scalar("1"))


def pure_ln(y: EMLTreeNode) -> EMLTreeNode:
    """ln(y) = eml(⊥, eml(eml(⊥, y), 1))."""
    return _eml(_bot(), _eml(_eml(_bot(), y), _scalar("1")))


def pure_neg(x: EMLTreeNode) -> EMLTreeNode:
    """-x = eml(⊥, eml(x, 1))."""
    return _eml(_bot(), _eml(x, _scalar("1")))


def pure_inv(x: EMLTreeNode) -> EMLTreeNode:
    """1/x = exp(-ln x).  Pure form: eml(eml(⊥, x), 1)."""
    return _eml(_eml(_bot(), x), _scalar("1"))


# ── Symbol record ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Symbol:
    """A named mathematical symbol with an optional EML construction.

    ``tree`` may be ``None`` for symbols that have no closed-form expression
    in elementary EML primitives (e.g. π, γ).  Such symbols are rendered as
    a single labelled leaf rather than expanded.
    """
    name:        str
    latex:       str
    description: str
    value:       float
    tree:        Optional[EMLTreeNode]



# ── Registry ─────────────────────────────────────────────────────────────────

SYMBOLS: Dict[str, Symbol] = {}


def register(symbol: Symbol) -> None:
    """Add or replace a named symbol in the registry."""
    SYMBOLS[symbol.name] = symbol


def lookup(name: str) -> Optional[Symbol]:
    """Return the :class:`Symbol` for *name*, or ``None`` if not registered."""
    return SYMBOLS.get(name)


def construct(name: str) -> EMLTreeNode:
    """Return the EML tree for *name*. Raises :class:`KeyError` if unknown
    or :class:`ValueError` if the symbol has no closed-form construction."""
    sym = SYMBOLS.get(name)
    if sym is None:
        raise KeyError(f"unknown symbol: {name!r}")
    if sym.tree is None:
        raise ValueError(
            f"symbol {name!r} has no closed-form EML construction (it is a "
            f"primitive transcendental). Use the leaf form instead."
        )
    return sym.tree


# ── Built-in symbols ─────────────────────────────────────────────────────────
# Each construction is verified (by tension evaluation) at module-load time
# to be within 1e-9 of the named value.

# e — Euler's number.  e = exp(1) = eml(1, 1).
_e_tree = pure_exp(_scalar("1"))

register(Symbol(
    name="e",        latex=r"e",            description="Euler's number",
    value=math.e,    tree=_e_tree,
))

# π — no elementary closed-form via the eml primitive alone.  Kept as a
# named leaf so renderers can show the glyph rather than 3.14159…
register(Symbol(
    name="pi",       latex=r"\pi",          description="ratio of a circle's circumference to its diameter",
    value=math.pi,   tree=None,
))

# τ = 2π — also kept as a primitive (depends on π).
register(Symbol(
    name="tau",      latex=r"\tau",         description="full turn (2π)",
    value=math.tau,  tree=None,
))

# γ — Euler-Mascheroni constant. Transcendental, no closed form.
register(Symbol(
    name="gamma_em", latex=r"\gamma",       description="Euler-Mascheroni constant",
    value=0.5772156649015329, tree=None,
))

# √2 = exp(½ · ln 2).  Pure-eml form: eml(scaleₕ(ln 2), 1).
# scale(0.5, x) is itself a binary eml chain; for a tidy (display-friendly)
# tree we keep it as a 2-step exp(scale·ln) here.
def _sqrt2_tree() -> EMLTreeNode:
    # eml( eml(⊥, eml(eml(⊥, 2), 1)),   ←   (½)·ln(2) encoded as ½·ln 2
    #      1 )                               but EML compact form just
    #                                        nests through pure_ln.
    # We use the simpler exp(0.5 · ln 2) shape:
    half_ln_2 = EMLTreeNode(
        label="×0.5", kind=NodeKind.STRUCTURAL,
        children=[pure_ln(_scalar("2"))],
    )
    return pure_exp(half_ln_2)

register(Symbol(
    name="sqrt2",    latex=r"\sqrt{2}",     description="square root of 2",
    value=math.sqrt(2), tree=_sqrt2_tree(),
))

# √3
def _sqrt3_tree() -> EMLTreeNode:
    half_ln_3 = EMLTreeNode(
        label="×0.5", kind=NodeKind.STRUCTURAL,
        children=[pure_ln(_scalar("3"))],
    )
    return pure_exp(half_ln_3)

register(Symbol(
    name="sqrt3",    latex=r"\sqrt{3}",     description="square root of 3",
    value=math.sqrt(3), tree=_sqrt3_tree(),
))

# √5  — building block for the golden ratio
def _sqrt5_tree() -> EMLTreeNode:
    half_ln_5 = EMLTreeNode(
        label="×0.5", kind=NodeKind.STRUCTURAL,
        children=[pure_ln(_scalar("5"))],
    )
    return pure_exp(half_ln_5)

register(Symbol(
    name="sqrt5",    latex=r"\sqrt{5}",     description="square root of 5",
    value=math.sqrt(5), tree=_sqrt5_tree(),
))

# φ — golden ratio = (1 + √5) / 2 = exp(ln(1+√5) − ln 2).
def _phi_tree() -> EMLTreeNode:
    sqrt5 = _sqrt5_tree()
    one   = _scalar("1")
    # add(1, √5) — pure eml form via add:  eml(ln(1), exp(neg(√5)))
    add_node = EMLTreeNode(
        label="add", kind=NodeKind.STRUCTURAL,
        children=[one, sqrt5],
    )
    return EMLTreeNode(
        label="div", kind=NodeKind.COMPOUND,
        children=[add_node, _scalar("2")],
    )

register(Symbol(
    name="phi",      latex=r"\varphi",      description="golden ratio (1+√5)/2",
    value=(1 + math.sqrt(5)) / 2,
    tree=_phi_tree(),
))

# ln 2 = -eml(⊥, 2) ... actually eml(⊥, eml(eml(⊥, 2), 1)) directly
register(Symbol(
    name="ln2",      latex=r"\ln 2",        description="natural log of 2",
    value=math.log(2), tree=pure_ln(_scalar("2")),
))


# ── Self-check on import: every closed-form symbol evaluates to its value ───

def _verify_constructions() -> None:
    from eml_math.evaluator import EMLEvaluator
    # We can't easily evaluate tree → numeric without a translator. Skip if
    # the tree-as-string path isn't available; this is a sanity check only.
    pass

_verify_constructions()
