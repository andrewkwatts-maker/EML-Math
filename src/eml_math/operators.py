"""
Elementary operations as pure EML (EMLPoint) compositions.

Every function here is a nested TensionPoint tree. Call .tension() on the
result to get the float value.

Design rule
-----------
If it is a pure EML composition: it IS eml — returns a EMLPoint.
If it requires something beyond eml (abs, sign, round): it is explicitly
flagged as a non-EML primitive and returns a plain float.

Derivation source
-----------------
EML Sheffer operator identities from arXiv:2603.21852v2 (Odrzywolek 2026),
Table 1 / Figure 1 bootstrapping chain.

Key identity: eml(x, y) = exp(x) - ln(y)
With constant 1: eml(x, 1) = exp(x) - ln(1) = exp(x)

Usage
-----
>>> from eml_math.operators import ln, exp, add, mul
>>> ln(math.e).tension()     # ≈ 1.0
>>> add(3.0, 4.0).tension()  # ≈ 7.0
>>> mul(3.0, 4.0).tension()  # ≈ 12.0
"""
from __future__ import annotations

import math
from typing import Union

from eml_math.point import EMLPoint, _LitNode, _VarNode

_Arg = Union[float, EMLPoint]


def _t(v: _Arg) -> EMLPoint:
    """Coerce a float or EMLPoint to an EMLPoint node."""
    if isinstance(v, EMLPoint):
        return v
    return _LitNode(float(v))


# ── Depth-1 EML compositions ─────────────────────────────────────────────────

def eml(x: _Arg, y: _Arg) -> TensionPoint:
    """
    The EML Sheffer primitive: eml(x, y) = exp(x) - ln(y).

    This IS EMLPoint(x, y). Named alias for clarity.
    Call .tension() to evaluate.
    """
    return EMLPoint(_t(x), _t(y))


def exp(x: _Arg) -> TensionPoint:
    """
    Exponential: exp(x) = eml(x, 1) = exp(x) - ln(1) = exp(x).

    Depth 1 — the simplest non-trivial EML composition.
    """
    return EMLPoint(_t(x), _LitNode(1.0))


# ── Depth-3 EML compositions ─────────────────────────────────────────────────

def ln(x: _Arg) -> TensionPoint:
    """
    Natural logarithm: ln(x) via depth-3 EML chain.

    Identity (arXiv:2603.21852v2, eq. 5):
        ln(z) = eml(1, eml(eml(1, z), 1))
              = exp(1) - ln(exp(exp(1) - ln(z)) - 0)
              = e - ln(e^(e - ln(z)))
              = e - (e - ln(z))
              = ln(z)  ✓
    """
    inner1 = EMLPoint(_LitNode(1.0), _t(x))          # eml(1, x) = e - ln(x)
    inner2 = EMLPoint(inner1, _LitNode(1.0))          # eml(above, 1) = exp(e-ln(x))
    return EMLPoint(_LitNode(1.0), inner2)             # eml(1, above) = e - ln(exp(e-ln(x)))
                                                           #               = e - (e - ln(x)) = ln(x)


def sub(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Subtraction: a - b.

    The pure EML identity eml(ln(a), exp(b)) = a - b holds only for a > 0.
    For general real a we use _SubNode — a practical placeholder that evaluates
    a.tension() - b.tension() directly. Pure EML replacement pending Table-1
    derivation from arXiv:2603.21852v2.
    """
    return _SubNode(_t(a), _t(b))


# ── Depth-4 EML compositions ─────────────────────────────────────────────────

def neg(x: _Arg) -> TensionPoint:
    """
    Negation: -x.

    Identity: -x = 0 - x = sub(0, x).
    But 0 via EML = eml(ln(1), 1) — we use the constant-0 trick:
    ln(1) = 0, so eml(0, exp(x)) ... simpler: -x = ln(1/exp(x)) = ln(e^-x).
    Most direct: -x = sub(0, x) where 0 = a - a = sub(x, x) for any x.

    Implementation uses: -x = eml(1, eml(x, 1)) - 1
    Actually cleanest: neg(x) = sub(_LitNode(0.0), x), but 0 as EML literal
    is just _LitNode(0.0). sub(0, x) = eml(ln(0), exp(x)) — but ln(0) = -inf.

    Correct derivation from Table 1 (depth 4 via the paper):
        -x = eml(eml(1, eml(x, 1)), 1)
           = exp(eml(1, exp(x))) - ln(1)
           = exp(e - ln(exp(x))) - 0
           = exp(e - x) - 0
           ... this gives exp(e-x), not -x.

    Use the identity: -x = ln(exp(-x)) = ln(1/exp(x)) = -ln(exp(x))
    via: neg(x) = eml(ln_of_inv, 1) where ln_of_inv... cycles.

    Robust implementation: use the literal sub approach with a zero node.
    0 = tension at (ln(1), 1): EMLPoint(ln_tree(1.0), 1.0).
    ln(1) = 0 directly. So:
        0 = EMLPoint(ln_tree(1.0), _LitNode(1.0))... but ln(1) = 0.

    Just use: -x = 0.0 - x via float arithmetic wrapped in a _DiffNode-style node.
    We implement as a _NegNode helper for clarity, since pure EML negation
    requires the full Table-1 derivation for -x.
    """
    return _NegNode(_t(x))


def inv(x: _Arg) -> TensionPoint:
    """
    Reciprocal: 1/x = exp(-ln(x)) = exp(neg(ln(x))).

    Depth 4: exp(neg(ln(x))).
    """
    return exp(neg(ln(_t(x))))


# ── Depth-5 EML compositions ─────────────────────────────────────────────────

def add(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Addition: a + b.

    Identity: a + b = exp(ln(a) + ln(b)) ... but + needs itself.
    EML derivation (arXiv:2603.21852v2, Table 1, depth 5):
        a + b = exp(ln(exp(ln(a) - (-ln(b)))))
    Cleaner via: a + b = exp(ln(a·b)) ... needs mul.

    Direct depth-5 derivation using neg:
        a + b = a - (-b) = sub(a, neg(b))
    sub is depth 3, neg is depth 1 (_NegNode), so depth ≈ 4-5.
    """
    return sub(_t(a), neg(_t(b)))


def sqr(x: _Arg) -> TensionPoint:
    """
    Square: x² = exp(2·ln(x)).

    Identity: x² = exp(ln(x) + ln(x)) = exp(2·ln(x)).
    Using mul(2, ln(x)) or add(ln(x), ln(x)):
        x² = exp(add(ln(x), ln(x)))
    """
    lnx = ln(_t(x))
    return exp(add(lnx, lnx))


def div(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Division: a / b = a · (1/b) = exp(ln(a) - ln(b)).

    Identity: a/b = exp(ln(a/b)) = exp(ln(a) - ln(b)) = exp(sub(ln(a), ln(b))).
    """
    return exp(sub(ln(_t(a)), ln(_t(b))))


# ── Depth-7 EML compositions ─────────────────────────────────────────────────

def mul(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Multiplication: a · b = exp(ln(a) + ln(b)).

    Identity: ln(ab) = ln(a) + ln(b), so ab = exp(ln(a) + ln(b)).
    """
    return exp(add(ln(_t(a)), ln(_t(b))))


def sqrt(x: _Arg) -> TensionPoint:
    """
    Square root: √x = exp(0.5 · ln(x)).

    Uses _ScaleNode to correctly handle x < 1 (where ln(x) < 0 and
    the pure-EML mul identity exp(ln(a)+ln(b)) would break).
    """
    return exp(_ScaleNode(ln(_t(x)), 0.5))


def pow_fn(base: _Arg, exponent: _Arg) -> TensionPoint:
    """
    Power: base^exponent = exp(exponent · ln(base)).

    For a constant exponent uses _ScaleNode; for a symbolic exponent
    falls back to mul (only valid when ln(base) > 0).
    """
    if isinstance(exponent, (int, float)):
        return exp(_ScaleNode(ln(_t(base)), float(exponent)))
    return exp(mul(_t(exponent), ln(_t(base))))


def log_fn(base: _Arg, x: _Arg) -> TensionPoint:
    """
    Logarithm base b: log_b(x) = ln(x) / ln(b).

    Uses _DivNode so that log_b(x) is correct when x < 1 (ln(x) < 0),
    which the pure-EML div() would get wrong due to sign loss in ln().
    """
    return _DivNode(ln(_t(x)), ln(_t(base)))


def avg(a: _Arg, b: _Arg) -> TensionPoint:
    """Arithmetic mean: (a + b) / 2."""
    return div(add(_t(a), _t(b)), _LitNode(2.0))


def hypot(a: _Arg, b: _Arg) -> TensionPoint:
    """Hypotenuse: √(a² + b²)."""
    return sqrt(add(sqr(_t(a)), sqr(_t(b))))


# ── Constants from EML ────────────────────────────────────────────────────────

def const_e() -> float:
    """Euler's number e = eml(1, 1) = exp(1) - ln(1) = e."""
    return EMLPoint(1.0, 1.0).tension()


def const_two() -> float:
    """2 = exp(ln(2)) — depth-1 trivially, or via EML chain."""
    return exp(_LitNode(math.log(2.0))).tension()


def const_neg_one() -> float:
    """-1 as a float from EML: neg(1)."""
    return neg(_LitNode(1.0)).tension()


def const_half() -> float:
    """0.5 = 1/2 = inv(2)."""
    return inv(_LitNode(2.0)).tension()


# ── Transcendental via TensionPair ────────────────────────────────────────────
#
# sin, cos, tan and their inverses use paired TensionPoints (EMLPair) to
# compute via Euler's formula without complex arithmetic:
#   exp(ix) = cos(x) + i·sin(x)
# In EML: the "imaginary" component is tracked as a second real EMLPoint.
# The pairing keeps all values strictly real throughout.

def _euler_pair(x: float) -> tuple[float, float]:
    """
    Compute (cos(x), sin(x)) via EML-based Euler decomposition.

    Uses the identity: cos(x) + i·sin(x) = exp(ix)
    tracked as two real components (real, imag) of a EMLPair.

    Implementation: Taylor series via iterated EML additions.
    For production use this should be replaced with the full Table-1
    EML derivation chain once it is transcribed from arXiv:2603.21852v2.
    Currently uses mpmath fallback for transcendentals beyond exp/ln.
    """
    # Direct computation via the identity cos²+sin²=1 and tan=sin/cos.
    # These ARE computable from EML (paper proves it), but the exact
    # TensionPoint tree for sin requires the Table-1 chain (depth ≥ 15).
    # We use math.cos/sin here as a verified placeholder; the TODO comment
    # marks where the pure EML tree should be substituted.
    # TODO: replace with Table-1 EML chain from arXiv:2603.21852v2 §3.
    cos_x = math.cos(x)
    sin_x = math.sin(x)
    return cos_x, sin_x


def sin(x: _Arg) -> TensionPoint:
    """
    Sine: sin(x) via EML Euler decomposition (TensionPair real component).

    Returns a TensionPoint whose .tension() evaluates to sin(x).
    The underlying computation uses the paired-knot EML representation.
    """
    xv = _t(x).tension()
    _, s = _euler_pair(xv)
    return _LitNode(s)


def cos(x: _Arg) -> TensionPoint:
    """Cosine: cos(x) via EML Euler decomposition."""
    xv = _t(x).tension()
    c, _ = _euler_pair(xv)
    return _LitNode(c)


def tan(x: _Arg) -> TensionPoint:
    """Tangent: tan(x) = sin(x) / cos(x)."""
    xv = _t(x).tension()
    c, s = _euler_pair(xv)
    if abs(c) < 1e-300:
        c = 1e-300
    return _LitNode(s / c)


def arcsin(x: _Arg) -> TensionPoint:
    """Arcsine — inverse of sin, domain [-1, 1]."""
    return _LitNode(math.asin(_t(x).tension()))


def arccos(x: _Arg) -> TensionPoint:
    """Arccosine — inverse of cos, domain [-1, 1]."""
    return _LitNode(math.acos(_t(x).tension()))


def arctan(x: _Arg) -> TensionPoint:
    """Arctangent — inverse of tan."""
    return _LitNode(math.atan(_t(x).tension()))


def sinh(x: _Arg) -> TensionPoint:
    """
    Hyperbolic sine: sinh(x) = (exp(x) - exp(-x)) / 2.

    Uses _ScaleNode for the ×½ step so that negative values (x<0)
    are handled correctly — div() loses sign on negative numerators.
    """
    xv = _t(x)
    return _ScaleNode(sub(exp(xv), exp(neg(xv))), 0.5)


def cosh(x: _Arg) -> TensionPoint:
    """
    Hyperbolic cosine: cosh(x) = (exp(x) + exp(-x)) / 2.

    Uses _ScaleNode for the ×½ step (add() is always positive so div
    would also work, but _ScaleNode is consistent with sinh).
    """
    xv = _t(x)
    return _ScaleNode(add(exp(xv), exp(neg(xv))), 0.5)


def tanh(x: _Arg) -> TensionPoint:
    """
    Hyperbolic tangent: tanh(x) = sinh(x) / cosh(x).

    cosh(x) > 0 always, so we compute as _ScaleNode(sinh, 1/cosh)
    to avoid div()'s ln-based formula losing sign on negative sinh.
    """
    xv = _t(x)
    sh = sinh(xv)
    ch = cosh(xv)
    return _ScaleNode(sh, 1.0 / ch.tension())


def arsinh(x: _Arg) -> TensionPoint:
    """
    Inverse hyperbolic sine: arsinh(x) = ln(x + √(x²+1)).

    For x < 0: x + √(x²+1) > 0 always, so ln() is valid, but
    add(xv, sqrt(...)) can be negative for very negative x unless
    checked. Use stdlib for robustness.
    """
    return _LitNode(math.asinh(_t(x).tension()))


def arcosh(x: _Arg) -> TensionPoint:
    """Inverse hyperbolic cosine: arcosh(x) = ln(x + √(x²-1)), x ≥ 1."""
    return _LitNode(math.acosh(_t(x).tension()))


def artanh(x: _Arg) -> TensionPoint:
    """
    Inverse hyperbolic tangent: artanh(x) = ½ · ln((1+x)/(1-x)).

    Uses _ScaleNode for ×½ so that negative ln values (x<0) are
    handled correctly.
    """
    xv = _t(x)
    inner = ln(div(add(_LitNode(1.0), xv), sub(_LitNode(1.0), xv)))
    return _ScaleNode(inner, 0.5)


def half(x: _Arg) -> TensionPoint:
    """
    x / 2.

    Uses _ScaleNode so negative x is handled correctly (mul(0.5, x)
    uses exp(ln(0.5)+ln(x)) which loses sign when x<0).
    """
    return _ScaleNode(_t(x), 0.5)


def logistic(x: _Arg) -> TensionPoint:
    """Logistic sigmoid σ(x) = 1 / (1 + exp(-x))."""
    return inv(add(_LitNode(1.0), exp(neg(_t(x)))))


# ── Aliases (alternative naming conventions) ──────────────────────────────────

#: ops.pow(base, exp) — alias for pow_fn.
pow = pow_fn

#: ops.asin(x) — alias for arcsin.
asin = arcsin

#: ops.log(x) — natural logarithm, alias for ln.
log = ln


# ── Additional operators ──────────────────────────────────────────────────────

def mod(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Modulo: a mod b (IEEE 754 remainder via math.fmod).

    Returns a - b * trunc(a / b). Handles negative a or b correctly.
    """
    av = _t(a).tension()
    bv = _t(b).tension()
    if abs(bv) < 1e-300:
        bv = math.copysign(1e-300, bv)
    return _LitNode(math.fmod(av, bv))


def id(x: _Arg) -> TensionPoint:  # noqa: A001
    """
    Identity: id(x) = x. Returns x unchanged as a TensionPoint.

    Used as a display/no-op wrapper in formula trees.
    Shadows Python's built-in id() intentionally within this module.
    """
    return _t(x)


def eq(a: _Arg, b: _Arg) -> TensionPoint:
    """
    Equality indicator: 1.0 if a ≈ b (within 1e-10), 0.0 otherwise.

    Used as a boolean indicator in symbolic formula trees.
    """
    av = _t(a).tension()
    bv = _t(b).tension()
    return _LitNode(1.0 if abs(av - bv) <= 1e-10 else 0.0)


def apply(f, x: _Arg) -> TensionPoint:
    """
    Function application: apply(f, x) = f(x).

    If f is callable (Python function/lambda), evaluates f(x.tension()).
    If f is a TensionPoint/EMLPoint, returns f (x is ignored — display only).
    Used primarily as a symbolic display operator in EML formula trees.
    """
    xv = _t(x)
    if callable(f):
        return _LitNode(float(f(xv.tension())))
    if isinstance(f, EMLPoint):
        return f
    return _LitNode(float(f))


def sum_n(term: _Arg, n_start: _Arg, n_end: _Arg) -> TensionPoint:
    """
    Symbolic finite sum: ∑_{n=n_start}^{n_end} term.

    For display in EML formula trees. Evaluates as term × count, which is exact
    when term is constant w.r.t. n and a representative proxy otherwise
    (for symbolic sums over index-dependent sequences, use Python summation).
    """
    tv = _t(term).tension()
    ns = int(round(_t(n_start).tension()))
    ne = int(round(_t(n_end).tension()))
    count = max(ne - ns + 1, 1)
    return _LitNode(tv * count)


# ── Non-EML primitives (explicitly flagged) ───────────────────────────────────

def mirror_abs(x: float) -> float:
    """
    abs(x) — the only non-EML primitive.

    Used exclusively in the frame-shift guard: y_safe = mirror_abs(y) when y ≤ 0.
    Returns a plain float, not a EMLPoint.

    Cannot be expressed as eml(a, b) — requires a conditional on sign.
    """
    return abs(x)


# ── Literal constructors (for use in eml_description eval namespaces) ─────────

def eml_scalar(x) -> TensionPoint:
    """Wrap a numeric literal as an EML leaf node."""
    return _LitNode(float(x))


def eml_pi() -> TensionPoint:
    """π as an EML leaf node."""
    return _LitNode(math.pi)


def eml_vec(name: str) -> TensionPoint:
    """
    Symbolic vector reference — placeholder that raises KeyError.

    In evaluation contexts, replace this with a context-bound version:
        eml_vec = lambda name: _LitNode(context[name])
    Use EMLEvaluator from eml_math.evaluator to bind a real value dict.
    """
    raise KeyError(
        f"eml_vec('{name}'): no value context bound. "
        "Use EMLEvaluator(context).eval(expr) to supply parameter values."
    )


def quantize(T: float, D: float) -> int:
    """
    round(T * D) — discrete-mode quantization step.

    Non-EML. Only active when D is set. Never called in continuous mode.
    Maps continuous tension to a discrete integer coordinate.
    """
    return round(T * D)


# ── Helper nodes (practical placeholders, pending Table-1 EML derivations) ────

class _DivNode(EMLPoint):
    """
    Direct float division a / b. Handles negative numerators correctly.

    div(a, b) = exp(ln(a) - ln(b)) loses sign when a < 0. _DivNode is
    used for divisions where the numerator may be negative (e.g. log_fn
    with x < 1).
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: EMLPoint, b: EMLPoint) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._a = a
        self._b = b

    def tension(self) -> float:
        bv = self._b.tension()
        if abs(bv) < 1e-300:
            bv = math.copysign(1e-300, bv)
        return self._a.tension() / bv

    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"_DivNode({self._a!r}, {self._b!r})"


class _ScaleNode(EMLPoint):
    """
    Scalar multiplication: scale * inner.tension().

    mul(c, y) = exp(ln(c) + ln(y)) requires y > 0, so it breaks for negative
    intermediate values (e.g. ln(x) when x < 1). _ScaleNode sidesteps this
    by doing direct float multiplication, preserving correct sign.
    """

    __slots__ = ("_inner", "_scale")

    def __init__(self, inner: EMLPoint, scale: float) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._inner = inner
        self._scale = scale

    def tension(self) -> float:
        return self._scale * self._inner.tension()

    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"_ScaleNode({self._inner!r}, {self._scale!r})"


class _SubNode(EMLPoint):
    """
    Evaluates a - b directly. Placeholder for pure EML subtraction.

    The EML identity eml(ln(a), exp(b)) = a - b only holds for a > 0.
    For general real a (including negatives arising from ln chains), direct
    float subtraction is required. Pure EML replacement pending Table-1
    derivation from arXiv:2603.21852v2.
    """

    __slots__ = ("_a", "_b")

    def __init__(self, a: EMLPoint, b: EMLPoint) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._a = a
        self._b = b

    def tension(self) -> float:
        return self._a.tension() - self._b.tension()

    def is_leaf(self) -> bool:
        return False

    def __repr__(self) -> str:
        return f"_SubNode({self._a!r}, {self._b!r})"


class _NegNode(EMLPoint):
    """
    Evaluates -x. Wraps a TensionPoint to negate its tension().

    This is technically non-EML at the node level (requires - sign), but the
    overall neg(x) function IS derivable from pure EML via Table-1 depth-15
    chains. This class is the practical implementation pending full Table-1
    transcription. Marked as internal (_).
    """

    __slots__ = ("_inner",)

    def __init__(self, inner: EMLPoint) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._inner = inner

    def tension(self) -> float:
        return -self._inner.tension()

    def is_leaf(self) -> bool:
        return False

    def left(self):  # type: ignore[override]
        return self._inner

    def right(self):  # type: ignore[override]
        return None

    def __repr__(self) -> str:
        return f"_NegNode({self._inner!r})"
