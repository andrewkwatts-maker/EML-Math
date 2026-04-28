"""
TensionPoint — the universal EML computation node.

EMLPoint(x, y) computes eml(x, y) = exp(x) - ln(y) via .tension().
Coordinates can be floats or other TensionPoints, making nesting natural:

    e   = EMLPoint(1, 1).tension()
    exp = EMLPoint(x, 1).tension()
    ln  = EMLPoint(1, EMLPoint(EMLPoint(1, x), 1)).tension()

This is both the state representation (EML Axioms 5-8) and the expression tree
node — no separate AST class is needed.

Modes
-----
- Continuous (default, D=None): smooth float arithmetic, frame-shift guard only.
- Discrete (opt-in, D=float): quantizes via round(T * D); enables Ddx/dsy snap.
"""
from __future__ import annotations

import math
from typing import Optional, Union

from eml_math.constants import OVERFLOW_THRESHOLD, PLANCK_D

try:
    from eml_math import eml_core as _core
    _RUST_POINT = True
except ImportError:
    _RUST_POINT = False

_Coord = Union[float, "EMLPoint"]


class EMLPoint:
    """
    A paired coordinate node computing eml(x, y) = exp(x) - ln(y).

    Parameters
    ----------
    x : float | TensionPoint, default 0.0
        Direct growth coordinate. May be another TensionPoint (nested EML).
    y : float | TensionPoint, default 1.0
        Mirror coordinate. May be another TensionPoint (nested EML).
        Must evaluate to > 0; if ≤ 0, the frame-shift guard applies |y|.
    D : float | None
        Quantization scale. None = continuous mode (default).
        Set to a positive float to enable discrete (quantized) mode.

    Examples
    --------
    >>> EMLPoint().tension()              # eml(0, 1) = 1
    1.0
    >>> EMLPoint(1, 1).tension()          # e
    2.718281828459045
    >>> EMLPoint(2, 1).tension()          # exp(2)
    7.38905609893065
    >>> # ln(e) via nested knot
    >>> EMLPoint(1, EMLPoint(EMLPoint(1, math.e), 1)).tension()
    1.0
    """

    __slots__ = ("_x", "_y", "_D", "_prev_x", "_prev_y")

    def __init__(
        self,
        x: _Coord = 0.0,
        y: _Coord = 1.0,
        D: Optional[float] = None,
    ) -> None:
        self._x: _Coord = x
        self._y: _Coord = y
        self._D: Optional[float] = D
        # Previous values — used only in discrete mode for Ddx/dsy snap detection.
        self._prev_x: Optional[float] = None
        self._prev_y: Optional[float] = None

    # ── coordinate access (evaluates nested TensionPoints lazily) ─────────────

    @property
    def x(self) -> float:
        """Direct growth coordinate, evaluated if nested."""
        if isinstance(self._x, EMLPoint):
            return self._x.tension()
        return float(self._x)

    @property
    def y(self) -> float:
        """Mirror coordinate, evaluated if nested."""
        if isinstance(self._y, EMLPoint):
            return self._y.tension()
        return float(self._y)

    @property
    def D(self) -> Optional[float]:
        """Quantization scale. None = continuous mode."""
        return self._D

    # ── core computation ──────────────────────────────────────────────────────

    def tension(self) -> float:
        """
        Compute eml(x, y) = exp(x) - ln(y).

        This is the EML Sheffer operator (arXiv:2603.21852v2),
        the fundamental tension formula (Axiom 5).

        The frame-shift guard makes this always real: if y ≤ 0, uses |y|.

        Returns
        -------
        float
            Always finite unless x exceeds OVERFLOW_THRESHOLD (Slipping Wheel).
        """
        xv = self.x
        yv = self.y
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)  # Slipping Wheel: self-mirror dampening
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300  # floor to prevent ln(0) = -inf
        return math.exp(xv) - math.log(y_safe)

    # ── iteration ──────────────────────────────────────────────────────

    def iterate(self) -> "EMLPoint":
        """One EML iteration step: x_new = y_safe, y_new = tension().

        ``y_safe = |y|`` if ``y ≤ 0`` (domain guard for ``ln``).
        ``x`` is dampened to ``ln(x)`` if it exceeds OVERFLOW_THRESHOLD
        (otherwise ``exp(x)`` would overflow IEEE-754 floats).

        In discrete mode (``D`` set) the new ``y`` is quantised to the
        nearest multiple of ``1/D``.

        Returns a new EMLPoint; the receiver is immutable.
        """
        xv = self.x
        yv = self.y

        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300

        T = math.exp(xv) - math.log(y_safe)
        x_new = y_safe
        y_new = T

        if self._D is not None:
            b_new = round(y_new * self._D)
            y_new = b_new / self._D
            nxt = EMLPoint(x_new, y_new, D=self._D)
            nxt._prev_x = xv
            nxt._prev_y = y_safe
        else:
            nxt = EMLPoint(x_new, y_new, D=None)

        return nxt

    # Backwards-compat aliases (deprecated — use .iterate())
    def mirror_pulse(self) -> "EMLPoint":
        """Deprecated alias for :meth:`iterate`."""
        return self.iterate()

    def pulse(self) -> "EMLPoint":
        """Deprecated alias for :meth:`iterate`."""
        return self.iterate()

    def eml(self) -> float:
        """Alias for :meth:`tension` — ``eml(x, y) = exp(x) − ln(y)``."""
        return self.tension()

    # ── tree introspection (for converter and differentiation) ────────────────

    def is_leaf(self) -> bool:
        """True if both coordinates are plain floats (no nested TensionPoints)."""
        return not isinstance(self._x, EMLPoint) and not isinstance(self._y, EMLPoint)

    def left(self) -> _Coord:
        """The raw left (x) input — float or nested EMLPoint."""
        return self._x

    def right(self) -> _Coord:
        """The raw right (y) input — float or nested EMLPoint."""
        return self._y

    # ── symbolic differentiation ──────────────────────────────────────────────

    def diff(self, var: str) -> "EMLPoint":
        """
        Differentiate this EML node with respect to variable `var`.

        Returns a new TensionPoint tree representing d(tension)/d(var).

        Rule: d/dv [exp(f) - ln(g)] = exp(f)·f' - g'/g
        The result is a _DiffNode (internal subclass) which evaluates the
        derivative expression using the same .tension() interface.

        Parameters
        ----------
        var : str
            Name of the variable (must match a _VarNode leaf in the tree).
        """
        return _diff_node(self, var)

    # NOTE: Lorentz/spacetime methods (minkowski_delta, boost, rapidity, …)
    # *and* the EMLPair conversion (.pair()) have moved out of EMLPoint as
    # part of the v1.2.0 split. They live in the sister package
    # eml-spectral as functional helpers:
    #   from eml_spectral.spacetime import minkowski_delta, boost, ...
    #   from eml_spectral.spacetime import pair      # returns EMLPair
    #
    #     from eml_spectral.spacetime import minkowski_delta, boost
    #     boost(point, rapidity=0.5)
    #
    # Same math, same numerics — just no longer mounted on the core type.

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = f", D={self._D}" if self._D is not None else ""
        return f"EMLPoint({self._x!r}, {self._y!r}{mode})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLPoint):
            return NotImplemented
        # Tolerant numerical equality on the evaluated tension. Cheap and
        # sufficient for the vast majority of equality checks; if you need
        # *exact* coordinate equality use `(p.x, p.y) == (q.x, q.y)`.
        try:
            t1 = self.tension()
            t2 = other.tension()
        except Exception:
            return False
        if abs(t2) < 1e-300:
            return abs(t1) < 1e-9
        return abs(t1 - t2) < 1e-9 * max(abs(t1), abs(t2), 1.0)

    def __hash__(self) -> int:
        return hash((round(self.tension(), 9), self._D))


# ── Variable leaf (for symbolic / deferred evaluation) ────────────────────────

class _VarNode(EMLPoint):
    """
    A named variable leaf. .tension() raises unless a binding is set.

    Created by to_mpm() when parsing symbolic expressions like "sin(x)".
    Bindings are injected before evaluation via .bind(**kwargs).
    """

    __slots__ = ("_name", "_binding")

    def __init__(self, name: str) -> None:
        # Dummy coords — never used; tension() is overridden
        super().__init__(0.0, 1.0, D=None)
        self._name: str = name
        self._binding: Optional[float] = None

    @property
    def name(self) -> str:
        return self._name

    def bind(self, value: float) -> "_VarNode":
        """Return a new _VarNode with the given numeric binding."""
        n = _VarNode(self._name)
        n._binding = value
        return n

    def tension(self) -> float:
        if self._binding is None:
            raise ValueError(
                f"Variable '{self._name}' is unbound. "
                "Call .bind(value) or pass via to_mpm(..., **{self._name}=value)."
            )
        return self._binding

    def is_leaf(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"_VarNode({self._name!r})"


# ── Derivative node (internal — not part of public API) ───────────────────────

class _DiffNode(EMLPoint):
    """
    Evaluates d/dv [eml(f, g)] = exp(f)·f' - g'/g.

    Not a pure EML node (requires mul/div), but evaluates via .tension()
    for compatibility with the rest of the tree. The .to_pure_eml() method
    on this class (future) will rewrite it using Table-1 EML identities.
    """

    __slots__ = ("_f", "_g", "_f_prime", "_g_prime")

    def __init__(
        self,
        f: EMLPoint,
        g: EMLPoint,
        f_prime: EMLPoint,
        g_prime: EMLPoint,
    ) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._f = f
        self._g = g
        self._f_prime = f_prime
        self._g_prime = g_prime

    def tension(self) -> float:
        # d/dv [exp(f) - ln(g)] = exp(f) * f' - g'/g
        exp_f = math.exp(self._f.tension())
        f_prime_val = self._f_prime.tension()
        g_val = self._g.tension()
        g_prime_val = self._g_prime.tension()
        if abs(g_val) < 1e-300:
            g_val = 1e-300
        return exp_f * f_prime_val - g_prime_val / g_val

    def is_leaf(self) -> bool:
        return False


class _LitNode(EMLPoint):
    """Numeric literal leaf — always returns the same float from .tension()."""

    __slots__ = ("_value",)

    def __init__(self, value: float) -> None:
        super().__init__(0.0, 1.0, D=None)
        self._value = float(value)

    def tension(self) -> float:
        return self._value

    def is_leaf(self) -> bool:
        return True

    def __repr__(self) -> str:
        return f"_LitNode({self._value})"


def _diff_node(node: EMLPoint, var: str) -> TensionPoint:
    """
    Recursively differentiate a TensionPoint tree w.r.t. `var`.

    Returns a TensionPoint (possibly a _DiffNode) whose .tension()
    evaluates the derivative.
    """
    ZERO = _LitNode(0.0)
    ONE = _LitNode(1.0)

    if isinstance(node, _LitNode):
        return ZERO

    if isinstance(node, _VarNode):
        return ONE if node.name == var else ZERO

    # Standard TensionPoint EML node: d/dv [exp(f) - ln(g)] = exp(f)*f' - g'/g
    f = node._x if isinstance(node._x, EMLPoint) else _LitNode(float(node._x))
    g = node._y if isinstance(node._y, EMLPoint) else _LitNode(float(node._y))
    f_prime = _diff_node(f, var)
    g_prime = _diff_node(g, var)
    return _DiffNode(f, g, f_prime, g_prime)
