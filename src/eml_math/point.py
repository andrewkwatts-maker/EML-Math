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
    x : float | TensionPoint
        Direct growth coordinate. May be another TensionPoint (nested EML).
    y : float | TensionPoint
        Mirror coordinate. May be another TensionPoint (nested EML).
        Must evaluate to > 0; if ≤ 0, the frame-shift guard applies |y|.
    D : float | None
        Quantization scale. None = continuous mode (default).
        Set to a positive float to enable discrete (quantized) mode.

    Examples
    --------
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
        x: _Coord,
        y: _Coord,
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

    # ── iteration (mirror pulse) ──────────────────────────────────────────────

    def mirror_pulse(self) -> "EMLPoint":
        """
        One EML iteration step, returning the next EMLPoint.

        Continuous mode (D=None) — two clean conditions:
          1. y ≤ 0  → frame-shift: use |y| (domain violation guard).
          2. x > OVERFLOW_THRESHOLD → self-mirror dampening: x = ln(x).

        Discrete mode (D set) — additionally applies:
          3. round(T * D) quantization on y_new.
          4. Ddx/dsy snap conditions (Axiom 8, active only when D is set).

        Returns a new TensionPoint; this one is immutable.
        """
        xv = self.x
        yv = self.y

        # Condition 2: Slipping Wheel — exp overflow dampening
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)  # self-mirror: bring x back into range

        # Condition 1: frame-shift guard
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300

        T = math.exp(xv) - math.log(y_safe)
        x_new = y_safe
        y_new = T

        if self._D is not None:
            # Discrete mode: quantize and detect snap conditions
            b_new = round(y_new * self._D)
            y_new = b_new / self._D
            nxt = EMLPoint(x_new, y_new, D=self._D)
            nxt._prev_x = xv
            nxt._prev_y = y_safe
        else:
            nxt = EMLPoint(x_new, y_new, D=None)

        return nxt

    # EML/Sheffer nomenclature aliases
    def eml(self) -> float:
        """Alias for tension() — eml(x, y) = exp(x) - ln(y) (Sheffer operator)."""
        return self.tension()

    def pulse(self) -> "EMLPoint":
        """Alias for mirror_pulse() — one Sheffer iteration step."""
        return self.mirror_pulse()

    def domain_shift(self) -> "EMLPoint":
        """Alias for frame_shift() — domain preservation step (Axiom 8)."""
        return self.frame_shift()

    def frame_shift(self) -> "EMLPoint":
        """
        Explicit Axiom 8 frame-shift: x_new = |y|, y_new = exp(x) - ln(|y|).

        In continuous mode mirror_pulse() applies this automatically when y ≤ 0.
        Call this directly to force a frame shift regardless of y's sign.
        """
        xv = self.x
        yv = abs(self.y)
        if yv == 0:
            yv = 1e-300
        T = math.exp(xv) - math.log(yv)
        return EMLPoint(yv, T, D=self._D)

    # ── state inspection ──────────────────────────────────────────────────────

    def is_slipping(self) -> bool:
        """True if x is near the Slipping Wheel threshold (exp overflow)."""
        return self.x > OVERFLOW_THRESHOLD

    def is_locked(self) -> bool:
        """True if tension ≈ 0 (Locked Wheel — minimum energy standstill)."""
        return abs(self.tension()) < 1e-12

    def conserves_tension(self, nxt: "EMLPoint", tol: float = 1e-9) -> bool:
        """
        Verify Axiom 10 (Conservation of Tension): T + ln(y) = exp(x).

        This is the structural balance: growth (exp(x)) equals the sum of
        the mirror component (ln(y)) and the residual tension (T).
        Rearranges trivially from T = exp(x) - ln(y), so it serves as a
        floating-point sanity check and is always true for valid (x, y).

        The `nxt` parameter is accepted for API symmetry with simulation
        verification loops but is not needed for the identity itself.
        """
        xv = self.x
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        try:
            T = math.exp(xv) - math.log(y_safe)
            return abs(T + math.log(y_safe) - math.exp(xv)) < tol
        except OverflowError:
            # exp(x) overflows — conservation holds by definition (T = exp(x) - ln(y))
            return True

    def resonates_with(self, other: "EMLPoint", tol: float = 1e-9) -> bool:
        """
        Axiom 14: Two points resonate if their tension ratios are equal.

        Resonance is the EML analog of equality — dynamic phase-matching.
        """
        t1 = self.tension()
        t2 = other.tension()
        if abs(t2) < 1e-300:
            return abs(t1) < tol
        return abs(t1 / t2 - 1.0) < tol

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

    # ── geometric / relativistic extensions ──────────────────────────────────

    def pair(self) -> "EMLPair":
        """Returns (exp(x), ln(|y|)) as an EMLPair — the canonical frame coordinates."""
        from eml_math.pair import EMLPair
        xv = self.x
        yv = self.y
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        return EMLPair.from_values(math.exp(xv), math.log(y_safe))

    def euclidean_delta(self) -> float:
        """sqrt(exp(2x) + (ln y)^2) — Euclidean distance invariant under 4-frame rotations."""
        if _RUST_POINT and self.is_leaf():
            return _core.EMLPoint(self.x, self.y).euclidean_delta()
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        ex = math.exp(xv)
        ly = math.log(y_safe)
        return math.sqrt(ex * ex + ly * ly)

    def minkowski_delta(self, signature: str = "+---", c: float = 1.0) -> float:
        """
        Minkowski invariant interval sqrt(|exp(2x) - (c*ln y)^2|).
        signature '+---': time-like when exp(2x) > (c*ln y)^2.
        signature '-+++': space-like convention.
        """
        if _RUST_POINT and self.is_leaf():
            plus_sig = signature.startswith("+")
            return _core.EMLPoint(self.x, self.y).minkowski_delta(plus_sig, c)
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        t_comp = math.exp(xv)
        x_comp = c * math.log(y_safe)
        if signature.startswith("+"):
            ds2 = t_comp * t_comp - x_comp * x_comp
        else:
            ds2 = x_comp * x_comp - t_comp * t_comp
        return math.sqrt(abs(ds2))

    def is_timelike(self, c: float = 1.0) -> bool:
        """True when exp(2x) > (c * ln y)^2 in (+---) signature."""
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        return math.exp(xv) ** 2 > (c * math.log(y_safe)) ** 2

    def is_spacelike(self, c: float = 1.0) -> bool:
        """True when exp(2x) < (c * ln y)^2 in (+---) signature."""
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        return math.exp(xv) ** 2 < (c * math.log(y_safe)) ** 2

    def is_lightlike(self, c: float = 1.0, tol: float = 1e-9) -> bool:
        """True when |exp(2x) - (c * ln y)^2| < tol."""
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        return abs(math.exp(xv) ** 2 - (c * math.log(y_safe)) ** 2) < tol

    def canonical_frame(self, k: int = 0) -> "EMLPair":
        """
        Rotate pair through frame k in {0,1,2,3} by multiplying by {1, i, -1, -i}.
        Frame 0: (exp(x),  ln(y))   identity
        Frame 1: (-ln(y),  exp(x))  x i
        Frame 2: (-exp(x), -ln(y))  x -1
        Frame 3: (ln(y),  -exp(x))  x -i
        The Euclidean delta is invariant across all frames.
        """
        from eml_math.pair import EMLPair
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        r = math.exp(xv)
        im = math.log(y_safe)
        k = k % 4
        if k == 0:
            return EMLPair.from_values(r, im)
        elif k == 1:
            return EMLPair.from_values(-im, r)
        elif k == 2:
            return EMLPair.from_values(-r, -im)
        else:
            return EMLPair.from_values(im, -r)

    def rapidity(self) -> float:
        """
        Extract rapidity phi = atanh(ln(y) / exp(x)) from pair coordinates.
        Raises ValueError for spacelike points where |ln(y)| >= |exp(x)|.
        """
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        t_comp = math.exp(xv)
        x_comp = math.log(y_safe)
        if abs(t_comp) < 1e-300:
            raise ValueError("Cannot compute rapidity: time component is zero")
        ratio = x_comp / t_comp
        if abs(ratio) >= 1.0:
            raise ValueError(
                f"Cannot compute rapidity: |space/time| = {abs(ratio):.6g} >= 1 (point is not timelike)"
            )
        return math.atanh(ratio)

    def boost(self, phi: float, c: float = 1.0, pure_eml: bool = False) -> "EMLPoint":
        """
        Apply a Lorentz boost by rapidity phi, returning a new EMLPoint with identical Δ_M.
        Boost matrix: t' = t*cosh(phi) - (x/c)*sinh(phi)
                      x' = x*cosh(phi) - t*c*sinh(phi)
        where t = exp(x_coord), x = ln(y_coord).
        pure_eml=True is reserved for future symbolic rewriting (currently ignored).
        """
        if _RUST_POINT and self.is_leaf():
            rust_result = _core.EMLPoint(self.x, self.y).boost(phi, c)
            return EMLPoint(rust_result.x, rust_result.y, D=self._D)
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        t_comp = math.exp(xv)
        x_comp = math.log(y_safe)
        ch = math.cosh(phi)
        sh = math.sinh(phi)
        t_new = t_comp * ch - (x_comp / c) * sh
        x_new = x_comp * ch - t_comp * c * sh
        if t_new <= 0:
            t_new = 1e-300
        x_out = math.log(t_new)
        # Guard exp overflow on x_new
        if x_new > 709.0:
            x_new = 709.0
        elif x_new < -709.0:
            x_new = -709.0
        y_out = math.exp(x_new)
        return EMLPoint(x_out, y_out, D=self._D)

    def boost_velocity(self, v: float, c: float = 1.0) -> "EMLPoint":
        """Boost by velocity v (converts to rapidity phi = atanh(v/c))."""
        if abs(v) >= c:
            raise ValueError(f"Speed |v| = {abs(v):.6g} must be less than c = {c:.6g}")
        phi = math.atanh(v / c)
        return self.boost(phi, c=c)

    def light_cone_coordinates(self, c: float = 1.0) -> "tuple[float, float]":
        """Returns null coordinates (u, v) = (t + x/c, t - x/c)."""
        xv = self.x
        if xv > OVERFLOW_THRESHOLD:
            xv = math.log(xv)
        yv = self.y
        y_safe = abs(yv) if yv <= 0 else yv
        if y_safe == 0:
            y_safe = 1e-300
        t_comp = math.exp(xv)
        x_comp = math.log(y_safe) / c
        return t_comp + x_comp, t_comp - x_comp

    def light_cone_type(self, c: float = 1.0) -> str:
        """Returns 'timelike', 'spacelike', or 'lightlike'."""
        if self.is_lightlike(c=c):
            return "lightlike"
        if self.is_timelike(c=c):
            return "timelike"
        return "spacelike"

    def future_light_cone(self, c: float = 1.0) -> bool:
        """True if this event is in the future light cone (timelike and exp(x) > 0)."""
        return self.is_timelike(c=c)  # exp(x) is always > 0, so timelike implies future

    def rest_energy(self, c: float = 1.0) -> float:
        """Δ_M in natural units — rest energy E_0 = m*c^2 (with c=1 this is rest mass)."""
        return self.minkowski_delta(signature="+---", c=c)

    def proper_time(self, c: float = 1.0) -> float:
        """Proper time tau = Δ_M / c for timelike worldlines."""
        return self.rest_energy(c=c) / c

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        mode = f", D={self._D}" if self._D is not None else ""
        return f"EMLPoint({self._x!r}, {self._y!r}{mode})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLPoint):
            return NotImplemented
        return self.resonates_with(other)


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
