"""
Formula discovery via BFS over EML expression trees.

Delegates to Rust eml_core.find_formula when available.
Falls back to a pure-Python BFS for environments without the compiled extension.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.discover.result import SearchResult

try:
    from eml_math import eml_core as _core
    _RUST = True
except ImportError:
    _RUST = False


class Searcher:
    """
    Searches for a closed-form EML formula fitting numerical data.

    Parameters
    ----------
    max_complexity : int
        Maximum expression tree depth (node count). Default 8.
    beam_width : int
        Number of candidates kept per BFS level. Default 2000.
    precision_goal : float
        Stop early when RMSE ≤ this value. Default 1e-10.
    use_trig : bool
        Include sin/cos as primitive operators. Default True.
    use_eml : bool
        Include the EML primitive eml(a,b)=exp(a)-ln(b). Default True.
    complexity_penalty : float
        Penalize longer expressions: score = error × (1 + complexity × penalty).
    """

    def __init__(
        self,
        max_complexity: int = 8,
        beam_width: int = 2000,
        precision_goal: float = 1e-10,
        use_trig: bool = True,
        use_eml: bool = True,
        complexity_penalty: float = 0.001,
    ) -> None:
        self.max_complexity = max_complexity
        self.beam_width = beam_width
        self.precision_goal = precision_goal
        self.use_trig = use_trig
        self.use_eml = use_eml
        self.complexity_penalty = complexity_penalty

    def find(
        self,
        x_data: "list[float] | list[list[float]]",
        y_data: "list[float]",
    ) -> Optional[SearchResult]:
        """
        Find a formula f such that f(x) ≈ y for all data points.

        Parameters
        ----------
        x_data : list[float] or list[list[float]]
            Input data. For univariate: a flat list [x0, x1, ...].
            For multivariate: a list of column vectors [[x0_0, x0_1, ...], [x1_0, ...]].
        y_data : list[float]
            Target output values.

        Returns
        -------
        SearchResult or None
            Best formula found, or None if no finite-error formula found.
        """
        # Normalise x to list-of-columns
        if isinstance(x_data[0], (int, float)):
            cols = [list(x_data)]
        else:
            cols = [list(col) for col in x_data]

        y = list(y_data)

        if _RUST:
            raw = _core.find_formula(
                cols, y,
                self.max_complexity,
                self.beam_width,
                self.precision_goal,
                self.use_trig,
                self.use_eml,
                self.complexity_penalty,
            )
            if raw is None:
                return None
            return SearchResult(
                formula=raw.formula,
                error=raw.error,
                complexity=raw.complexity,
                params=list(raw.params),
            )
        else:
            return _python_search(cols, y, self)

    def recognize(self, value: float) -> Optional[SearchResult]:
        """
        Identify a constant (e.g. 3.14159 → 'π').

        Checks known constants then searches for EML compositions of depth ≤ 4.
        """
        known = {
            math.pi: "π",
            math.e: "e",
            math.sqrt(2): "sqrt(2)",
            math.log(2): "ln(2)",
            1.6180339887498949: "φ (golden ratio)",
            0.5772156649015328: "γ (Euler-Mascheroni)",
        }
        for const, name in known.items():
            if abs(value - const) / (abs(const) + 1e-300) < 1e-8:
                return SearchResult(formula=name, error=0.0, complexity=1, params=[])

        # Try to express as EML of small integers
        for a in range(1, 6):
            for b in range(1, 6):
                candidate = math.exp(a) - math.log(b)
                if abs(value - candidate) < 1e-8:
                    return SearchResult(
                        formula=f"eml({a}, {b})",
                        error=abs(value - candidate),
                        complexity=3,
                        params=[],
                    )
        return None


# ── Pure-Python fallback BFS ──────────────────────────────────────────────────

def _rmse(predicted: list[float], y: list[float]) -> float:
    if len(predicted) != len(y):
        return math.inf
    mse = sum((p - t) ** 2 for p, t in zip(predicted, y)) / len(y)
    return math.sqrt(mse)


def _eval_expr(expr_fn, x_cols: list[list[float]]) -> Optional[list[float]]:
    try:
        result = [expr_fn(*[col[i] for col in x_cols]) for i in range(len(x_cols[0]))]
        if all(math.isfinite(v) for v in result):
            return result
    except (ValueError, ZeroDivisionError, OverflowError):
        pass
    return None


def _python_search(
    cols: list[list[float]],
    y: list[float],
    config: Searcher,
) -> Optional[SearchResult]:
    """Minimal pure-Python BFS — covers depth-1 and depth-2 expressions only."""
    n_vars = len(cols)
    var_names = [chr(ord('x') + i) for i in range(n_vars)]

    # Seed candidates: constants + variables
    seeds: list[tuple[str, object]] = [
        ("0", lambda *_: 0.0),
        ("1", lambda *_: 1.0),
    ]
    for i, name in enumerate(var_names):
        idx = i
        seeds.append((name, (lambda j: lambda *args: args[j])(idx)))

    best_formula, best_error = None, math.inf

    def check(name: str, fn) -> None:
        nonlocal best_formula, best_error
        pred = _eval_expr(fn, cols)
        if pred is None:
            return
        err = _rmse(pred, y)
        if err < best_error:
            best_error = err
            best_formula = name

    # Depth 1: seeds
    for name, fn in seeds:
        check(name, fn)

    # Depth 2: unary ops on seeds
    unary = [
        ("exp({})", lambda f: lambda *a: math.exp(f(*a))),
        ("ln({})", lambda f: lambda *a: math.log(f(*a)) if f(*a) > 0 else float('nan')),
        ("(-{})", lambda f: lambda *a: -f(*a)),
        ("(1/{})", lambda f: lambda *a: 1.0 / f(*a) if f(*a) != 0 else float('nan')),
        ("sqrt({})", lambda f: lambda *a: math.sqrt(f(*a)) if f(*a) >= 0 else float('nan')),
    ]
    if config.use_trig:
        unary += [
            ("sin({})", lambda f: lambda *a: math.sin(f(*a))),
            ("cos({})", lambda f: lambda *a: math.cos(f(*a))),
        ]

    depth2 = []
    for name, fn in seeds:
        for fmt, wrap in unary:
            new_name = fmt.format(name)
            new_fn = wrap(fn)
            check(new_name, new_fn)
            depth2.append((new_name, new_fn))

    # Depth 3: binary ops on seeds × seeds
    binary = [
        ("({} + {})", lambda f, g: lambda *a: f(*a) + g(*a)),
        ("({} - {})", lambda f, g: lambda *a: f(*a) - g(*a)),
        ("({} * {})", lambda f, g: lambda *a: f(*a) * g(*a)),
        ("({} / {})", lambda f, g: lambda *a: f(*a) / g(*a) if g(*a) != 0 else float('nan')),
    ]
    if config.use_eml:
        def _eml(f, g):
            def _fn(*a):
                xv = f(*a)
                yv = g(*a)
                if xv > 709.78:
                    xv = math.log(xv)
                y_safe = abs(yv) if yv <= 0 else yv
                y_safe = max(y_safe, 1e-300)
                return math.exp(xv) - math.log(y_safe)
            return _fn
        binary.append(("eml({}, {})", _eml))

    for (n1, f1) in seeds:
        for (n2, f2) in seeds:
            for fmt, wrap in binary:
                new_name = fmt.format(n1, n2)
                new_fn = wrap(f1, f2)
                check(new_name, new_fn)

    if best_formula is None:
        return None
    return SearchResult(formula=best_formula, error=best_error, complexity=2, params=[])
