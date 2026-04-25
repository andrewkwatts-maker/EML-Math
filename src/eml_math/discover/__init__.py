"""
EML Formula Discovery — find closed-form expressions from numerical data.

Uses the Rust eml_core extension when available (fast, parallel BFS).
Falls back to a pure-Python reference implementation otherwise.

Quick start
-----------
**From data points:**

>>> from eml_math.discover import Searcher
>>> import math
>>> x = [i * 0.1 for i in range(1, 31)]
>>> y = [math.exp(xi) - math.log(xi) for xi in x]
>>> result = Searcher().find(x, y)
>>> print(result)   # SearchResult(formula='eml(x, x)', error=..., complexity=3)

**Compress a known function to EML form:**

>>> from eml_math.discover import compress
>>> result = compress(lambda x: x * x, x_lo=0.5, x_hi=3.0)
>>> print(result.formula)   # "(x * x)" or equivalent

**Identify a constant:**

>>> from eml_math.discover import recognize
>>> print(recognize(3.14159265))   # SearchResult(formula='π', ...)
"""
from __future__ import annotations

import math
from typing import Callable, Optional

from eml_math.discover.search import Searcher
from eml_math.discover.result import SearchResult


def compress(
    fn: Callable[[float], float],
    x_lo: float = 0.2,
    x_hi: float = 3.0,
    n_points: int = 40,
    max_complexity: int = 8,
    precision_goal: float = 1e-8,
    use_trig: bool = True,
    use_eml: bool = True,
) -> Optional[SearchResult]:
    """
    Compress a mathematical function to its minimal EML closed form.

    Samples ``fn`` over ``[x_lo, x_hi]``, runs the beam-search formula
    discovery engine, and returns the most compact EML expression that
    reproduces the function within ``precision_goal`` RMSE.

    This is a one-call equation compression API: give it any Python
    callable and get back a symbolic formula string, its error, and a
    LaTeX/Python rendering.

    Parameters
    ----------
    fn : Callable[[float], float]
        The function to compress. Must accept a single float.
    x_lo, x_hi : float
        Sample range. Avoid 0 and negative values unless your function
        handles them (the EML primitive requires y > 0 internally).
    n_points : int
        Number of sample points in [x_lo, x_hi].
    max_complexity : int
        Maximum expression tree size (node count). Higher = slower but
        finds more complex compressions.
    precision_goal : float
        RMSE threshold; search stops when this is reached.
    use_trig : bool
        Allow sin/cos as primitive operators.
    use_eml : bool
        Allow the EML primitive eml(a,b) = exp(a) − ln(b).

    Returns
    -------
    SearchResult or None
        Compressed formula; None if no finite-error form was found.

    Examples
    --------
    >>> import math
    >>> from eml_math.discover import compress
    >>>
    >>> # exp(x) compresses to itself
    >>> r = compress(math.exp)
    >>> print(r.formula, r.error)
    >>>
    >>> # sin²(x) + cos²(x) = 1  (Pythagorean identity compression)
    >>> r = compress(lambda x: math.sin(x)**2 + math.cos(x)**2)
    >>> assert r.error < 1e-8
    >>>
    >>> # eml(x, x) = exp(x) − ln(x)
    >>> r = compress(lambda x: math.exp(x) - math.log(x))
    >>> print(r.formula)   # 'eml(x, x)'
    """
    step = (x_hi - x_lo) / max(n_points - 1, 1)
    xs = [x_lo + i * step for i in range(n_points)]
    ys: list[float] = []
    for xi in xs:
        try:
            v = fn(xi)
            ys.append(v if math.isfinite(v) else None)  # type: ignore[arg-type]
        except Exception:
            ys.append(None)  # type: ignore[arg-type]

    # Drop any x where fn raised or returned non-finite
    pairs = [(x, y) for x, y in zip(xs, ys) if y is not None]
    if len(pairs) < 4:
        return None
    xs_clean, ys_clean = zip(*pairs)

    return Searcher(
        max_complexity=max_complexity,
        precision_goal=precision_goal,
        use_trig=use_trig,
        use_eml=use_eml,
    ).find(list(xs_clean), list(ys_clean))


def recognize(value: float) -> Optional[SearchResult]:
    """
    Identify a numeric constant as a known mathematical symbol.

    Checks π, e, √2, ln 2, φ, γ, and small EML integer compositions.

    Parameters
    ----------
    value : float
        The constant to identify.

    Returns
    -------
    SearchResult or None

    Examples
    --------
    >>> from eml_math.discover import recognize
    >>> recognize(3.141592653589793).formula
    'π'
    >>> recognize(2.718281828459045).formula
    'e'
    """
    return Searcher().recognize(value)


__all__ = ["Searcher", "SearchResult", "compress", "recognize"]
