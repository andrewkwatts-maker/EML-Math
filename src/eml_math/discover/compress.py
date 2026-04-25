"""
EML equation compression and decompression pipeline.

Converts traditional math expressions (Python string or LaTeX) to their
minimal EML closed form, and renders results back to any notation.

Pipeline
--------
    expr_str  ──► compress_str()  ──► SearchResult ──► decompress()  ──► latex / math / python / mathml
    latex_str ──► compress_latex() ─►       ↑
    value     ──► get(symbol)      ─►       ↑

Known-simplification examples
------------------------------
>>> compress_str("sin(x)**2 + cos(x)**2")          # → "1"
>>> compress_str("exp(log(x))", x_lo=0.5)           # → "x"
>>> compress_latex(r"\\sin^2(x) + \\cos^2(x)")     # → "1"
>>> get('e')                                         # → SearchResult(formula='eml(1, 1)')
>>> get('pi')                                        # → SearchResult(formula='π')
"""
from __future__ import annotations

import math
import re
from typing import Callable, Optional

from eml_math.discover.result import SearchResult
from eml_math.discover.search import Searcher


# ── Symbol table ──────────────────────────────────────────────────────────────

_SYMBOL_TABLE: dict[str, tuple[float, str]] = {
    # symbol_name: (numeric_value, canonical_eml_formula)
    "e":              (math.e,                      "eml(1, 1)"),
    "pi":             (math.pi,                     "π"),
    "π":              (math.pi,                     "π"),
    "1":              (1.0,                          "eml(0, 1)"),
    "0":              (0.0,                          "eml(0, e)"),
    "-1":             (-1.0,                         "(-eml(0, 1))"),
    "2":              (2.0,                          "(eml(0, 1) + eml(0, 1))"),
    "sqrt2":          (math.sqrt(2),                 "sqrt(2)"),
    "√2":             (math.sqrt(2),                 "sqrt(2)"),
    "ln2":            (math.log(2),                  "ln(2)"),
    "log2":           (math.log(2),                  "ln(2)"),
    "phi":            ((1 + math.sqrt(5)) / 2,       "φ (golden ratio)"),
    "φ":              ((1 + math.sqrt(5)) / 2,       "φ (golden ratio)"),
    "golden_ratio":   ((1 + math.sqrt(5)) / 2,       "φ (golden ratio)"),
    "gamma":          (0.5772156649015328,            "γ (Euler-Mascheroni)"),
    "γ":              (0.5772156649015328,            "γ (Euler-Mascheroni)"),
    "euler_mascheroni": (0.5772156649015328,          "γ (Euler-Mascheroni)"),
    "inf":            (math.inf,                     "∞"),
    "infinity":       (math.inf,                     "∞"),
    "tau":            (2 * math.pi,                  "(2·π)"),
    "τ":              (2 * math.pi,                  "(2·π)"),
    "half":           (0.5,                          "(1/2)"),
    "e2":             (math.e ** 2,                  "eml(2, 1)"),
    "1_over_e":       (1.0 / math.e,                 "eml(-1, 1)"),
}


def get(symbol: str) -> Optional[SearchResult]:
    """
    Return the EML derivation of a named mathematical symbol or constant.

    Maps the symbol to its exact EML expression (where one exists) or the
    closest numerical EML approximation.

    Supported symbols
    -----------------
    ``e``, ``pi`` / ``π``, ``1``, ``0``, ``-1``, ``2``,
    ``sqrt2`` / ``√2``, ``ln2``, ``phi`` / ``φ`` / ``golden_ratio``,
    ``gamma`` / ``γ`` / ``euler_mascheroni``,
    ``tau`` / ``τ``, ``half``, ``inf``, ``e2``, ``1_over_e``

    Parameters
    ----------
    symbol : str
        Case-insensitive symbol name. Strips whitespace and underscores.

    Returns
    -------
    SearchResult or None
        The EML formula, its numeric error, and complexity.
        Returns None if the symbol is not recognised.

    Examples
    --------
    >>> from eml_math.discover import get
    >>> get('e').formula
    'eml(1, 1)'
    >>> get('pi').formula
    'π'
    >>> get('sqrt2').formula
    'sqrt(2)'
    >>> get('1').formula
    'eml(0, 1)'
    """
    key = symbol.strip().lower().replace(" ", "_")
    entry = _SYMBOL_TABLE.get(key) or _SYMBOL_TABLE.get(symbol.strip())
    if entry is None:
        return None
    value, formula = entry
    error = abs(value - _eval_formula(formula, value))
    return SearchResult(formula=formula, error=error, complexity=_formula_complexity(formula), params=[])


def _eval_formula(formula: str, fallback: float) -> float:
    """Numerically evaluate a formula string to verify its value."""
    try:
        s = (formula
             .replace("eml(1, 1)", str(math.e))
             .replace("eml(0, 1)", "1.0")
             .replace("eml(0, e)", "0.0")
             .replace("eml(2, 1)", str(math.e**2))
             .replace("eml(-1, 1)", str(1/math.e))
             .replace("sqrt(2)", str(math.sqrt(2)))
             .replace("ln(2)", str(math.log(2)))
             .replace("π", str(math.pi))
             .replace("φ (golden ratio)", str((1+math.sqrt(5))/2))
             .replace("γ (Euler-Mascheroni)", "0.5772156649015328")
             .replace("∞", str(math.inf))
             .replace("(2·π)", str(2*math.pi))
             .replace("(1/2)", "0.5"))
        v = float(eval(s, {"__builtins__": {}}, {}))  # noqa: S307
        return v if math.isfinite(v) else fallback
    except Exception:
        return fallback


def _formula_complexity(formula: str) -> int:
    """Rough node-count estimate for a formula string."""
    ops = sum(formula.count(op) for op in ["eml(", "exp(", "ln(", "sqrt(", "sin(", "cos(", "+", "-", "*", "/"])
    return max(1, ops)


# ── LaTeX → Python conversion ─────────────────────────────────────────────────

_LATEX_MAP = [
    # LaTeX pattern → Python replacement (applied in order)
    (r"\\sin\s*\^2\s*\(([^)]+)\)",   r"(sin(\1)**2)"),
    (r"\\cos\s*\^2\s*\(([^)]+)\)",   r"(cos(\1)**2)"),
    (r"\\sin\s*\^2\s*([a-zA-Z])",    r"(sin(\1)**2)"),
    (r"\\cos\s*\^2\s*([a-zA-Z])",    r"(cos(\1)**2)"),
    (r"\\sin",                        "sin"),
    (r"\\cos",                        "cos"),
    (r"\\tan",                        "tan"),
    (r"\\exp",                        "exp"),
    (r"\\ln",                         "log"),
    (r"\\log",                        "log"),
    (r"\\sqrt\{([^}]+)\}",           r"sqrt(\1)"),
    (r"\\sqrt\s+(\w+)",              r"sqrt(\1)"),
    (r"\\frac\{([^}]+)\}\{([^}]+)\}", r"((\1)/(\2))"),
    (r"\\pi",                         "pi"),
    (r"\\infty",                      "inf"),
    (r"\\cdot",                       "*"),
    (r"\\times",                      "*"),
    # Power patterns MUST run before generic { } → ( ) replacement
    (r"([a-zA-Z0-9)])\s*\^\{([^}]+)\}", r"(\1**(\2))"),
    (r"([a-zA-Z0-9)])\s*\^2",        r"(\1**2)"),
    (r"([a-zA-Z0-9)])\s*\^3",        r"(\1**3)"),
    (r"([a-zA-Z0-9)])\s*\^([0-9]+)", r"(\1**\2)"),
    (r"\{",                           "("),
    (r"\}",                           ")"),
    (r"\\left\(",                     "("),
    (r"\\right\)",                    ")"),
    (r"\\left\[",                     "("),
    (r"\\right\]",                    ")"),
]

def _latex_to_python(latex: str) -> str:
    """Convert common LaTeX math notation to a Python math expression string."""
    s = latex.strip()
    # Strip display/inline math delimiters
    for delim in (r"\[", r"\]", r"\(", r"\)", "$$", "$"):
        s = s.replace(delim, "")
    for pattern, repl in _LATEX_MAP:
        s = re.sub(pattern, repl, s)
    return s.strip()


def _python_to_latex(expr: str) -> str:
    """Convert a Python math expression string to LaTeX."""
    s = expr
    replacements = [
        ("math.exp(", r"\exp("),
        ("math.log(", r"\ln("),
        ("math.sqrt(", r"\sqrt{"),  # crude: won't add closing } correctly for nested
        ("math.sin(", r"\sin("),
        ("math.cos(", r"\cos("),
        ("math.tan(", r"\tan("),
        ("math.pi", r"\pi"),
        ("math.inf", r"\infty"),
        ("exp(", r"\exp("),
        ("log(", r"\ln("),
        ("sqrt(", r"\sqrt{"),
        ("sin(", r"\sin("),
        ("cos(", r"\cos("),
        ("tan(", r"\tan("),
        ("**2", "^2"),
        ("**3", "^3"),
        ("eml(", r"\mathrm{eml}("),
        ("pi", r"\pi"),
        ("inf", r"\infty"),
    ]
    for src, dst in replacements:
        s = s.replace(src, dst)
    return s


# ── Safe expression evaluator ─────────────────────────────────────────────────

_MATH_NS = {
    name: getattr(math, name)
    for name in dir(math)
    if not name.startswith("_")
}
_MATH_NS.update({"abs": abs, "round": round})


def _make_callable(expr: str) -> Optional[Callable[[float], float]]:
    """
    Build a single-variable callable from a Python math expression string.

    The variable is ``x``. Uses a restricted eval namespace (math module only).
    Returns None if the expression fails to parse or evaluate.
    """
    # Quick syntax check
    try:
        code = compile(expr, "<expr>", "eval")
    except SyntaxError:
        return None

    def fn(x: float) -> float:
        ns = dict(_MATH_NS)
        ns["x"] = x
        return eval(code, {"__builtins__": {}}, ns)  # noqa: S307

    return fn


# ── compress_str / compress_latex ─────────────────────────────────────────────

def compress_str(
    expr: str,
    x_lo: float = 0.2,
    x_hi: float = 3.0,
    n_points: int = 40,
    max_complexity: int = 8,
    precision_goal: float = 1e-8,
    use_trig: bool = True,
    use_eml: bool = True,
) -> Optional[SearchResult]:
    """
    Compress a Python math expression string to its minimal EML form.

    The expression is evaluated over ``[x_lo, x_hi]`` and the beam-search
    engine finds the shortest EML formula that reproduces it.

    Parameters
    ----------
    expr : str
        A Python math expression using standard names from the ``math``
        module plus the variable ``x``.
        Examples: ``"sin(x)**2 + cos(x)**2"``, ``"exp(log(x))"``,
        ``"x * x"``, ``"exp(x) - log(x)"``
    x_lo, x_hi : float
        Sampling range. Avoid 0 if the expression includes ``log(x)``.
    n_points : int
        Number of sample points.
    max_complexity : int
        Maximum EML tree depth (higher = finds more complex compressions).
    precision_goal : float
        RMSE threshold for early termination.
    use_trig : bool
        Allow sin/cos in the output formula.
    use_eml : bool
        Allow the eml(a,b) primitive in the output formula.

    Returns
    -------
    SearchResult or None
        Compressed formula, RMSE error, complexity, and rendering methods.

    Examples
    --------
    >>> compress_str("sin(x)**2 + cos(x)**2")
    SearchResult(formula='1', error=..., complexity=1)
    >>> compress_str("exp(log(x))", x_lo=0.5)
    SearchResult(formula='x', error=..., complexity=1)
    >>> compress_str("exp(x) - log(x)", x_lo=0.5)
    SearchResult(formula='eml(x, x)', error=..., complexity=3)
    """
    fn = _make_callable(expr)
    if fn is None:
        return None
    from eml_math.discover import compress
    return compress(fn, x_lo=x_lo, x_hi=x_hi, n_points=n_points,
                    max_complexity=max_complexity, precision_goal=precision_goal,
                    use_trig=use_trig, use_eml=use_eml)


def compress_latex(
    latex: str,
    x_lo: float = 0.2,
    x_hi: float = 3.0,
    n_points: int = 40,
    max_complexity: int = 8,
    precision_goal: float = 1e-8,
    use_trig: bool = True,
    use_eml: bool = True,
) -> Optional[SearchResult]:
    """
    Compress a LaTeX math expression to its minimal EML form.

    Converts the LaTeX string to a Python expression via a regex translator,
    then runs the beam-search compressor. Supports common LaTeX constructs:
    ``\\sin``, ``\\cos``, ``\\exp``, ``\\ln``, ``\\sqrt{}``, ``\\frac{}{}``,
    ``^2``, ``^n``, ``\\pi``, ``\\cdot``.

    Parameters
    ----------
    latex : str
        A LaTeX math expression. May include ``$...$`` or ``\\(..\\)``
        delimiters (stripped automatically).
        Examples: ``r"\\sin^2(x) + \\cos^2(x)"``,
        ``r"e^{\\ln(x)}"``, ``r"\\frac{1}{x}"``

    Returns
    -------
    SearchResult or None

    Examples
    --------
    >>> compress_latex(r"\\sin^2(x) + \\cos^2(x)")
    SearchResult(formula='1', error=..., complexity=1)
    >>> compress_latex(r"e^{\\ln(x)}", x_lo=0.5)
    SearchResult(formula='x', error=..., complexity=1)
    """
    python_expr = _latex_to_python(latex)
    return compress_str(python_expr, x_lo=x_lo, x_hi=x_hi, n_points=n_points,
                        max_complexity=max_complexity, precision_goal=precision_goal,
                        use_trig=use_trig, use_eml=use_eml)


# ── decompress ────────────────────────────────────────────────────────────────

def decompress(
    result: SearchResult,
    fmt: str = "math",
) -> str:
    """
    Render a SearchResult back to a human-readable mathematical notation.

    Parameters
    ----------
    result : SearchResult
        Output of ``compress()``, ``compress_str()``, ``compress_latex()``,
        ``get()``, or any ``Searcher.find()`` call.
    fmt : str
        Output format — one of:

        ``"math"``
            Clean standard notation: ``exp(x) - ln(x)``.
        ``"latex"``
            LaTeX with proper command names: ``\\exp(x) - \\ln(x)``.
            Ready for ``$...$`` / ``\\(...\\)`` / MathJax rendering.
        ``"mathml"``
            MathML markup string (inline ``<math>`` element).
        ``"python"``
            Runnable Python: ``import math; f = lambda x: math.exp(x) - ...``
        ``"eml"``
            Raw EML formula string (same as ``result.formula``).

    Returns
    -------
    str

    Examples
    --------
    >>> r = compress_str("sin(x)**2 + cos(x)**2")
    >>> decompress(r, fmt="latex")
    '1'
    >>> decompress(r, fmt="mathml")
    '<math><mn>1</mn></math>'
    """
    if fmt == "eml":
        return result.formula
    if fmt == "python":
        return result.to_python()
    if fmt == "latex":
        return result.to_latex()
    if fmt == "mathml":
        return _formula_to_mathml(result.formula)
    # Default: clean "math" notation
    return _formula_to_math(result.formula)


def _formula_to_math(formula: str) -> str:
    """Convert internal formula string to clean standard math notation."""
    s = formula
    s = s.replace("eml(", "eml(")   # keep as-is; eml is now standard
    s = s.replace("ln(", "ln(")
    s = s.replace("sqrt(", "√(")
    s = s.replace("pi", "π")
    s = s.replace("inf", "∞")
    return s


def _formula_to_mathml(formula: str) -> str:
    """
    Convert a formula string to a minimal MathML representation.

    Uses a token-by-token approach to avoid cascading string-replace bugs
    (e.g. the '/' inside '</mo>' being re-replaced by an operator handler).
    """
    s = formula.strip()

    # Named-constant shortcuts
    _CONSTANTS = {
        "0": "<mn>0</mn>", "1": "<mn>1</mn>", "2": "<mn>2</mn>",
        "3": "<mn>3</mn>", "4": "<mn>4</mn>", "5": "<mn>5</mn>",
        "π": "<mi>&pi;</mi>", "pi": "<mi>&pi;</mi>",
        "e": "<mi>e</mi>",
        "x": "<mi>x</mi>",
        "∞": "<mi>&infin;</mi>",
    }
    if s in _CONSTANTS:
        return f"<math>{_CONSTANTS[s]}</math>"

    # Tokenise: split on recognised function/operator/variable patterns.
    # Order matters — longer tokens first.
    _FUNC_MAP = {
        "eml": "eml", "exp": "exp", "ln": "ln", "log": "ln",
        "sqrt": "sqrt", "sin": "sin", "cos": "cos", "tan": "tan",
    }
    _OP_MAP = {"+": "+", "-": "-", "*": "&sdot;", "/": "/"}

    parts: list[str] = []
    i = 0
    while i < len(s):
        c = s[i]

        # Skip spaces
        if c == " ":
            i += 1
            continue

        # Digit / number
        if c.isdigit() or c == ".":
            j = i
            while j < len(s) and (s[j].isdigit() or s[j] == "."):
                j += 1
            parts.append(f"<mn>{s[i:j]}</mn>")
            i = j
            continue

        # Named function or known constant
        matched_func = False
        for fname in sorted(_FUNC_MAP, key=len, reverse=True):
            if s[i:i+len(fname)] == fname:
                mml_name = _FUNC_MAP[fname]
                parts.append(f"<mi>{mml_name}</mi>")
                i += len(fname)
                matched_func = True
                break
        if matched_func:
            continue

        # Named constant (pi, inf)
        if s[i:i+2] == "pi":
            parts.append("<mi>&pi;</mi>")
            i += 2
            continue
        if s[i:i+3] == "inf":
            parts.append("<mi>&infin;</mi>")
            i += 3
            continue
        if c == "π":
            parts.append("<mi>&pi;</mi>")
            i += 1
            continue
        if c == "∞":
            parts.append("<mi>&infin;</mi>")
            i += 1
            continue

        # Operator
        if c in _OP_MAP:
            parts.append(f"<mo>{_OP_MAP[c]}</mo>")
            i += 1
            continue

        # Parentheses / comma
        if c == "(":
            parts.append("<mo>(</mo>")
            i += 1
            continue
        if c == ")":
            parts.append("<mo>)</mo>")
            i += 1
            continue
        if c == ",":
            parts.append("<mo>,</mo>")
            i += 1
            continue

        # Variable / identifier character
        if c.isalpha() or c == "_":
            j = i
            while j < len(s) and (s[j].isalnum() or s[j] == "_"):
                j += 1
            parts.append(f"<mi>{s[i:j]}</mi>")
            i = j
            continue

        # Anything else: pass through as-is inside an mi
        parts.append(f"<mi>{c}</mi>")
        i += 1

    return "<math>" + "".join(parts) + "</math>"
