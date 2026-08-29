"""
EML expression evaluator — evaluate eml_description strings against a value context.

An ``eml_description`` string has the form::

    "EML: <python-expr> — <human-readable description>"

The ``<python-expr>`` uses operators from ``eml_math.operators`` plus three
literal constructors:

* ``eml_scalar(x)``   — wrap a numeric literal
* ``eml_pi()``        — π
* ``eml_vec(name)``   — look up *name* in the supplied value context dict

This module provides :class:`EMLEvaluator` which binds a ``{path: value}``
context so ``eml_vec`` references resolve to real numbers, and
:func:`eml_eval` as a convenience one-shot function.

Example::

    from eml_math.evaluator import EMLEvaluator

    ctx = {"gauge.alpha_s": 0.118, "gauge.M_GUT": 2e16}
    ev  = EMLEvaluator(ctx)

    val = ev.eval("EML: ops.mul(eml_vec('gauge.alpha_s'), eml_scalar(2.0)) — 2*alpha_s")
    # val ≈ 0.236
"""
from __future__ import annotations

import ast
import math
import re
from typing import Any, Dict, Optional

import eml_math.operators as ops
from eml_math.point import EMLPoint, _LitNode

__all__ = ["EMLEvaluator", "eml_eval", "ParseError"]

# Separator between expression and human-readable description.
_SEP = " — "
_PREFIX = "EML: "


class _NormalisingNamespace(dict):
    """Evaluation namespace that falls back to a case/underscore-blind match.

    Expression authors and value registries spell the same quantity
    differently -- ``Vcb`` vs ``V_cb``, ``M_Planck`` vs ``M_PLANCK``,
    ``alpha_GUT_inv`` vs ``ALPHA_GUT_INV``, ``M_KK`` vs ``m_KK``. Every one
    of those raised NameError and was recorded as an expression that "did not
    evaluate", when the only difference was capitalisation.

    Exact spellings always win. The normalised fallback resolves a name ONLY
    when every candidate sharing its normalised form carries the same value;
    where two genuinely different quantities collide, the lookup fails as
    before. Guessing there would score an expression against the wrong
    number, which is worse than not evaluating it.

    Implemented via ``__missing__``, which CPython consults for a dict
    subclass used as the *globals* mapping of ``eval``.
    """

    __slots__ = ("_normalised",)

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        buckets: Dict[str, list] = {}
        for key, value in self.items():
            buckets.setdefault(_normalise_name(key), []).append(value)
        # Keep only unambiguous buckets: one distinct value.
        self._normalised = {}
        for norm, values in buckets.items():
            distinct = {_hashable(v) for v in values}
            if len(distinct) == 1:
                self._normalised[norm] = values[0]

    def __missing__(self, key: str):
        try:
            return self._normalised[_normalise_name(key)]
        except (KeyError, TypeError):
            raise KeyError(key) from None


def _longest_valid_expression(s: str) -> str:
    """Longest leading substring of *s* that parses as a Python expression.

    Returns *s* unchanged when it already parses, or when no prefix does --
    in the latter case the caller surfaces the genuine syntax error instead
    of a silently truncated fragment, which would be worse than failing.
    """
    if not s:
        return s
    try:
        ast.parse(s, mode="eval")
        return s
    except SyntaxError:
        pass

    # Candidate cut points: positions at bracket depth 0, longest first.
    cuts: list[int] = []
    depth = 0
    in_str: str | None = None
    for i, ch in enumerate(s):
        if in_str:
            if ch == in_str:
                in_str = None
            continue
        if ch in "'\"":
            in_str = ch
        elif ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
            if depth == 0:
                cuts.append(i + 1)
    for end in reversed(cuts):
        candidate = s[:end].strip()
        try:
            ast.parse(candidate, mode="eval")
            return candidate
        except SyntaxError:
            continue
    return s


def _normalise_name(name: str) -> str:
    """Spelling-insensitive key: drop underscores, fold case."""
    return name.replace("_", "").lower() if isinstance(name, str) else name


def _hashable(value):
    """Round floats so 1.0 and 1.0000000000001 do not read as a collision."""
    if isinstance(value, float):
        return round(value, 12)
    try:
        hash(value)
        return value
    except TypeError:
        return id(value)

# Regex that strips the prefix and optional description tail.
_EXPR_RE = re.compile(r"^EML:\s*(.*?)(?:\s+[—–-]{1,3}\s+.*)?$", re.DOTALL)


class ParseError(ValueError):
    """Raised when an eml_description string cannot be parsed."""


class EMLEvaluator:
    """
    Evaluate EML expression strings with a bound parameter-value context.

    Parameters
    ----------
    context:
        Mapping of ``{parameter_path: numeric_value}``.  Used to resolve
        ``eml_vec(name)`` calls inside expressions.
    strict:
        If *True* (default), unknown ``eml_vec`` names raise :exc:`KeyError`.
        If *False*, unknown names silently return ``eml_scalar(0.0)`` and the
        call is recorded in :attr:`missing_refs`.
    """

    def __init__(
        self,
        context: Dict[str, Any],
        *,
        strict: bool = True,
    ) -> None:
        self.context = {k: v for k, v in context.items() if v is not None}
        self.strict = strict
        self.missing_refs: list[str] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def eval(self, eml_description: str) -> float:
        """
        Parse and evaluate one ``eml_description`` string.

        Returns the ``float`` tension value of the resulting EML expression.

        Raises
        ------
        ParseError
            If the string cannot be parsed or the expression raises a
            non-KeyError exception.
        KeyError
            If ``strict=True`` and an ``eml_vec`` name is not in the context.
        """
        expr = self._parse(eml_description)
        ns = self._namespace()
        try:
            result = eval(expr, {"__builtins__": {}}, ns)  # noqa: S307
        except KeyError:
            raise
        except Exception as exc:
            raise ParseError(
                f"Failed to evaluate EML expression {expr!r}: {exc}"
            ) from exc

        if isinstance(result, EMLPoint):
            return result.tension()
        try:
            return float(result)
        except (TypeError, ValueError) as exc:
            raise ParseError(
                f"EML expression did not return a numeric value: {result!r}"
            ) from exc

    def try_eval(self, eml_description: str) -> Optional[float]:
        """Like :meth:`eval` but returns *None* on any error instead of raising."""
        try:
            return self.eval(eml_description)
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse(eml_description: str) -> str:
        """Extract the Python expression from an eml_description string.

        The commentary tail is separated from the expression by an em-dash
        in most entries, but not all: some append prose with other
        punctuation, e.g.

            EML: ops.add(...) = -23/24 ≈ -0.9583
            EML: ops.div(...) where a1=ops.div(...)
            EML: ops.div(...) at z=0

        Splitting on dashes alone left that tail attached, and the whole
        entry failed with "invalid syntax" or "invalid character '≈'" --
        reported as a broken expression when the expression itself was fine
        and only its annotation was in the way.

        So after the dash split, take the LONGEST leading substring that is
        a valid Python expression. Cuts are only attempted at bracket depth
        zero, so a comma or space inside a call is never a split point. If
        no prefix parses, the original string is returned unchanged and the
        caller reports the real syntax error rather than a truncation.
        """
        s = eml_description.strip()
        if not s.startswith(_PREFIX):
            raise ParseError(
                f"eml_description must start with 'EML: '; got {s[:40]!r}"
            )
        # Strip prefix
        s = s[len(_PREFIX):]
        # Strip human-readable tail after em-dash / en-dash / double-hyphen
        for sep in (" — ", " – ", " -- "):
            if sep in s:
                s = s.split(sep, 1)[0]
        s = s.strip()
        return _longest_valid_expression(s)

    def _eml_vec(self, name: str) -> float:
        """Context-bound eml_vec resolver. Returns a plain float.

        Returning float (not _LitNode) ensures ops.pow() uses the fast
        _ScaleNode path rather than the exp(mul(n, ln(base))) path, which
        breaks for base < 1 when n is a TensionPoint.
        """
        if name in self.context:
            val = self.context[name]
            try:
                return float(val)
            except (TypeError, ValueError):
                if self.strict:
                    raise KeyError(
                        f"eml_vec('{name}'): value {val!r} is not numeric"
                    )
                self.missing_refs.append(name)
                return 0.0
        if self.strict:
            raise KeyError(
                f"eml_vec('{name}'): not found in context (context has "
                f"{len(self.context)} entries)"
            )
        self.missing_refs.append(name)
        return 0.0

    #: Namespace entries that define the DSL itself. Context values may never
    #: shadow these -- a registry parameter innocently named "math" or "ops"
    #: would otherwise break every expression at once.
    _RESERVED = ("ops", "math", "eml_scalar", "eml_pi", "eml_vec")

    #: Exposed for tests: the namespace mapping type used for evaluation.
    _namespace_type = staticmethod(lambda ctx: _NormalisingNamespace(ctx))

    def _namespace(self) -> dict:
        # Context values are exposed as BARE NAMES as well as through
        # eml_vec(). Expressions are written both ways --
        #   ops.mul(ops.div(b3, chi_eff), ...)          <- bare
        #   ops.mul(ops.div(eml_vec('b3'), ...), ...)   <- explicit
        # -- and only the second used to resolve, so the first raised
        # NameError and was recorded as "did not evaluate". That misreads a
        # naming convention as a broken expression: 66 downstream parameters
        # were counted as failures with nothing actually wrong with them.
        # The context exists to supply exactly these values (see the callers'
        # _build_context), so it belongs in the namespace.
        ns = _NormalisingNamespace(self.context)
        # Reserved names are written last so they always win.
        ns.update({
            # Sign-aware ops shim — keeps the log-space EML algebra pure
            # internally but extracts and re-applies sign at the operator
            # boundary so expressions like ops.mul(ops.neg(...), x) give
            # the correct numeric result.
            "ops": _SignedOps,
            "math": math,
            # Return plain floats so ops.pow(x, eml_scalar(n)) uses the
            # _ScaleNode path (correct for fractional/negative exponents).
            "eml_scalar": float,
            "eml_pi": lambda: math.pi,
            "eml_vec": self._eml_vec,
        })
        return ns


# ---------------------------------------------------------------------------
# Sign-aware operator shim
# ---------------------------------------------------------------------------
#
# Pure EML defines  mul(a,b) = exp(ln a + ln b)  —  which is only valid for
# positive a, b. When an eml_description writes  ops.mul(ops.neg(x), y)  the
# inner neg flips the sign and the outer mul would then take ln of a negative
# number and silently lose the sign.
#
# This shim wraps the affected ops so they extract the sign separately:
#   mul(a, b)  →  sign(a)*sign(b) * pure_mul(|a|, |b|)
#   div(a, b)  →  same with /
#   pow(x, n)  →  sign-correct for integer exponents
#
# All other ops pass through unchanged.

def _to_float(x: Any) -> float:
    if isinstance(x, EMLPoint):
        return x.tension()
    return float(x)


class _SignedOpsMeta(type):
    """All-static-method passthrough to ops.* with sign-aware overrides."""

    def __getattr__(cls, name: str):
        return getattr(ops, name)


class _SignedOps(metaclass=_SignedOpsMeta):
    @staticmethod
    def mul(a: Any, b: Any, *rest: Any) -> float:
        """Product of two OR MORE operands.

        Multiplication is associative, and expression authors write it that
        way -- ``ops.mul(x, y, z)``. The two-argument signature rejected
        those with "mul() takes 2 positional arguments but 3 were given",
        which the cross-check recorded as an expression that failed to
        evaluate rather than as a call the shim could not accept.
        Folding left keeps the sign handling below unchanged.
        """
        result = _SignedOps._mul2(a, b)
        for operand in rest:
            result = _SignedOps._mul2(result, operand)
        return result

    @staticmethod
    def _mul2(a: Any, b: Any) -> float:
        af, bf = _to_float(a), _to_float(b)
        if af == 0.0 or bf == 0.0:
            return 0.0
        sign = (1 if af > 0 else -1) * (1 if bf > 0 else -1)
        magnitude = ops.mul(abs(af), abs(bf))
        return sign * _to_float(magnitude)

    @staticmethod
    def add(a: Any, b: Any, *rest: Any) -> float:
        """Sum of two OR MORE operands, for the same reason as mul.

        ``ops.add(x, y, z)`` appears in registry expressions; addition is
        associative and the arity limit was an implementation detail, not a
        statement about the algebra.
        """
        total = _to_float(ops.add(a, b))
        for operand in rest:
            total = _to_float(ops.add(total, operand))
        return total

    @staticmethod
    def div(a: Any, b: Any) -> float:
        af, bf = _to_float(a), _to_float(b)
        if bf == 0.0:
            # delegate to ops.div so it produces whatever the algebra says
            return _to_float(ops.div(a, b))
        sign = (1 if af >= 0 else -1) * (1 if bf > 0 else -1)
        if af == 0.0:
            return 0.0
        magnitude = ops.div(abs(af), abs(bf))
        return sign * _to_float(magnitude)

    @staticmethod
    def inv(x: Any) -> float:
        # ops.inv(x) is exp(neg(ln(x))) — exactly the log-space form this
        # shim exists to correct, but it was falling through __getattr__
        # unwrapped, so inv(-2) returned +0.5 while div(1, -2) returned -0.5.
        return _SignedOps.div(1.0, x)

    @staticmethod
    def pow(base: Any, exponent: Any) -> float:
        bf, ef = _to_float(base), _to_float(exponent)
        if bf >= 0:
            return _to_float(ops.pow(base, exponent))
        # negative base, integer exponent → sign(-1)^n * |base|^n
        if ef == int(ef):
            sign = -1.0 if int(ef) % 2 else 1.0
            return sign * _to_float(ops.pow(abs(bf), exponent))
        # fractional power of negative — would be complex; pass through (NaN)
        return _to_float(ops.pow(base, exponent))


def eml_eval(
    eml_description: str,
    context: Dict[str, Any],
    *,
    strict: bool = True,
) -> float:
    """
    One-shot convenience wrapper around :class:`EMLEvaluator`.

    Parameters
    ----------
    eml_description:
        A string starting with ``"EML: "`` followed by a Python expression
        using ``ops.*``, ``eml_scalar``, ``eml_pi``, and ``eml_vec``.
    context:
        ``{parameter_path: numeric_value}`` mapping for ``eml_vec`` lookups.
    strict:
        Forwarded to :class:`EMLEvaluator`.

    Returns
    -------
    float
        The ``.tension()`` value of the evaluated EML expression.
    """
    return EMLEvaluator(context, strict=strict).eval(eml_description)
