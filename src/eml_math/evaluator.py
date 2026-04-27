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

import math
import re
from typing import Any, Dict, Optional

import eml_math.operators as ops
from eml_math.point import EMLPoint, _LitNode

__all__ = ["EMLEvaluator", "eml_eval", "ParseError"]

# Separator between expression and human-readable description.
_SEP = " — "
_PREFIX = "EML: "

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
        """Extract the Python expression from an eml_description string."""
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
        return s.strip()

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

    def _namespace(self) -> dict:
        return {
            "ops": ops,
            "math": math,
            # Return plain floats so ops.pow(x, eml_scalar(n)) uses the
            # _ScaleNode path (correct for fractional/negative exponents).
            "eml_scalar": float,
            "eml_pi": lambda: math.pi,
            "eml_vec": self._eml_vec,
        }


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
