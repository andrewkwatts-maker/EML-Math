"""eml_math.datasheet — JSON datasheet API for named math constants.

This is the eml-math layer's contribution to the cross-package ``Get()``
convention used throughout the EML stack:

* ``eml_math.Get('pi')``       → math constants (this module)
* ``metaphysica.Get('Up')``    → physics constants + quarks
* (future) ``periodica.Get('Iron')`` → material constants

Each ``Get(name)`` returns a JSON-serialisable ``dict`` carrying everything
a downstream tool needs to *both* render the constant for a human
(``latex``, ``description``, ``formula``) and rebuild its tree
programmatically (``eml_tree`` in compact form, plus the live
``EMLPoint`` is still reachable via :func:`eml_math.get_tree`).

This is a thin wrapper over the existing :func:`eml_math.get` /
:func:`eml_math.get_tree` / :func:`eml_math.list_symbols` API — it
doesn't re-derive any constants, just packages the existing ``SearchResult``
+ tree into the dict shape that downstream Get() consumers expect.

Quickstart
----------
>>> import eml_math
>>> eml_math.Get('pi')
{'name': 'pi', 'value': 3.141592653589793, 'formula': '4·arctan(1)',
 'eml_tree': [...], 'complexity': 1}
>>> eml_math.Get('phi')['value']
1.618033988749...

The return dict is deliberately compact so it can be embedded in
larger datasheets (e.g. metaphysica's quark datasheets reference EML
expressions that ultimately bottom out at one of these constants).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


def Get(name: str, *, as_json: bool = False) -> Dict[str, Any] | str:
    """Return a JSON-serialisable datasheet for the named math constant.

    Parameters
    ----------
    name : str
        A name from :func:`list_symbols`. Lookup is case-insensitive
        for the canonical names ('Pi', 'PI', 'pi' all work).
    as_json : bool, default False
        When True return the result as a JSON-encoded string instead
        of a dict. Convenience for shell users.

    Raises
    ------
    KeyError
        If the name is not in the symbol registry.

    Returns
    -------
    dict (or str if ``as_json``):

        {
          "name":        canonical-cased name (e.g. "pi"),
          "value":       float,                # numeric value
          "formula":     str,                  # human-readable formula
          "eml_tree":    list | None,          # to_compact() form
          "complexity":  int,                  # SearchResult.complexity
          "kind":        "math",               # const-tag for downstream filters
        }
    """
    # Lazy import — the heavy `discover` package only loads if Get() is called.
    from eml_math import get as _get
    from eml_math import get_tree as _get_tree
    from eml_math.tree import to_compact, parse_eml_tree

    try:
        result = _get(name)
    except KeyError as exc:
        raise KeyError(f"unknown math constant: {name!r}") from exc
    if result is None:
        # eml_math.get() returns None for unknown names. Normalise to KeyError
        # so the API matches dict[name] semantics.
        raise KeyError(f"unknown math constant: {name!r}")

    # Serialise the live EMLPoint tree by walking it. Each internal
    # eml(L, R) node maps to ["eml", "p", L, R]; numeric leaves to
    # [str(value), "C"]. Same codes the rest of the package uses
    # (see eml_math.tree.KIND_CHAR).
    eml_tree: Optional[list] = None
    try:
        live = _get_tree(name)
        eml_tree = _emlpoint_to_compact(live)
    except Exception:
        eml_tree = None

    # Pull numeric value out of the SearchResult (params[0] per the existing
    # convention). Fall back to evaluating the live tree.
    value: float
    if getattr(result, "params", None):
        try:
            value = float(result.params[0])
        except (TypeError, ValueError):
            value = float("nan")
    else:
        try:
            value = float(_get_tree(name).tension())
        except Exception:
            value = float("nan")

    out = {
        "name":       name,
        "value":      value,
        "formula":    getattr(result, "formula", str(result)),
        "eml_tree":   eml_tree,
        "complexity": getattr(result, "complexity", None),
        "kind":       "math",
    }

    if as_json:
        import json
        return json.dumps(out, indent=2, ensure_ascii=False, default=str)
    return out


def _emlpoint_to_compact(node: Any) -> Any:
    """Walk an :class:`eml_math.EMLPoint` into a compact JSON-friendly array.

    Mirrors :func:`eml_math.tree.to_compact` for EMLTreeNode. Leaf form is
    ``[label, kind_char]``; internal node form is
    ``[label, kind_char, child0, child1]``. Kind chars match
    :data:`eml_math.tree.KIND_CHAR` ('p' = primitive, 'C' = const).
    """
    from eml_math import EMLPoint
    # Numeric leaf (x is a plain float, not another EMLPoint)
    if not isinstance(node, EMLPoint):
        return [str(node), "C"]
    is_leaf = node.is_leaf() if hasattr(node, "is_leaf") else (
        not isinstance(node.x, EMLPoint) and not isinstance(node.y, EMLPoint)
    )
    if is_leaf:
        # Two-leaf eml(x_val, y_val). Render as eml( float, float ).
        return [
            "eml", "p",
            [str(node.x), "C"],
            [str(node.y), "C"],
        ]
    # Internal node: x and/or y is itself an EMLPoint.
    return [
        "eml", "p",
        _emlpoint_to_compact(node.x),
        _emlpoint_to_compact(node.y),
    ]


def list_constants() -> List[str]:
    """Return all names recognised by :func:`Get` (alias for ``list_symbols``)."""
    from eml_math import list_symbols
    return list(list_symbols())


__all__ = ["Get", "list_constants"]
