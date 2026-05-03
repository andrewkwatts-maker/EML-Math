"""
eml_math.render — pluggable diagram renderer for EML formula trees.

Pipeline (three independent stages):

    raw formula JSON  ──compute_layout()──►  layout dict  ──Renderer().render()──►  output

* **Raw formula JSON** — the structural ``to_dict()`` form on EMLTreeNode.
  No positions, no colours, no canvas — just labels, kinds, and children.
* **Layout dict** — node positions + edge endpoints + colours + edge styles.
  This is the contract every renderer consumes.
* **Renderer** — anything implementing ``render(layout, **opts) -> str | bytes``.

Built-in renderers
------------------
``SVGRenderer``    — stdlib only, returns a self-contained ``<svg>`` string
``HTMLRenderer``   — stdlib only, wraps SVG in a ``<div>`` for embedding
``PNGRenderer``    — needs Pillow (optional dep ``eml-math[render-raster]``)
``PDFRenderer``    — needs Pillow

Custom renderers
----------------
Implement the ``Renderer`` protocol and pass an instance directly, or register
a name globally::

    from eml_math.render import Renderer, register, compute_layout

    class GraphMLRenderer:
        def render(self, layout, **opts):
            ...

    register("graphml", GraphMLRenderer())
    out = render_with("graphml", layout)

Edge styles
-----------
``"straight"`` — single ``L`` line; tightest, sharp angles.
``"curve"``    — single cubic Bezier; soft and natural (default).
``"spline"``   — Catmull-Rom through optional waypoints; routes around obstacles.
"""
from __future__ import annotations

from typing import Any, Dict, Protocol, Union, runtime_checkable

from eml_math.render.layout import compute_layout, EDGE_STYLES, DIRECTIONS
from eml_math.render.palette import DEFAULT_PALETTE, pastel_for_label
from eml_math.render.renderers.svg import SVGRenderer
from eml_math.render.renderers.html import HTMLRenderer
from eml_math.render.renderers.raster import PNGRenderer, PDFRenderer

__all__ = [
    "Renderer",
    "compute_layout",
    "register",
    "renderer_for",
    "render_with",
    "DEFAULT_PALETTE",
    "pastel_for_label",
    "EDGE_STYLES",
    "DIRECTIONS",
    "SVGRenderer",
    "HTMLRenderer",
    "PNGRenderer",
    "PDFRenderer",
]


@runtime_checkable
class Renderer(Protocol):
    """Anything callable as ``renderer.render(layout, **opts)``.

    ``layout`` is the dict returned by :func:`compute_layout`.
    Return value is ``str`` (textual formats) or ``bytes`` (raster formats).
    """

    def render(self, layout: Dict[str, Any], **opts: Any) -> Union[str, bytes]: ...


# ── Renderer registry ────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Renderer] = {}


def register(name: str, renderer: Renderer) -> None:
    """Register *renderer* under *name* so :func:`render_with` can find it."""
    if not hasattr(renderer, "render"):
        raise TypeError(
            f"renderer for {name!r} must have a .render(layout, **opts) method"
        )
    _REGISTRY[name] = renderer


def renderer_for(name: str) -> Renderer:
    """Look up a registered renderer by name. Raises KeyError if absent."""
    try:
        return _REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"no renderer registered for {name!r}; "
            f"known: {sorted(_REGISTRY)}"
        ) from e


def render_with(name: str, layout: Dict[str, Any], **opts: Any) -> Union[str, bytes]:
    """Look up *name* in the registry and render *layout* with it."""
    return renderer_for(name).render(layout, **opts)


# Pre-register the built-ins.
register("svg", SVGRenderer())
register("html", HTMLRenderer())
register("png", PNGRenderer())
register("pdf", PDFRenderer())
