"""HTML wrapper around :class:`SVGRenderer`.

Renders a layout to ``<div class="eml-flow">…SVG…</div>`` for direct page
embedding. Pure stdlib.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from eml_math.render.renderers.svg import SVGRenderer

__all__ = ["HTMLRenderer", "render_html"]


class HTMLRenderer:
    """Wraps SVG output in a styled ``<div>`` for HTML embedding."""

    def __init__(self) -> None:
        self._svg = SVGRenderer()

    def render(
        self,
        layout: Dict[str, Any],
        *,
        container_id: Optional[str] = None,
        container_class: str = "eml-flow",
        inline_style: Optional[str] = None,
        **svg_opts: Any,
    ) -> str:
        svg = self._svg.render(layout, **svg_opts)
        cid = f' id="{container_id}"' if container_id else ""
        style = f' style="{inline_style}"' if inline_style else ""
        return f'<div class="{container_class}"{cid}{style}>{svg}</div>'


def render_html(layout: Dict[str, Any], **opts) -> str:
    """Module-level helper — equivalent to ``HTMLRenderer().render(layout, **opts)``."""
    return HTMLRenderer().render(layout, **opts)
