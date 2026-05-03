"""
SVG renderer for the EML formula layout dict.

Pure stdlib. Reads a layout dict, dispatches per-edge to the path generator
matching ``edge["style"]``, emits a self-contained ``<svg>`` string.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from eml_math.render.edges import path_for
from eml_math.render.palette import rgb_hex

__all__ = ["SVGRenderer", "render_svg"]


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;")
             .replace("<", "&lt;")
             .replace(">", "&gt;")
             .replace('"', "&quot;"))


def _node_by_id(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {n["id"]: n for n in layout["nodes"]}


class SVGRenderer:
    """Self-contained inline ``<svg>`` rendering of a layout dict."""

    def render(
        self,
        layout: Dict[str, Any],
        *,
        edge_width: float = 3.0,
        junction_radius: float = 4.5,
        leaf_radius: float = 5.0,
        label_font_size: int = 18,
        background: str = "",       # "" → transparent
        bias: float = 0.5,
        font_family: str = "Inter, Helvetica, Arial, sans-serif",
        show_leaf_labels: bool = True,
        show_output_label: bool = False,
        output_label: str = "Out",
        output_font_size: int = 22,
    ) -> str:
        """Render *layout* to an SVG string.

        Notes
        -----
        Edge style is read from each edge's ``style`` field (set by
        :func:`eml_math.render.compute_layout` according to its
        ``edge_style=`` argument). Per-edge overrides are honoured.
        """
        canvas = layout["canvas"]
        w, h = int(canvas["width"]), int(canvas["height"])
        direction = layout["direction"]
        nodes_by_id = _node_by_id(layout)

        parts: list[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{w}" height="{h}" viewBox="0 0 {w} {h}" '
            f'font-family="{font_family}">'
        ]
        if background:
            parts.append(
                f'<rect x="0" y="0" width="{w}" height="{h}" fill="{background}"/>'
            )

        # Edges.
        for e in layout["edges"]:
            a = nodes_by_id[e["from"]]
            b = nodes_by_id[e["to"]]
            d = path_for(
                e.get("style", layout.get("edge_style", "curve")),
                (a["x"], a["y"]), (b["x"], b["y"]),
                direction,
                bias=bias,
                waypoints=tuple(map(tuple, e.get("waypoints", ()))),
            )
            stroke = rgb_hex(tuple(e.get("color", a.get("color", (128, 128, 128)))))
            parts.append(
                f'<path d="{d}" stroke="{stroke}" stroke-width="{edge_width}" '
                f'fill="none" stroke-linecap="round"/>'
            )

        # Nodes — junctions as filled circles, leaves as small dots + label.
        for n in layout["nodes"]:
            col = rgb_hex(tuple(n["color"]))
            x, y = n["x"], n["y"]
            if n["is_leaf"]:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{leaf_radius:.1f}" '
                    f'fill="{col}" stroke="#222" stroke-width="0.6"/>'
                )
                if show_leaf_labels:
                    lx, ly = _label_offset(x, y, direction, label_font_size, "lead")
                    anchor = _text_anchor(direction, "lead")
                    parts.append(
                        f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{col}" '
                        f'text-anchor="{anchor}" font-weight="700" '
                        f'font-size="{label_font_size}">{_esc(n["label"])}</text>'
                    )
            else:
                parts.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{junction_radius:.1f}" '
                    f'fill="{col}" stroke="#222" stroke-width="0.8"/>'
                )

        # Optional output label below/right of the root.
        if show_output_label:
            root = _find_root(layout, nodes_by_id)
            if root is not None:
                ox, oy = _output_position(root["x"], root["y"], direction,
                                          output_font_size)
                anchor = _text_anchor(direction, "trail")
                parts.append(
                    f'<text x="{ox:.1f}" y="{oy:.1f}" text-anchor="{anchor}" '
                    f'font-weight="700" font-size="{output_font_size}" '
                    f'fill="#222">{_esc(output_label)}</text>'
                )

        parts.append("</svg>")
        return "\n".join(parts)


def render_svg(layout: Dict[str, Any], **opts) -> str:
    """Module-level helper — equivalent to ``SVGRenderer().render(layout, **opts)``."""
    return SVGRenderer().render(layout, **opts)


# ── direction-aware label placement helpers ──────────────────────────────────

def _label_offset(x: float, y: float, direction: str, font_size: int,
                  end: str) -> Tuple[float, float]:
    pad = 12.0
    if direction == "down":
        return (x, y - pad if end == "lead" else y + pad + font_size)
    if direction == "up":
        return (x, y + pad + font_size if end == "lead" else y - pad)
    if direction == "right":
        return (x - pad if end == "lead" else x + pad, y + font_size * 0.35)
    return (x + pad if end == "lead" else x - pad, y + font_size * 0.35)


def _text_anchor(direction: str, end: str) -> str:
    if direction in ("down", "up"):
        return "middle"
    if direction == "right":
        return "end" if end == "lead" else "start"
    return "start" if end == "lead" else "end"


def _output_position(rx: float, ry: float, direction: str,
                     font_size: int) -> Tuple[float, float]:
    pad = font_size * 0.7
    if direction == "down":  return (rx, ry + pad + font_size)
    if direction == "up":    return (rx, ry - pad)
    if direction == "right": return (rx + pad, ry + font_size * 0.35)
    return (rx - pad, ry + font_size * 0.35)


def _find_root(layout: Dict[str, Any],
               nodes_by_id: Dict[str, Dict[str, Any]]):
    """The root is the node that's never the ``to`` of any edge — wait, it
    IS the ``to`` of every edge in our convention. The root is the node that
    appears as ``to`` but never as ``from``."""
    children = {e["from"] for e in layout["edges"]}
    for n in layout["nodes"]:
        if not n["is_leaf"] and n["id"] not in children:
            return n
    # Fallback: deepest 0-depth node.
    for n in layout["nodes"]:
        if n.get("depth") == 0:
            return n
    return None
