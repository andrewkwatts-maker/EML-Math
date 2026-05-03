"""
Raster renderers (PNG / PDF) for the EML formula layout dict.

Pillow is **lazy-imported** inside :meth:`render` and only required when
this renderer is actually called. The rest of the ``render`` package has
zero hard dependencies; ``import eml_math.render`` does not import Pillow.

Install with::

    pip install eml-math[render-raster]
"""
from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Tuple

from eml_math.render.edges import sample_path

__all__ = ["PNGRenderer", "PDFRenderer", "render_png", "render_pdf"]


_PILLOW_HINT = (
    "raster rendering requires Pillow — install with "
    "`pip install eml-math[render-raster]` or `pip install Pillow`"
)


def _require_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont   # noqa: F401
        return Image, ImageDraw, ImageFont
    except ImportError as e:
        raise ImportError(_PILLOW_HINT) from e


class PNGRenderer:
    """Rasterise a layout dict to PNG bytes using Pillow."""

    def render(
        self,
        layout: Dict[str, Any],
        *,
        scale: float = 2.0,
        edge_width: float = 3.0,
        leaf_radius: float = 5.0,
        junction_radius: float = 4.5,
        label_font_size: int = 18,
        background: Tuple[int, int, int, int] | None = None,
    ) -> bytes:
        Image, ImageDraw, _ = _require_pil()
        canvas = layout["canvas"]
        w, h = int(canvas["width"]), int(canvas["height"])
        sw, sh = int(w * scale), int(h * scale)
        bg = background if background is not None else (255, 255, 255, 0)
        img = Image.new("RGBA", (sw, sh), bg)
        draw = ImageDraw.Draw(img)
        nodes_by_id = {n["id"]: n for n in layout["nodes"]}
        direction = layout["direction"]

        # Edges.
        for e in layout["edges"]:
            a = nodes_by_id[e["from"]]
            b = nodes_by_id[e["to"]]
            pts = sample_path(
                e.get("style", layout.get("edge_style", "curve")),
                (a["x"] * scale, a["y"] * scale),
                (b["x"] * scale, b["y"] * scale),
                direction,
                samples=48,
            )
            stroke = tuple(e.get("color", a["color"])) + (255,)
            for i in range(len(pts) - 1):
                draw.line(
                    [pts[i], pts[i + 1]],
                    fill=stroke,
                    width=int(edge_width * scale),
                )

        # Nodes.
        for n in layout["nodes"]:
            cx, cy = n["x"] * scale, n["y"] * scale
            col = tuple(n["color"]) + (255,)
            r = (leaf_radius if n["is_leaf"] else junction_radius) * scale
            draw.ellipse(
                [cx - r, cy - r, cx + r, cy + r],
                fill=col, outline=(34, 34, 34, 255), width=max(1, int(scale)),
            )

        buf = BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()


class PDFRenderer:
    """One-page PDF wrapping the rasterised PNG."""

    def __init__(self) -> None:
        self._png = PNGRenderer()

    def render(self, layout: Dict[str, Any], **opts) -> bytes:
        Image, _, _ = _require_pil()
        png_bytes = self._png.render(layout, **opts)
        img = Image.open(BytesIO(png_bytes)).convert("RGB")
        scale = opts.get("scale", 2.0)
        buf = BytesIO()
        img.save(buf, format="PDF", resolution=int(72 * scale))
        return buf.getvalue()


def render_png(layout: Dict[str, Any], **opts) -> bytes:
    return PNGRenderer().render(layout, **opts)


def render_pdf(layout: Dict[str, Any], **opts) -> bytes:
    return PDFRenderer().render(layout, **opts)
