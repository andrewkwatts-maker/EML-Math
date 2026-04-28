"""
eml_math.flow — flow-diagram renderer for EML operator trees.

Visualises a (typically pure-eml) tree as a top-down flow:

* every **leaf** becomes an *input* at the top of the canvas, drawn in its
  own palette colour;
* each binary internal node is a **junction** — no label, since the only
  internal operator is ``eml(L, R) = exp(L) − ln(R)``: the left branch is
  the *exp-side* input, the right branch is the *ln-side* input;
* each junction's colour is the average of its two children's colours, so
  related sub-trees share visual tone;
* the **output** sits at the bottom of the canvas, labelled *Out*.

Render targets
--------------
``flow_svg(node)``   — self-contained ``<svg>...</svg>`` string
``flow_png(node)``   — ``bytes`` of a PNG (uses cairosvg if available, else PIL)
``flow_html(node)``  — small ``<div class="eml-flow">…</div>`` snippet for embedding

Use directly or via the convenience methods on :class:`EMLTreeNode`.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Iterable, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from eml_math.tree import EMLTreeNode

__all__ = [
    "DEFAULT_PALETTE",
    "flow_svg",
    "flow_png",
    "flow_pdf",
    "flow_html",
]


# ── Default palette ──────────────────────────────────────────────────────────
# Distinct, vivid hues that survive averaging without becoming muddy.

DEFAULT_PALETTE: Sequence[Tuple[int, int, int]] = (
    (231,  29,  54),  # red
    ( 33, 196, 196),  # cyan
    ( 31,  79, 231),  # blue
    (231,  33, 177),  # magenta
    ( 33, 196,  79),  # green
    (231, 131,  33),  # orange
    (131,  33, 231),  # purple
    (131,  79,  33),  # brown
    ( 33,  79, 131),  # navy
    (196, 231,  33),  # lime
    (196,  33, 131),  # rose
    ( 79, 131,  33),  # olive
)


# ── Internal layout state attached to nodes ──────────────────────────────────
#
# We deliberately do NOT add a dataclass field — flow rendering is opt-in and
# the layout helpers just write _fx/_fy/_fcolor onto whatever node objects
# they receive.

def _collect_leaves(node: "EMLTreeNode") -> List["EMLTreeNode"]:
    if not node.children:
        return [node]
    out: List["EMLTreeNode"] = []
    for c in node.children:
        out.extend(_collect_leaves(c))
    return out


def _binarize(node: "EMLTreeNode") -> "EMLTreeNode":
    """Pre-process tree for flow rendering: every internal node ends up binary.

    Rules
    -----
    * Leaf — pass through unchanged.
    * Unary internal — skip the node entirely; the child becomes the result.
      This eliminates "1 → 1" junction artefacts from operators like sqrt,
      sin, exp, ln, neg.
    * Binary internal — keep as-is.
    * N-ary internal (N>2) — left-fold into nested binary so the diagram
      shows N−1 binary joins instead of an N-into-1 merge. So
      ``std(a, b, c)`` renders as ``[[a, b], c]``.
    """
    # imported lazily to avoid the circular import at module load time
    from eml_math.tree import EMLTreeNode

    if not node.children:
        return node

    cc = [_binarize(c) for c in node.children]
    if len(cc) == 1:
        return cc[0]   # collapse unary
    if len(cc) == 2:
        return EMLTreeNode(label=node.label, kind=node.kind, children=cc,
                           eml_form=node.eml_form)
    # n-ary → left-fold
    folded = cc[0]
    for c in cc[1:]:
        folded = EMLTreeNode(label=node.label, kind=node.kind,
                             children=[folded, c], eml_form=node.eml_form)
    return folded


def _height(node: "EMLTreeNode") -> int:
    """Tree height: leaves are 0, root is the maximum."""
    if not node.children:
        return 0
    return 1 + max(_height(c) for c in node.children)


def _assign_xy(
    node: "EMLTreeNode",
    leaf_x: dict,
    top: float,
    layer_h: float,
) -> None:
    h = _height(node)
    node._fy = top + h * layer_h
    if not node.children:
        node._fx = leaf_x[id(node)]
        return
    for c in node.children:
        _assign_xy(c, leaf_x, top, layer_h)
    node._fx = sum(c._fx for c in node.children) / len(node.children)


def _assign_colors(
    node: "EMLTreeNode",
    leaf_color: dict,
) -> Tuple[float, float, float]:
    if not node.children:
        rgb = leaf_color[id(node)]
        node._fcolor = rgb
        return rgb
    rgbs = [_assign_colors(c, leaf_color) for c in node.children]
    avg = (
        sum(c[0] for c in rgbs) / len(rgbs),
        sum(c[1] for c in rgbs) / len(rgbs),
        sum(c[2] for c in rgbs) / len(rgbs),
    )
    node._fcolor = avg
    return avg


def _rgb_hex(rgb: Tuple[float, float, float]) -> str:
    return "#{:02X}{:02X}{:02X}".format(
        max(0, min(255, int(round(rgb[0])))),
        max(0, min(255, int(round(rgb[1])))),
        max(0, min(255, int(round(rgb[2])))),
    )


def _esc(s: str) -> str:
    return (
        s.replace("&", "&amp;")
         .replace("<", "&lt;")
         .replace(">", "&gt;")
    )


# ── Layout entry point ───────────────────────────────────────────────────────

def _layout(
    node: "EMLTreeNode",
    *,
    width: int,
    height: int,
    margin_top: float,
    margin_bottom: float,
    margin_lr: float,
    palette: Sequence[Tuple[int, int, int]],
) -> Tuple["EMLTreeNode", List["EMLTreeNode"]]:
    """Returns (binarised_root, leaves_left_to_right)."""
    # Always binarise first — guarantees every internal node has 2 children
    # so the flow diagram is a true binary tree. Unaries collapse, n-aries
    # left-fold.
    node = _binarize(node)

    leaves = _collect_leaves(node)
    n = len(leaves)
    # x-positions for leaves: evenly spaced
    if n == 1:
        leaf_x = {id(leaves[0]): width / 2}
    else:
        usable = width - 2 * margin_lr
        leaf_x = {
            id(l): margin_lr + usable * i / (n - 1)
            for i, l in enumerate(leaves)
        }
    # vertical layer spacing — always span the full canvas regardless of depth
    h = max(_height(node), 1)
    layer_h = (height - margin_top - margin_bottom) / h
    _assign_xy(node, leaf_x, margin_top, layer_h)
    # colours: assign one palette colour per leaf, then propagate up by averaging
    leaf_color = {
        id(l): tuple(palette[i % len(palette)]) for i, l in enumerate(leaves)
    }
    _assign_colors(node, leaf_color)
    return node, leaves


# ── SVG renderer ─────────────────────────────────────────────────────────────

def flow_svg(
    node: "EMLTreeNode",
    *,
    width: int = 800,
    height: int = 600,
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    show_output_label: bool = True,
    output_label: str = "Out",
    label_font_size: int = 18,
    output_font_size: int = 22,
    edge_width: float = 3.0,
    junction_radius: float = 4.0,
    background: Optional[str] = None,
) -> str:
    """Render *node* as a flow-diagram SVG string.

    Parameters
    ----------
    node :
        Root :class:`EMLTreeNode` of the tree to render. Works on any
        expansion mode but is designed for pure-eml trees where every
        internal node is the binary primitive ``eml(L, R)``.
    width, height :
        Canvas dimensions in pixels.
    palette :
        Sequence of ``(r, g, b)`` tuples (0-255) to colour the leaf inputs.
        Cycles if the tree has more leaves than palette entries.
    show_output_label, output_label :
        Whether to draw the bottom *Out* label and what text to use.
    label_font_size, output_font_size :
        Font sizes for the leaf labels and the output label.
    edge_width :
        Stroke width for the connecting curves.
    junction_radius :
        Radius of the small circle drawn at each internal-node junction.
    background :
        If given, a solid-fill ``<rect>`` is drawn first as the canvas
        background (useful for PNG export onto a non-transparent surface).
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    margin_top    = float(label_font_size) * 2.2
    margin_bottom = float(output_font_size) * 2.0 if show_output_label else 30.0
    # Margin must accommodate half the widest leaf label so the leftmost /
    # rightmost text doesn't run off the canvas. Approximate width as
    # 0.6 * font_size per character (decent for proportional fonts).
    leaves_preview = _collect_leaves(_binarize(node))
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    half_label_w   = 0.5 * 0.6 * label_font_size * max_label_len
    margin_lr      = max(40.0, half_label_w + 12.0)

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_top=margin_top, margin_bottom=margin_bottom, margin_lr=margin_lr,
        palette=palette,
    )

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">',
    ]
    if background:
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>')

    # ── edges (drawn first so junctions/labels sit on top) ──────────────────
    _emit_flow_edges(node, parts, edge_width)

    # ── if the binarised tree is a bare leaf (e.g. exp(neg(x)) collapsed
    #    all unaries away), draw a direct line from the leaf to the output
    #    position so the diagram still shows a connection.
    output_y = height - margin_bottom
    if not node.children:
        col = _rgb_hex(node._fcolor)
        my = (node._fy + output_y) / 2
        parts.append(
            f'<path d="M{node._fx:.1f},{node._fy:.1f} '
            f'C{node._fx:.1f},{my:.1f} {width/2:.1f},{my:.1f} {width/2:.1f},{output_y:.1f}" '
            f'stroke="{col}" stroke-width="{edge_width}" fill="none" stroke-linecap="round"/>'
        )

    # ── junction dots at every internal node ────────────────────────────────
    _emit_junctions(node, parts, junction_radius)

    # ── input labels at the top ─────────────────────────────────────────────
    for leaf in leaves:
        col = _rgb_hex(leaf._fcolor)
        x = leaf._fx
        y = leaf._fy - 12  # just above the curve start
        parts.append(
            f'<text x="{x:.1f}" y="{y:.1f}" fill="{col}" text-anchor="middle" '
            f'font-weight="700" font-size="{label_font_size}">{_esc(leaf.label)}</text>'
        )

    # ── output label at the bottom ──────────────────────────────────────────
    if show_output_label:
        ox = width / 2
        oy = height - margin_bottom * 0.35
        parts.append(
            f'<text x="{ox:.1f}" y="{oy:.1f}" text-anchor="middle" '
            f'font-weight="700" font-size="{output_font_size}" '
            f'fill="#222">{_esc(output_label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def _emit_flow_edges(
    node: "EMLTreeNode",
    out: List[str],
    edge_width: float,
) -> None:
    if not node.children:
        return
    px, py = node._fx, node._fy
    for c in node.children:
        cx, cy = c._fx, c._fy
        col = _rgb_hex(c._fcolor)
        # smooth S-curve from child (higher up) to parent (lower down)
        my = (py + cy) / 2
        out.append(
            f'<path d="M{cx:.1f},{cy:.1f} C{cx:.1f},{my:.1f} '
            f'{px:.1f},{my:.1f} {px:.1f},{py:.1f}" '
            f'stroke="{col}" stroke-width="{edge_width}" '
            f'fill="none" stroke-linecap="round"/>'
        )
        _emit_flow_edges(c, out, edge_width)


def _emit_junctions(
    node: "EMLTreeNode",
    out: List[str],
    radius: float,
) -> None:
    if not node.children:
        return
    col = _rgb_hex(node._fcolor)
    out.append(
        f'<circle cx="{node._fx:.1f}" cy="{node._fy:.1f}" r="{radius:.1f}" '
        f'fill="{col}" stroke="#222" stroke-width="0.8"/>'
    )
    for c in node.children:
        _emit_junctions(c, out, radius)


# ── HTML wrapper ─────────────────────────────────────────────────────────────

def flow_html(
    node: "EMLTreeNode",
    *,
    container_id: Optional[str] = None,
    container_class: str = "eml-flow",
    inline_style: Optional[str] = None,
    **svg_kw,
) -> str:
    """Wrap :func:`flow_svg` output in a ``<div>`` for direct page embedding.

    The returned string is safe to inject into any HTML page — the SVG is
    self-contained (no external font/stylesheet refs).
    """
    svg = flow_svg(node, **svg_kw)
    cid = f' id="{container_id}"' if container_id else ""
    style = f' style="{inline_style}"' if inline_style else ""
    return f'<div class="{container_class}"{cid}{style}>{svg}</div>'


# ── PNG renderer ─────────────────────────────────────────────────────────────

def flow_png(
    node: "EMLTreeNode",
    *,
    width: int = 800,
    height: int = 600,
    scale: float = 2.0,
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    background: str = "white",
    **svg_kw,
) -> bytes:
    """Rasterise the flow diagram to a PNG.

    Tries ``cairosvg`` first (best quality, vector→raster). Falls back to a
    direct Pillow renderer if cairosvg is unavailable.

    The ``scale`` parameter controls the rasterisation resolution
    multiplier (default 2× for crisp output on hi-DPI displays).
    """
    # Attempt cairosvg path first
    try:
        import cairosvg  # type: ignore
        svg = flow_svg(
            node, width=width, height=height,
            palette=palette, background=background, **svg_kw,
        )
        return cairosvg.svg2png(
            bytestring=svg.encode("utf-8"),
            output_width=int(width * scale),
            output_height=int(height * scale),
        )
    except ImportError:
        pass

    # Pillow fallback — direct draw using the same layout logic
    return _flow_png_pillow(
        node, width=width, height=height, scale=scale,
        palette=palette, background=background, **svg_kw,
    )


def _bezier_points(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
    p3: Tuple[float, float],
    *,
    samples: int = 32,
) -> List[Tuple[float, float]]:
    """Sample a cubic Bezier curve at *samples* equally-spaced t."""
    out: List[Tuple[float, float]] = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        x = (u**3 * p0[0]
             + 3*u*u*t * p1[0]
             + 3*u*t*t * p2[0]
             + t**3 * p3[0])
        y = (u**3 * p0[1]
             + 3*u*u*t * p1[1]
             + 3*u*t*t * p2[1]
             + t**3 * p3[1])
        out.append((x, y))
    return out


def flow_pdf(
    node: "EMLTreeNode",
    *,
    width: int = 800,
    height: int = 600,
    scale: float = 2.0,
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    background: str = "white",
    **svg_kw,
) -> bytes:
    """Rasterise the flow diagram to a one-page PDF.

    Tries ``cairosvg`` first (true-vector PDF). Falls back to Pillow's
    PDF writer (raster image embedded in a PDF page) if cairosvg is not
    installed.
    """
    # Vector-PDF path
    try:
        import cairosvg  # type: ignore
        svg = flow_svg(
            node, width=width, height=height,
            palette=palette, background=background, **svg_kw,
        )
        return cairosvg.svg2pdf(bytestring=svg.encode("utf-8"))
    except ImportError:
        pass

    # Raster-PDF fallback via Pillow
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError(
            "flow_pdf() requires either 'cairosvg' or 'Pillow' to be installed."
        )
    png_bytes = flow_png(
        node, width=width, height=height, scale=scale,
        palette=palette, background=background, **svg_kw,
    )
    img = Image.open(BytesIO(png_bytes)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PDF", resolution=int(72 * scale))
    return buf.getvalue()


def _flow_png_pillow(
    node: "EMLTreeNode",
    *,
    width: int,
    height: int,
    scale: float,
    palette: Optional[Sequence[Tuple[int, int, int]]],
    background: str,
    show_output_label: bool = True,
    output_label: str = "Out",
    label_font_size: int = 18,
    output_font_size: int = 22,
    edge_width: float = 3.0,
    junction_radius: float = 4.0,
    **_: object,
) -> bytes:
    from io import BytesIO
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError(
            "flow_png() requires either 'cairosvg' or 'Pillow' to be installed."
        )

    if palette is None:
        palette = DEFAULT_PALETTE

    # All sizes scaled up for hi-DPI rendering
    W = int(width * scale)
    H = int(height * scale)
    fs_label  = int(label_font_size * scale)
    fs_output = int(output_font_size * scale)
    ew        = max(1, int(edge_width * scale))
    jr        = max(1, int(junction_radius * scale))

    margin_top    = float(label_font_size) * 2.2
    margin_bottom = float(output_font_size) * 2.0 if show_output_label else 30.0
    leaves_preview = _collect_leaves(_binarize(node))
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    half_label_w   = 0.5 * 0.6 * label_font_size * max_label_len
    margin_lr      = max(40.0, half_label_w + 12.0)

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_top=margin_top, margin_bottom=margin_bottom, margin_lr=margin_lr,
        palette=palette,
    )

    img = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(img)

    # Try to load a decent font; fall back to default bitmap if not found.
    def _try_font(size: int) -> "ImageFont.ImageFont":
        for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
            try:
                return ImageFont.truetype(name, size)
            except (OSError, IOError):
                continue
        return ImageFont.load_default()
    font_label  = _try_font(fs_label)
    font_output = _try_font(fs_output)

    # Edges
    def _draw_edges(n):
        if not n.children:
            return
        for c in n.children:
            p0 = (c._fx * scale, c._fy * scale)
            p3 = (n._fx * scale, n._fy * scale)
            my = (p0[1] + p3[1]) / 2
            p1 = (p0[0], my)
            p2 = (p3[0], my)
            pts = _bezier_points(p0, p1, p2, p3, samples=40)
            col = tuple(int(round(v)) for v in c._fcolor)
            draw.line(pts, fill=col, width=ew, joint="curve")
            _draw_edges(c)
    _draw_edges(node)

    # If binarised tree is a bare leaf (all unaries collapsed), draw a
    # synthetic edge from the leaf to where the output label sits.
    if not node.children:
        output_y = (height - margin_bottom) * scale
        p0 = (node._fx * scale, node._fy * scale)
        p3 = (W / 2, output_y)
        my = (p0[1] + p3[1]) / 2
        p1 = (p0[0], my)
        p2 = (p3[0], my)
        pts = _bezier_points(p0, p1, p2, p3, samples=40)
        col = tuple(int(round(v)) for v in node._fcolor)
        draw.line(pts, fill=col, width=ew, joint="curve")

    # Junction dots
    def _draw_junctions(n):
        if not n.children:
            return
        col = tuple(int(round(v)) for v in n._fcolor)
        cx, cy = n._fx * scale, n._fy * scale
        draw.ellipse(
            [cx - jr, cy - jr, cx + jr, cy + jr],
            fill=col, outline=(34, 34, 34), width=max(1, ew // 3),
        )
        for c in n.children:
            _draw_junctions(c)
    _draw_junctions(node)

    # Leaf labels
    for leaf in leaves:
        col = tuple(int(round(v)) for v in leaf._fcolor)
        x = leaf._fx * scale
        y = leaf._fy * scale - fs_label * 0.7 - 12 * scale
        # PIL draws from top-left; centre by measuring text
        bbox = draw.textbbox((0, 0), leaf.label, font=font_label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x - tw / 2, y - th / 2), leaf.label, fill=col, font=font_label)

    # Output label
    if show_output_label:
        bbox = draw.textbbox((0, 0), output_label, font=font_output)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = W / 2 - tw / 2
        y = H - margin_bottom * 0.35 * scale - th / 2
        draw.text((x, y), output_label, fill=(34, 34, 34), font=font_output)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
