"""
eml_math.flow_layout — JSON-intermediate pipeline for the flow renderer.

The default flow_svg / flow_png / flow_pdf API is a one-shot: parse → layout
→ render. For more artistic control, this module exposes the *intermediate*
layout graph as a serialisable dict so callers can apply post-processing
transforms (move nodes, smooth curves, spread sub-trees …) before rendering.

Pipeline
--------

    1. parse:        EML expression  →  EMLTreeNode
    2. layout:       EMLTreeNode     →  layout dict   (this module's `to_layout`)
    3. post-process: layout dict     →  layout dict   (apply zero or more)
    4. render:       layout dict     →  SVG / PNG / PDF   (`render_svg` etc.)

Layout dict schema (v1)::

    {
      "schema": "eml-flow-layout/v1",
      "width":  720,
      "height": 440,
      "direction": "down",
      "output_label": "E"   |  ["x_+", "x_-"],
      "nodes": [
        { "id": "n0", "label": "m", "kind": "vec",
          "x": 60.0, "y": 220.0, "color": [242, 165, 152],
          "is_leaf": true,  "is_inline": false  },
        ...
      ],
      "edges": [
        { "from": "n0", "to": "n5", "color": [...],
          "vertical_bias": 0.5  },
        ...
      ]
    }

Built-in post-processes
-----------------------
    gentle_curves(layout, bend=0.3)
        Reduce edge `vertical_bias` so all branches read as long, gentle
        sweeps rather than sharp S-curves. Lower `bend` = straighter.

    tighten_base(layout, by=0.4)
        Pull root-side junction x-coords toward the parent's x by `by`,
        reducing wiggle in the deeper layers of the tree.

    spread_horizontal(layout, factor=1.3)
        Multiply every node's cross-axis position by `factor` so the tree
        spreads outward — more breathing room between input chains.

Use a sequence of post-processes by composing them yourself.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from eml_math.tree import EMLTreeNode

__all__ = [
    "to_layout",
    "render_svg",
    "render_png",
    "render_pdf",
    # Built-in post-processes
    "gentle_curves",
    "flowing_sideways",
    "tighten_base",
    "spread_horizontal",
    "fit_to_canvas",
    "organic_layout",
]


# ── to_layout: EMLTreeNode → JSON-friendly dict ─────────────────────────────

def to_layout(
    node: "EMLTreeNode",
    *,
    width: int = 720,
    height: int = 440,
    direction: str = "down",
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    output_label: Any = "Out",
    expand_symbols: bool = False,
    merge_inputs: bool = False,
    inline_constants: bool = False,
    fixed_colors: Optional[dict] = None,
    bypass_identity_blend: bool = True,
    random_palette: bool = False,
    label_font_size: int = 18,
    output_font_size: int = 22,
) -> Dict[str, Any]:
    """Run layout on *node* and return the position/colour graph as a JSON-
    serialisable dict.  Apply post-processes to it (see this module's
    `gentle_curves` etc.), then pass the dict to :func:`render_svg` /
    :func:`render_png` / :func:`render_pdf`.
    """
    from eml_math.flow import (
        DEFAULT_PALETTE, FIXED_COLORS, _layout, _binarize,
    )

    if palette is None:
        palette = DEFAULT_PALETTE
    fc = FIXED_COLORS if fixed_colors is None else fixed_colors

    multi_output = not isinstance(output_label, str)
    primary_label_size  = float(label_font_size) * 2.2
    primary_output_size = float(output_font_size) * 2.0
    if multi_output:
        primary_output_size *= 1.6
    if merge_inputs:
        primary_label_size = max(primary_label_size, height * 0.5)

    from eml_math.flow import _expand_symbols_in_tree, _collect_leaves
    preview_root = _binarize(_expand_symbols_in_tree(node) if expand_symbols else node)
    leaves_preview = _collect_leaves(preview_root)
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    cross_margin   = max(40.0, 0.5 * 0.6 * label_font_size * max_label_len + 12.0)

    laid_root, leaves = _layout(
        node,
        width=width, height=height,
        margin_lead=primary_label_size,
        margin_trail=primary_output_size,
        margin_cross=cross_margin,
        palette=palette, direction=direction,
        expand_symbols=expand_symbols,
        fixed_colors=fc,
        bypass_identity_blend=bypass_identity_blend,
        random_palette=random_palette,
    )

    # Reposition 0/1 (and inline-numeric) leaves to short stubs.
    from eml_math.flow import _stub_inline_leaves
    _stub_inline_leaves(laid_root, direction=direction, fixed_labels=fc.keys(),
                         inline_constants=inline_constants,
                         label_font_size=label_font_size)

    # ── Walk the tree and assign stable ids ─────────────────────────────
    nodes: List[dict] = []
    edges: List[dict] = []
    counter = [0]
    def _id():
        counter[0] += 1
        return f"n{counter[0] - 1}"

    leaf_id_set = set()
    def _walk(n, parent_id=None):
        nid = _id()
        is_leaf = not n.children
        nodes.append({
            "id":        nid,
            "label":     n.label,
            "kind":      n.kind,
            "x":         float(n._fx),
            "y":         float(n._fy),
            "color":     [int(round(v)) for v in n._fcolor],
            "is_leaf":   is_leaf,
            "is_inline": is_leaf and n.label in fc,
        })
        if is_leaf:
            leaf_id_set.add(nid)
        if parent_id is not None:
            edges.append({
                "from":          nid,         # child id
                "to":            parent_id,   # parent id
                "color":         [int(round(v)) for v in n._fcolor],
                "vertical_bias": 0.5,
            })
        for c in n.children:
            _walk(c, nid)

    _walk(laid_root)

    return {
        "schema":        "eml-flow-layout/v1",
        "width":         width,
        "height":        height,
        "direction":     direction,
        "output_label":  list(output_label) if multi_output else output_label,
        "nodes":         nodes,
        "edges":         edges,
        "render_hints": {
            "label_font_size":     label_font_size,
            "output_font_size":    output_font_size,
            "primary_label_size":  primary_label_size,
            "primary_output_size": primary_output_size,
            "cross_margin":        cross_margin,
            "merge_inputs":        merge_inputs,
            "inline_constants":    inline_constants,
            "omit_identity_labels": True,
        },
    }


# ── Built-in post-processes ─────────────────────────────────────────────────

def gentle_curves(layout: Dict[str, Any], *, bend: float = 0.55) -> Dict[str, Any]:
    """Set every edge's vertical_bias to *bend* (0..1) for flowing curves.

    The cubic-Bezier control points sit *bend* of the way along the primary
    axis from each endpoint, giving:
        bend ≈ 0.5  — classic S-curve (default 0.55 here adds a touch of
                      vertical lead-in / lead-out so curves *flow* rather
                      than read as straight lines that meet at angles)
        bend > 0.7  — strongly vertical lead-in/out, big lateral sweep
        bend < 0.3  — near-straight line; can look angular at junctions

    Returns a new layout dict (does not mutate the input).
    """
    bend = max(0.05, min(0.95, float(bend)))
    out = _shallow_copy(layout)
    out["edges"] = [
        {**e, "vertical_bias": bend}
        for e in layout["edges"]
    ]
    return out


def flowing_sideways(layout: Dict[str, Any], *,
                     amplitude: float = 0.35,
                     bend: float = 0.55) -> Dict[str, Any]:
    """Add a sideways bow to every edge so the diagram has organic sweep
    instead of straight vertical chains.

    For each edge, the cubic-Bezier control points are perturbed on the
    cross axis by `amplitude × edge_length`, alternating sign by edge so
    consecutive branches lean opposite directions (creates a snake-like
    flow rather than all bending the same way).

    `bend` sets the primary-axis component (same meaning as
    `gentle_curves`).
    """
    bend = max(0.05, min(0.95, float(bend)))
    amp  = max(-1.0, min(1.0, float(amplitude)))
    direction = layout.get("direction", "down")
    cross_is_x = direction in ("down", "up")

    nodes_by_id = {n["id"]: n for n in layout["nodes"]}

    out = _shallow_copy(layout)
    new_edges = []
    for i, e in enumerate(layout["edges"]):
        c = nodes_by_id[e["from"]]
        p = nodes_by_id[e["to"]]
        # Edge length on primary axis.
        if cross_is_x:
            primary_len = abs(p["y"] - c["y"])
        else:
            primary_len = abs(p["x"] - c["x"])
        side = (1.0 if (i % 2 == 0) else -1.0) * amp * primary_len
        new_edges.append({**e,
                          "vertical_bias": bend,
                          "sideways_offset": side})
    out["edges"] = new_edges
    return out


def tighten_base(layout: Dict[str, Any], *, by: float = 0.4) -> Dict[str, Any]:
    """Pull root-side junctions toward their parent on the cross axis.

    `by` ∈ [0, 1]: the fraction by which to drag a child's cross coordinate
    toward its parent's. 0 = no change; 1 = collapse onto parent.

    Operates iteratively from the root outward; the closer to the root,
    the more dampening — by default this trims the wiggle that builds up
    in the deeper layers of long pure-EML chains.
    """
    by = max(0.0, min(1.0, float(by)))
    nodes_by_id = {n["id"]: dict(n) for n in layout["nodes"]}
    children_of: Dict[str, List[str]] = {n["id"]: [] for n in layout["nodes"]}
    parent_of: Dict[str, Optional[str]] = {n["id"]: None for n in layout["nodes"]}
    for e in layout["edges"]:
        children_of[e["to"]].append(e["from"])
        parent_of[e["from"]] = e["to"]

    direction = layout.get("direction", "down")
    cross_axis = "x" if direction in ("down", "up") else "y"

    # Walk top-down (parent first).  A node's contribution to the average is
    # weighted by depth from the root — closer to the root = more pull.
    # Find root: node with no parent.
    roots = [nid for nid, p in parent_of.items() if p is None]

    def _walk(nid, depth):
        if not children_of[nid]:
            return
        for c in children_of[nid]:
            n  = nodes_by_id[c]
            pn = nodes_by_id[nid]
            # Stronger pull near the root (small depth), tapering off.
            w = by / (1 + depth * 0.6)
            n[cross_axis] = n[cross_axis] * (1 - w) + pn[cross_axis] * w
            _walk(c, depth + 1)
    for r in roots:
        _walk(r, 0)

    out = _shallow_copy(layout)
    out["nodes"] = list(nodes_by_id.values())
    return out


def spread_horizontal(layout: Dict[str, Any], *, factor: float = 1.7) -> Dict[str, Any]:
    """Spread every node along the cross axis by *factor* about the centre.

    Default 1.7 — input chains separate visibly. The output canvas size
    isn't changed, so values >>1 will push nodes off the edge unless you
    also bump the canvas width/height.
    """
    direction = layout.get("direction", "down")
    cross_axis = "x" if direction in ("down", "up") else "y"
    span = layout["width"] if cross_axis == "x" else layout["height"]
    centre = span / 2.0

    out = _shallow_copy(layout)
    out["nodes"] = [
        {**n, cross_axis: centre + (n[cross_axis] - centre) * factor}
        for n in layout["nodes"]
    ]
    return out


def organic_layout(layout: Dict[str, Any], *,
                   branch_angle: float = 22.0,
                   length_scale: float = 38.0,
                   length_decay: float = 0.92,
                   angle_decay: float  = 0.96,
                   bend: float = 0.55) -> Dict[str, Any]:
    """Re-place every node by *growing* the tree from the root outward,
    branch by branch, like a real tree.

    The root sits at the diagram's "trunk" position. From every junction
    its two children are placed at ±``branch_angle`` from the parent's
    growing direction. As we move away from the root the lengths and
    angles decay so the tree fans gracefully outward without exploding.
    `fit_to_canvas` is automatically applied at the end so the result
    fills the canvas without cropping.

    Parameters
    ----------
    branch_angle : degrees the L/R children separate from their parent's
        growing direction at the root. Bigger angle = wider tree.
    length_scale : pixel length of the root branch.
    length_decay : factor by which child branches shrink (per generation).
    angle_decay  : factor by which the branch angle narrows per
        generation (so deep branches don't curl back on themselves).
    bend         : edge vertical_bias (0..1) for the renderer; default
        0.55 keeps branches gracefully curving rather than straight.
    """
    import math
    branch_angle = max(2.0, min(80.0, float(branch_angle)))
    length_scale = max(5.0, float(length_scale))
    length_decay = max(0.4, min(1.05, float(length_decay)))
    angle_decay  = max(0.5, min(1.05, float(angle_decay)))
    bend         = max(0.05, min(0.95, float(bend)))

    direction = layout.get("direction", "down")

    # Build adjacency.
    nodes_by_id = {n["id"]: dict(n) for n in layout["nodes"]}
    children_of: Dict[str, List[str]] = {nid: [] for nid in nodes_by_id}
    parent_of: Dict[str, Optional[str]] = {nid: None for nid in nodes_by_id}
    for e in layout["edges"]:
        children_of[e["to"]].append(e["from"])
        parent_of[e["from"]] = e["to"]
    root_id = next(nid for nid, p in parent_of.items() if p is None)

    # Root sits in the middle of the trail-end of the canvas.
    W = float(layout["width"])
    H = float(layout["height"])
    if direction == "down":
        root_pos    = (W / 2.0, H * 0.85)
        growing_dir = (0.0, -1.0)        # branches go upward
    elif direction == "up":
        root_pos    = (W / 2.0, H * 0.15)
        growing_dir = (0.0, 1.0)
    elif direction == "right":
        root_pos    = (W * 0.85, H / 2.0)
        growing_dir = (-1.0, 0.0)
    else:  # left
        root_pos    = (W * 0.15, H / 2.0)
        growing_dir = (1.0, 0.0)

    def _rot(v, deg):
        a = math.radians(deg)
        c, s = math.cos(a), math.sin(a)
        return (v[0] * c - v[1] * s, v[0] * s + v[1] * c)

    def _grow(nid, pos, gdir, depth):
        n = nodes_by_id[nid]
        n["x"], n["y"] = pos
        kids = children_of[nid]
        if not kids:
            return
        L = length_scale * (length_decay ** depth)
        a = branch_angle * (angle_decay ** depth)
        # children[0] = L (exp side), children[1] = R (ln side)
        # Place L on the left, R on the right of the growing direction.
        signs = (-1.0, +1.0)
        for child_id, sign in zip(kids, signs):
            child_dir = _rot(gdir, sign * a)
            child_pos = (pos[0] + child_dir[0] * L,
                         pos[1] + child_dir[1] * L)
            _grow(child_id, child_pos, child_dir, depth + 1)

    _grow(root_id, root_pos, growing_dir, 0)

    out = _shallow_copy(layout)
    out["nodes"] = list(nodes_by_id.values())
    # Apply gentle bend to every edge so branches read as curves not lines.
    out["edges"] = [{**e, "vertical_bias": bend} for e in layout["edges"]]
    # Auto-fit so a wide tree never crops.
    return fit_to_canvas(out, margin=40.0)


def fit_to_canvas(layout: Dict[str, Any], *,
                  margin: float = 30.0,
                  preserve_aspect: bool = True) -> Dict[str, Any]:
    """Scale & translate every node so the diagram fits cleanly inside the
    canvas with a uniform `margin` on all sides.

    Use this AS THE LAST STEP after any post-process that may have pushed
    nodes off the canvas (spread_horizontal, flowing_sideways with large
    amplitude, …). Also useful for tightening up a sparse diagram.

    Output layout has the same width/height as the input — only node
    coordinates change.
    """
    nodes = layout["nodes"]
    if not nodes:
        return _shallow_copy(layout)
    xs = [n["x"] for n in nodes]
    ys = [n["y"] for n in nodes]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    src_w = max(maxx - minx, 1e-6)
    src_h = max(maxy - miny, 1e-6)
    dst_w = layout["width"]  - 2 * margin
    dst_h = layout["height"] - 2 * margin
    sx = dst_w / src_w
    sy = dst_h / src_h
    if preserve_aspect:
        s = min(sx, sy)
        sx = sy = s
    # Centre the scaled bbox inside the canvas.
    cx = layout["width"]  / 2.0
    cy = layout["height"] / 2.0
    src_cx = (minx + maxx) / 2.0
    src_cy = (miny + maxy) / 2.0

    out = _shallow_copy(layout)
    out["nodes"] = [
        {**n,
         "x": cx + (n["x"] - src_cx) * sx,
         "y": cy + (n["y"] - src_cy) * sy}
        for n in nodes
    ]
    return out


def _shallow_copy(d: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(d)
    out["nodes"] = [dict(n) for n in d["nodes"]]
    out["edges"] = [dict(e) for e in d["edges"]]
    if "render_hints" in d:
        out["render_hints"] = dict(d["render_hints"])
    return out


# ── render: layout dict → SVG / PNG / PDF ───────────────────────────────────

def render_svg(layout: Dict[str, Any], *,
               background: Optional[str] = None,
               edge_width: float = 3.0,
               junction_radius: float = 4.0) -> str:
    """Render a (possibly post-processed) layout dict as an SVG string."""
    from eml_math.flow import (
        _curve_d, _esc, _label_offset, _text_anchor,
        _output_position, FIXED_LABEL_COLORS,
    )

    width  = int(layout["width"])
    height = int(layout["height"])
    direction = layout.get("direction", "down")
    out_label = layout.get("output_label", "Out")
    multi_output = not isinstance(out_label, str)
    output_labels = list(out_label) if multi_output else [out_label]

    hints = layout.get("render_hints", {})
    label_font_size  = int(hints.get("label_font_size", 18))
    output_font_size = int(hints.get("output_font_size", 22))
    primary_output_size = float(hints.get("primary_output_size",
                                          output_font_size * 2.0))
    omit_identity_labels = bool(hints.get("omit_identity_labels", True))

    nodes_by_id = {n["id"]: n for n in layout["nodes"]}
    children_of: Dict[str, List[str]] = {n["id"]: [] for n in layout["nodes"]}
    parent_of: Dict[str, Optional[str]] = {n["id"]: None for n in layout["nodes"]}
    for e in layout["edges"]:
        children_of[e["to"]].append(e["from"])
        parent_of[e["from"]] = e["to"]
    root_id = next(nid for nid, p in parent_of.items() if p is None)
    root = nodes_by_id[root_id]

    def _hex(rgb): return "#{:02X}{:02X}{:02X}".format(*[max(0, min(255, int(round(v)))) for v in rgb])

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" font-family="Inter, Helvetica, Arial, sans-serif">',
    ]
    if background:
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>')

    # Edges with their own vertical_bias
    for e in layout["edges"]:
        c = nodes_by_id[e["from"]]
        p = nodes_by_id[e["to"]]
        parts.append(_curve_d(c["x"], c["y"], p["x"], p["y"], direction,
                              _hex(e.get("color", c["color"])),
                              edge_width,
                              vertical_bias=e.get("vertical_bias", 0.5)))

    # Junctions (every internal node)
    for n in layout["nodes"]:
        if n.get("is_leaf"):
            continue
        parts.append(
            f'<circle cx="{n["x"]:.1f}" cy="{n["y"]:.1f}" r="{junction_radius}" '
            f'fill="{_hex(n["color"])}" stroke="#222" stroke-width="0.8"/>'
        )

    # Leaf labels
    for n in layout["nodes"]:
        if not n.get("is_leaf"):
            continue
        # If it's an L=0 / R=1 identity leaf and omit_identity_labels is on, skip.
        if omit_identity_labels and _is_identity_leaf(n, parent_of, children_of, nodes_by_id):
            continue
        text_col = _hex(FIXED_LABEL_COLORS.get(n["label"], n["color"]))
        if n.get("is_inline"):
            # Inline: label sits at branch endpoint
            if direction == "down":
                tx, ty = n["x"], n["y"] - 4
                anchor = "middle"
            elif direction == "up":
                tx, ty = n["x"], n["y"] + label_font_size
                anchor = "middle"
            elif direction == "right":
                tx, ty = n["x"] + 4, n["y"] + label_font_size * 0.35
                anchor = "start"
            else:
                tx, ty = n["x"] - 4, n["y"] + label_font_size * 0.35
                anchor = "end"
        else:
            tx, ty = _label_offset(n["x"], n["y"], direction, label_font_size, "lead")
            anchor = _text_anchor(direction, "lead")
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" fill="{text_col}" text-anchor="{anchor}" '
            f'font-weight="700" font-size="{label_font_size}">{_esc(n["label"])}</text>'
        )

    # Output label
    out_x, out_y = _output_position(direction, width, height, primary_output_size)
    if multi_output and len(output_labels) > 1:
        n = len(output_labels)
        spread = max(60.0, output_font_size * 1.2 * max(len(s) for s in output_labels))
        for i, lbl in enumerate(output_labels):
            offset = (i - (n - 1) / 2) * spread
            if direction in ("down", "up"):
                ox, oy = out_x + offset, out_y
            else:
                ox, oy = out_x, out_y + offset
            parts.append(_curve_d(root["x"], root["y"], ox, oy, direction,
                                   _hex(root["color"]), edge_width))
            parts.append(
                f'<text x="{ox:.1f}" y="{oy:.1f}" text-anchor="middle" '
                f'font-weight="700" font-size="{output_font_size}" fill="#222">{_esc(lbl)}</text>'
            )
    else:
        anchor = _text_anchor(direction, "trail")
        parts.append(
            f'<text x="{out_x:.1f}" y="{out_y:.1f}" text-anchor="{anchor}" '
            f'font-weight="700" font-size="{output_font_size}" fill="#222">{_esc(output_labels[0])}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def render_png(layout: Dict[str, Any], *, scale: float = 2.0,
               background: Optional[str] = None) -> bytes:
    """Render a layout dict as PNG bytes — uses the SVG path then rasterises
    via Pillow (no SVG parser needed; we draw straight onto a Pillow canvas
    with the layout's coordinates)."""
    from io import BytesIO
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        raise RuntimeError("render_png() requires Pillow.")
    from eml_math.flow import (
        _bezier_points, _output_position, FIXED_LABEL_COLORS,
    )

    width  = int(layout["width"])
    height = int(layout["height"])
    direction = layout.get("direction", "down")
    out_label = layout.get("output_label", "Out")
    multi_output = not isinstance(out_label, str)
    output_labels = list(out_label) if multi_output else [out_label]

    hints = layout.get("render_hints", {})
    label_font_size  = int(hints.get("label_font_size", 18))
    output_font_size = int(hints.get("output_font_size", 22))
    primary_output_size = float(hints.get("primary_output_size",
                                          output_font_size * 2.0))
    omit_identity_labels = bool(hints.get("omit_identity_labels", True))
    edge_width      = max(1, int(3 * scale))
    junction_radius = max(1, int(4 * scale))

    W, H = int(width * scale), int(height * scale)
    if background is None:
        img = Image.new("RGBA", (W, H), (255, 255, 255, 0))
    else:
        img = Image.new("RGB", (W, H), background)
    draw = ImageDraw.Draw(img)

    def _try_font(size: int):
        for name in ("arial.ttf", "DejaVuSans.ttf", "Helvetica.ttf"):
            try: return ImageFont.truetype(name, size)
            except (OSError, IOError): continue
        return ImageFont.load_default()
    font_label  = _try_font(int(label_font_size * scale))
    font_output = _try_font(int(output_font_size * scale))

    nodes_by_id = {n["id"]: n for n in layout["nodes"]}
    children_of: Dict[str, List[str]] = {n["id"]: [] for n in layout["nodes"]}
    parent_of: Dict[str, Optional[str]] = {n["id"]: None for n in layout["nodes"]}
    for e in layout["edges"]:
        children_of[e["to"]].append(e["from"])
        parent_of[e["from"]] = e["to"]
    root_id = next(nid for nid, p in parent_of.items() if p is None)
    root = nodes_by_id[root_id]

    def _curve_pts(c_x, c_y, p_x, p_y, vb=0.5):
        vb = max(0.05, min(0.95, vb))
        p0 = (c_x * scale, c_y * scale); p3 = (p_x * scale, p_y * scale)
        if direction in ("down", "up"):
            c_y_ctrl = p0[1] + (p3[1] - p0[1]) * vb
            p_y_ctrl = p3[1] - (p3[1] - p0[1]) * vb
            p1 = (p0[0], c_y_ctrl); p2 = (p3[0], p_y_ctrl)
        else:
            c_x_ctrl = p0[0] + (p3[0] - p0[0]) * vb
            p_x_ctrl = p3[0] - (p3[0] - p0[0]) * vb
            p1 = (c_x_ctrl, p0[1]); p2 = (p_x_ctrl, p3[1])
        return _bezier_points(p0, p1, p2, p3, samples=40)

    # Edges
    for e in layout["edges"]:
        c = nodes_by_id[e["from"]]
        p = nodes_by_id[e["to"]]
        col = tuple(int(round(v)) for v in e.get("color", c["color"]))
        pts = _curve_pts(c["x"], c["y"], p["x"], p["y"], e.get("vertical_bias", 0.5))
        draw.line(pts, fill=col, width=edge_width, joint="curve")

    # Junctions
    for n in layout["nodes"]:
        if n.get("is_leaf"):
            continue
        col = tuple(int(round(v)) for v in n["color"])
        cx, cy = n["x"] * scale, n["y"] * scale
        draw.ellipse([cx - junction_radius, cy - junction_radius,
                      cx + junction_radius, cy + junction_radius],
                     fill=col, outline=(34, 34, 34), width=max(1, edge_width // 3))

    # Leaf labels
    pad = 12 * scale
    for n in layout["nodes"]:
        if not n.get("is_leaf"):
            continue
        if omit_identity_labels and _is_identity_leaf(n, parent_of, children_of, nodes_by_id):
            continue
        text_rgb = tuple(int(round(v)) for v in
                         FIXED_LABEL_COLORS.get(n["label"], n["color"]))
        bbox = draw.textbbox((0, 0), n["label"], font=font_label)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        lx, ly = n["x"] * scale, n["y"] * scale
        if n.get("is_inline"):
            if direction == "down":
                tx = lx - tw / 2; ty = ly - th - 2 * scale
            elif direction == "up":
                tx = lx - tw / 2; ty = ly + 2 * scale
            elif direction == "right":
                tx = lx + 4 * scale; ty = ly - th / 2
            else:
                tx = lx - tw - 4 * scale; ty = ly - th / 2
        else:
            if direction == "down":
                tx = lx - tw / 2; ty = ly - pad - th
            elif direction == "up":
                tx = lx - tw / 2; ty = ly + pad
            elif direction == "right":
                tx = lx - pad - tw; ty = ly - th / 2
            else:
                tx = lx + pad; ty = ly - th / 2
        draw.text((tx, ty), n["label"], fill=text_rgb, font=font_label)

    # Output label
    ox, oy = _output_position(direction, width, height, primary_output_size)
    ox *= scale; oy *= scale
    if multi_output and len(output_labels) > 1:
        n = len(output_labels)
        spread = max(60.0, output_font_size * 1.2 *
                     max(len(s) for s in output_labels)) * scale
        for i, lbl in enumerate(output_labels):
            offset = (i - (n - 1) / 2) * spread
            if direction in ("down", "up"):
                lx, ly = ox + offset, oy
            else:
                lx, ly = ox, oy + offset
            pts = _curve_pts(root["x"], root["y"], lx / scale, ly / scale, 0.5)
            draw.line(pts, fill=tuple(int(round(v)) for v in root["color"]),
                      width=edge_width, joint="curve")
            bbox = draw.textbbox((0, 0), lbl, font=font_output)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            draw.text((lx - tw / 2, ly - th / 2), lbl,
                      fill=(34, 34, 34), font=font_output)
    else:
        bbox = draw.textbbox((0, 0), output_labels[0], font=font_output)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        if direction in ("down", "up"):
            draw.text((ox - tw / 2, oy - th / 2), output_labels[0],
                      fill=(34, 34, 34), font=font_output)
        elif direction == "right":
            draw.text((ox, oy - th / 2), output_labels[0],
                      fill=(34, 34, 34), font=font_output)
        else:
            draw.text((ox - tw, oy - th / 2), output_labels[0],
                      fill=(34, 34, 34), font=font_output)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def render_pdf(layout: Dict[str, Any], *, scale: float = 2.0,
               background: str = "white") -> bytes:
    """Render layout as a single-page raster PDF (background flattened to
    `background` because PDF doesn't carry alpha for our flat case)."""
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("render_pdf() requires Pillow.")
    png = render_png(layout, scale=scale, background=background)
    img = Image.open(BytesIO(png)).convert("RGB")
    buf = BytesIO()
    img.save(buf, format="PDF", resolution=int(72 * scale))
    return buf.getvalue()


# ── Helpers ─────────────────────────────────────────────────────────────────

def _is_identity_leaf(n, parent_of, children_of, nodes_by_id) -> bool:
    """True if leaf is L=0 (children[0]=='0') or R=1 (children[1]=='1')."""
    pid = parent_of.get(n["id"])
    if pid is None:
        return False
    sibs = children_of[pid]
    if len(sibs) != 2:
        return False
    L_id, R_id = sibs[0], sibs[1]
    if n["id"] == L_id and n["label"] == "0":
        return True
    if n["id"] == R_id and n["label"] == "1":
        return True
    return False
