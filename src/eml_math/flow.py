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
    "DIRECTIONS",
    "flow_svg",
    "flow_png",
    "flow_pdf",
    "flow_html",
]

# Supported flow directions.
#
# `down`  : inputs at top, output at bottom        (default; ln-side child to right)
# `up`    : inputs at bottom, output at top         (mirrored vertical)
# `right` : inputs on left, output on right         (ln-side child below — fits log-arithmetic visually)
# `left`  : inputs on right, output on left         (mirrored horizontal)
DIRECTIONS = ("down", "up", "right", "left")


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

# Sentinel leaves get a neutral muted colour so they don't compete with
# real-input colours visually.
SENTINEL_COLOR: Tuple[int, int, int] = (160, 160, 160)
# Only the bottom (exp(⊥)=0) is treated as a sentinel — it's a pure-EML
# internal artefact with no semantic meaning to a human reader. The number
# 1 is a real numeric input (it just happens to be ln(1)=0 in EML), so it
# gets a normal palette colour and participates in merging just like any
# other constant.
SENTINEL_LABELS = frozenset({"⊥"})


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


def _assign_logical(node: "EMLTreeNode", leaf_cross: dict) -> None:
    """Compute logical positions on every node:
        _fdepth ∈ [0, max_height]   — 0 at leaves, increases toward root
        _fcross ∈ [0, 1]            — leaves spread evenly along cross axis
    These are direction-agnostic; :func:`_to_screen` projects them onto x/y.
    """
    h = _height(node)
    node._fdepth = h
    if not node.children:
        node._fcross = leaf_cross[id(node)]
        return
    for c in node.children:
        _assign_logical(c, leaf_cross)
    node._fcross = sum(c._fcross for c in node.children) / len(node.children)


def _to_screen(
    node: "EMLTreeNode",
    *,
    direction: str,
    width: float,
    height: float,
    margin_lead: float,    # space before the leaves (top for "down", left for "right", …)
    margin_trail: float,   # space after the root (bottom for "down", right for "right", …)
    margin_cross: float,   # space at each end of the cross axis
) -> None:
    """Project (depth, cross) → (x, y) for the requested direction."""
    h = max(node._fdepth, 1)

    def _t(n):
        # _fdepth = tree-height(node): 0 at leaves, max at root.
        # progress p ∈ [0, 1]: 0 at leaves (lead end), 1 at root (trail end)
        p = n._fdepth / h
        c = n._fcross
        if direction == "down":
            primary_span = height - margin_lead - margin_trail
            cross_span   = width - 2 * margin_cross
            n._fy = margin_lead + p * primary_span
            n._fx = margin_cross + c * cross_span
        elif direction == "up":
            primary_span = height - margin_lead - margin_trail
            cross_span   = width - 2 * margin_cross
            n._fy = (height - margin_lead) - p * primary_span
            n._fx = margin_cross + c * cross_span
        elif direction == "right":
            primary_span = width - margin_lead - margin_trail
            cross_span   = height - 2 * margin_cross
            n._fx = margin_lead + p * primary_span
            n._fy = margin_cross + c * cross_span
        elif direction == "left":
            primary_span = width - margin_lead - margin_trail
            cross_span   = height - 2 * margin_cross
            n._fx = (width - margin_lead) - p * primary_span
            n._fy = margin_cross + c * cross_span
        else:
            raise ValueError(
                f"unknown direction {direction!r}; expected one of {DIRECTIONS}"
            )
        for ch in n.children:
            _t(ch)

    _t(node)


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

def _to_pure_eml_tree(node: "EMLTreeNode") -> "EMLTreeNode":
    """Pre-expand any compact-mode internal node (``mul``, ``pow``, …) into
    the binary-eml primitive form so every internal node of the tree is
    genuinely ``eml(L, R) = exp(L) − ln(R)`` — no operator names hidden
    behind an unlabelled junction."""
    from eml_math.tree import _to_pure_eml
    return _to_pure_eml(node)


def _expand_symbols_in_tree(node: "EMLTreeNode") -> "EMLTreeNode":
    """Recursively replace any leaf whose label is a known named symbol with
    its EML construction tree (from :mod:`eml_math.symbols`)."""
    from eml_math.symbols import lookup
    if not node.children:
        sym = lookup(node.label)
        if sym is not None and sym.tree is not None:
            return sym.tree
        return node
    new_children = [_expand_symbols_in_tree(c) for c in node.children]
    if all(c is orig for c, orig in zip(new_children, node.children)):
        return node
    from eml_math.tree import EMLTreeNode
    return EMLTreeNode(
        label=node.label, kind=node.kind, children=new_children,
        eml_form=node.eml_form,
    )


def _layout(
    node: "EMLTreeNode",
    *,
    width: int,
    height: int,
    margin_lead: float,
    margin_trail: float,
    margin_cross: float,
    palette: Sequence[Tuple[int, int, int]],
    direction: str,
    expand_symbols: bool,
) -> Tuple["EMLTreeNode", List["EMLTreeNode"]]:
    """Returns (binarised_root, leaves_in_cross-axis-order)."""
    if expand_symbols:
        node = _expand_symbols_in_tree(node)
    # Always binarise — guarantees a true binary tree (no unary passthroughs,
    # no n-into-1 merges).
    node = _binarize(node)

    leaves = _collect_leaves(node)
    n = len(leaves)
    if n == 1:
        leaf_cross = {id(leaves[0]): 0.5}
    else:
        leaf_cross = {id(l): i / (n - 1) for i, l in enumerate(leaves)}

    _assign_logical(node, leaf_cross)
    _to_screen(
        node, direction=direction,
        width=width, height=height,
        margin_lead=margin_lead, margin_trail=margin_trail,
        margin_cross=margin_cross,
    )

    # Colour assignment: equal labels share a colour. Sentinel leaves
    # (⊥, 1) get a single neutral grey so they fade into the background
    # next to the real-input colours.
    label_to_color: dict[str, Tuple[int, int, int]] = {}
    next_palette_idx = 0
    for leaf in leaves:
        if leaf.label in SENTINEL_LABELS:
            label_to_color.setdefault(leaf.label, SENTINEL_COLOR)
            continue
        if leaf.label not in label_to_color:
            label_to_color[leaf.label] = tuple(palette[next_palette_idx % len(palette)])
            next_palette_idx += 1
    leaf_color = {id(l): label_to_color[l.label] for l in leaves}
    _assign_colors(node, leaf_color)
    return node, leaves


# ── SVG renderer ─────────────────────────────────────────────────────────────

def flow_svg(
    node: "EMLTreeNode",
    *,
    width: int = 800,
    height: int = 600,
    direction: str = "down",
    palette: Optional[Sequence[Tuple[int, int, int]]] = None,
    show_output_label: bool = True,
    output_label = "Out",                    # str | Sequence[str] for multi-output
    expand_symbols: bool = False,
    merge_inputs: bool = False,              # deduplicate identical inputs into one
    show_sentinel_labels: bool = False,      # show ⊥ / 1 sentinel labels on leaves
    label_font_size: int = 18,
    output_font_size: int = 22,
    edge_width: float = 3.0,
    junction_radius: float = 4.0,
    background: Optional[str] = None,
) -> str:
    """Render *node* as a flow-diagram SVG string.

    Parameters
    ----------
    node : EMLTreeNode
        Root of the tree to render.
    width, height : int
        Canvas dimensions in pixels.
    direction : {"down", "up", "right", "left"}
        Flow direction. ``"down"`` = inputs at top, output at bottom (default).
        ``"right"`` = inputs on left, output on right (ln-side child below).
    palette : sequence of (r, g, b)
        Per-input colours, recycled if there are more leaves than entries.
    show_output_label : bool
    output_label : str | sequence of str
        Output label(s) drawn at the *trailing* end of the diagram. Pass a
        sequence (e.g. ``["x_+", "x_-"]``) for multi-valued formulas — both
        labels are drawn near the output and joined back to the root.
    expand_symbols : bool
        If True, leaves whose label matches a known named symbol from
        :mod:`eml_math.symbols` are replaced in-line with the symbol's EML
        construction tree (e.g. ``e`` becomes ``eml(1, 1)``). Symbols with
        no closed form (π, τ, γ) remain as labelled leaves.
    label_font_size, output_font_size : int
    edge_width : float
    junction_radius : float
    background : str | None
        Optional solid background fill colour.
    """
    if palette is None:
        palette = DEFAULT_PALETTE
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    multi_output = not isinstance(output_label, str)
    output_labels = list(output_label) if multi_output else [output_label]

    # Figure out leaf-label text length for cross-axis margin sizing.
    preview_root = _binarize(_expand_symbols_in_tree(node) if expand_symbols else node)
    leaves_preview = _collect_leaves(preview_root)
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    half_label_w   = 0.5 * 0.6 * label_font_size * max_label_len
    cross_margin   = max(40.0, half_label_w + 12.0)

    # Lead margin = where the leaves sit; trail margin = where the output sits.
    primary_label_size = float(label_font_size) * 2.2
    primary_output_size = float(output_font_size) * 2.0 if show_output_label else 30.0
    if multi_output:
        primary_output_size *= 1.6   # extra room for multiple labels

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_lead=primary_label_size,
        margin_trail=primary_output_size,
        margin_cross=cross_margin,
        palette=palette, direction=direction,
        expand_symbols=expand_symbols,
    )

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width}" height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Inter, Helvetica, Arial, sans-serif">',
    ]
    if background:
        parts.append(f'<rect x="0" y="0" width="{width}" height="{height}" fill="{background}"/>')

    # ── edges between every parent/child ────────────────────────────────────
    _emit_flow_edges(node, parts, edge_width, direction)

    # ── if binarised tree is a bare leaf, draw a direct edge to the output ──
    out_x, out_y = _output_position(direction, width, height, primary_output_size)
    if not node.children:
        col = _rgb_hex(node._fcolor)
        parts.append(_curve_d(node._fx, node._fy, out_x, out_y, direction,
                              col, edge_width))

    # ── junction dots ───────────────────────────────────────────────────────
    _emit_junctions(node, parts, junction_radius)

    # ── input labels at the LEAD end ────────────────────────────────────────
    if merge_inputs:
        # Group leaves by label; render ONE label per unique input and draw
        # 1-to-N redirector curves from the merged-input position to each
        # usage point (the leaf's screen position in the tree).
        unique_labels: List[str] = []
        for leaf in leaves:
            if leaf.label not in unique_labels:
                unique_labels.append(leaf.label)
        n_uniq = len(unique_labels)

        merged_pos: dict[str, Tuple[float, float]] = {}
        # Place the merged inputs evenly along the cross axis at the lead end.
        for i, lbl in enumerate(unique_labels):
            cross_t = 0.5 if n_uniq == 1 else i / (n_uniq - 1)
            if direction in ("down", "up"):
                mx = cross_margin + cross_t * (width - 2 * cross_margin)
                my = primary_label_size if direction == "down" else height - primary_label_size
            else:
                my = cross_margin + cross_t * (height - 2 * cross_margin)
                mx = primary_label_size if direction == "right" else width - primary_label_size
            merged_pos[lbl] = (mx, my)

        # Redirector curves: from each merged-input position to every leaf's tree position.
        for leaf in leaves:
            mx, my = merged_pos[leaf.label]
            col = _rgb_hex(leaf._fcolor)
            parts.append(_curve_d(mx, my, leaf._fx, leaf._fy, direction, col, edge_width))

        # Render one label per unique input (skip sentinels unless asked)
        for lbl in unique_labels:
            is_sentinel = lbl in SENTINEL_LABELS
            if is_sentinel and not show_sentinel_labels:
                continue
            mx, my = merged_pos[lbl]
            col = _rgb_hex(SENTINEL_COLOR if is_sentinel
                           else next(iter(c for l, c in _label_colors_iter(leaves) if l == lbl)))
            lx, ly = _label_offset(mx, my, direction, label_font_size, "lead")
            anchor = _text_anchor(direction, "lead")
            fs     = int(label_font_size * 0.7) if is_sentinel else label_font_size
            weight = "400" if is_sentinel else "700"
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{col}" text-anchor="{anchor}" '
                f'font-weight="{weight}" font-size="{fs}">{_esc(lbl)}</text>'
            )
    else:
        for leaf in leaves:
            is_sentinel = leaf.label in SENTINEL_LABELS
            if is_sentinel and not show_sentinel_labels:
                continue   # tiny grey dot is enough — see _emit_junctions
            col = _rgb_hex(leaf._fcolor)
            lx, ly = _label_offset(leaf._fx, leaf._fy, direction, label_font_size, "lead")
            anchor = _text_anchor(direction, "lead")
            fs     = int(label_font_size * 0.7) if is_sentinel else label_font_size
            weight = "400" if is_sentinel else "700"
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{col}" text-anchor="{anchor}" '
                f'font-weight="{weight}" font-size="{fs}">{_esc(leaf.label)}</text>'
            )

    # ── output label(s) at the TRAIL end ────────────────────────────────────
    if show_output_label:
        # For multi-output: spread labels on the cross axis around the output point.
        if multi_output and len(output_labels) > 1:
            n = len(output_labels)
            spread = max(60.0, output_font_size * 1.2 * max(len(s) for s in output_labels))
            for i, lbl in enumerate(output_labels):
                offset = (i - (n - 1) / 2) * spread
                ox, oy = _multi_output_position(out_x, out_y, direction, offset)
                # connecting line from the root junction to this output position
                parts.append(_curve_d(node._fx, node._fy, ox, oy, direction,
                                      _rgb_hex(node._fcolor), edge_width))
                parts.append(
                    f'<text x="{ox:.1f}" y="{oy:.1f}" text-anchor="middle" '
                    f'font-weight="700" font-size="{output_font_size}" '
                    f'fill="#222">{_esc(lbl)}</text>'
                )
            # small ± indicator at the root
            indicator_x, indicator_y = (node._fx, node._fy + 16) if direction == "down" \
                else (node._fx + 16, node._fy) if direction == "right" \
                else (node._fx, node._fy - 16) if direction == "up" \
                else (node._fx - 16, node._fy)
            parts.append(
                f'<text x="{indicator_x:.1f}" y="{indicator_y:.1f}" text-anchor="middle" '
                f'font-size="{output_font_size}" fill="#666" font-style="italic">±</text>'
            )
        else:
            anchor = _text_anchor(direction, "trail")
            parts.append(
                f'<text x="{out_x:.1f}" y="{out_y:.1f}" text-anchor="{anchor}" '
                f'font-weight="700" font-size="{output_font_size}" '
                f'fill="#222">{_esc(output_labels[0])}</text>'
            )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Direction-aware helpers ──────────────────────────────────────────────────

def _label_offset(x: float, y: float, direction: str, font_size: int, end: str
                  ) -> Tuple[float, float]:
    """Return the (x, y) for the label of a leaf or the output, offset away
    from the diagram edge.  end='lead' = above leaves; end='trail' = below root."""
    pad = 12.0
    if direction == "down":
        return (x, y - pad if end == "lead" else y + pad + font_size)
    if direction == "up":
        return (x, y + pad + font_size if end == "lead" else y - pad)
    if direction == "right":
        return (x - pad if end == "lead" else x + pad, y + font_size * 0.35)
    if direction == "left":
        return (x + pad if end == "lead" else x - pad, y + font_size * 0.35)
    return (x, y)


def _text_anchor(direction: str, end: str) -> str:
    if direction in ("down", "up"):
        return "middle"
    if direction == "right":
        return "end" if end == "lead" else "start"
    # left
    return "start" if end == "lead" else "end"


def _output_position(direction: str, width: float, height: float,
                     trail_margin: float) -> Tuple[float, float]:
    if direction == "down":   return (width / 2, height - trail_margin * 0.35)
    if direction == "up":     return (width / 2, trail_margin * 0.35)
    if direction == "right":  return (width - trail_margin * 0.35, height / 2)
    if direction == "left":   return (trail_margin * 0.35, height / 2)
    return (width / 2, height - trail_margin * 0.35)


def _multi_output_position(out_x: float, out_y: float, direction: str,
                           offset: float) -> Tuple[float, float]:
    """Spread multiple output labels across the cross axis."""
    if direction in ("down", "up"):
        return (out_x + offset, out_y)
    return (out_x, out_y + offset)


def _label_colors_iter(leaves):
    """Yield (label, color) pairs in iteration order (after _layout has set _fcolor)."""
    seen = set()
    for l in leaves:
        if l.label in seen:
            continue
        seen.add(l.label)
        yield l.label, l._fcolor


def _curve_d(cx: float, cy: float, px: float, py: float, direction: str,
             stroke: str, edge_width: float) -> str:
    """Cubic-Bezier path string from (cx, cy) to (px, py).  Control points are
    offset along the *primary* axis so the curve flows naturally."""
    if direction in ("down", "up"):
        m = (cy + py) / 2
        d = f"M{cx:.1f},{cy:.1f} C{cx:.1f},{m:.1f} {px:.1f},{m:.1f} {px:.1f},{py:.1f}"
    else:  # right or left
        m = (cx + px) / 2
        d = f"M{cx:.1f},{cy:.1f} C{m:.1f},{cy:.1f} {m:.1f},{py:.1f} {px:.1f},{py:.1f}"
    return (f'<path d="{d}" stroke="{stroke}" stroke-width="{edge_width}" '
            f'fill="none" stroke-linecap="round"/>')


def _emit_flow_edges(
    node: "EMLTreeNode",
    out: List[str],
    edge_width: float,
    direction: str = "down",
) -> None:
    if not node.children:
        return
    for c in node.children:
        col = _rgb_hex(c._fcolor)
        out.append(_curve_d(c._fx, c._fy, node._fx, node._fy, direction,
                            col, edge_width))
        _emit_flow_edges(c, out, edge_width, direction)


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
    """Render the flow diagram to a single-page PDF (raster, via Pillow)."""
    from io import BytesIO
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("flow_pdf() requires Pillow.")
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
    direction: str = "down",
    show_output_label: bool = True,
    output_label = "Out",
    expand_symbols: bool = False,
    merge_inputs: bool = False,
    show_sentinel_labels: bool = False,
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
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")

    multi_output = not isinstance(output_label, str)
    output_labels = list(output_label) if multi_output else [output_label]

    # All sizes scaled up for hi-DPI rendering
    W = int(width * scale)
    H = int(height * scale)
    fs_label  = int(label_font_size * scale)
    fs_output = int(output_font_size * scale)
    ew        = max(1, int(edge_width * scale))
    jr        = max(1, int(junction_radius * scale))

    primary_label_size  = float(label_font_size) * 2.2
    primary_output_size = float(output_font_size) * 2.0 if show_output_label else 30.0
    if multi_output:
        primary_output_size *= 1.6

    preview_root = _binarize(_expand_symbols_in_tree(node) if expand_symbols else node)
    leaves_preview = _collect_leaves(preview_root)
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    half_label_w   = 0.5 * 0.6 * label_font_size * max_label_len
    cross_margin   = max(40.0, half_label_w + 12.0)

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_lead=primary_label_size,
        margin_trail=primary_output_size,
        margin_cross=cross_margin,
        palette=palette, direction=direction,
        expand_symbols=expand_symbols,
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

    def _curve_pts_for_pil(c_x, c_y, p_x, p_y):
        p0 = (c_x * scale, c_y * scale)
        p3 = (p_x * scale, p_y * scale)
        if direction in ("down", "up"):
            m = (p0[1] + p3[1]) / 2
            p1 = (p0[0], m); p2 = (p3[0], m)
        else:  # right or left
            m = (p0[0] + p3[0]) / 2
            p1 = (m, p0[1]); p2 = (m, p3[1])
        return _bezier_points(p0, p1, p2, p3, samples=40)

    # Edges
    def _draw_edges(n):
        if not n.children:
            return
        for c in n.children:
            pts = _curve_pts_for_pil(c._fx, c._fy, n._fx, n._fy)
            col = tuple(int(round(v)) for v in c._fcolor)
            draw.line(pts, fill=col, width=ew, joint="curve")
            _draw_edges(c)
    _draw_edges(node)

    # Bare-leaf root: synthetic edge to the output position.
    out_x_logical, out_y_logical = _output_position(direction, width, height,
                                                     primary_output_size)
    if not node.children:
        pts = _curve_pts_for_pil(node._fx, node._fy, out_x_logical, out_y_logical)
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

    # Leaf labels — direction-aware placement.  Skip pure-EML sentinel
    # labels by default so they don't clutter the diagram.
    pad = 12 * scale

    def _draw_leaf_label(label_text, color_rgb, lx_screen, ly_screen):
        bbox = draw.textbbox((0, 0), label_text, font=font_label)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        if direction == "down":
            tx = lx_screen - tw / 2; ty = ly_screen - pad - th
        elif direction == "up":
            tx = lx_screen - tw / 2; ty = ly_screen + pad
        elif direction == "right":
            tx = lx_screen - pad - tw; ty = ly_screen - th / 2
        else:  # left
            tx = lx_screen + pad; ty = ly_screen - th / 2
        draw.text((tx, ty), label_text, fill=color_rgb, font=font_label)

    if merge_inputs:
        # One input position per unique leaf label; redirector curves from
        # the merged position to every usage point in the tree.
        unique_labels = []
        for l in leaves:
            if l.label not in unique_labels:
                unique_labels.append(l.label)
        n_uniq = len(unique_labels)

        # Compute merged-input screen positions (in "logical" units, scaled below).
        merged_pos = {}
        for i, lbl in enumerate(unique_labels):
            cross_t = 0.5 if n_uniq == 1 else i / (n_uniq - 1)
            if direction in ("down", "up"):
                mx = cross_margin + cross_t * (width - 2 * cross_margin)
                my = primary_label_size if direction == "down" else height - primary_label_size
            else:
                my = cross_margin + cross_t * (height - 2 * cross_margin)
                mx = primary_label_size if direction == "right" else width - primary_label_size
            merged_pos[lbl] = (mx, my)

        # Redirector curves
        for leaf in leaves:
            mx, my = merged_pos[leaf.label]
            col = tuple(int(round(v)) for v in leaf._fcolor)
            pts = _curve_pts_for_pil(mx, my, leaf._fx, leaf._fy)
            draw.line(pts, fill=col, width=ew, joint="curve")

        # One label per unique input
        label_color = {}
        for lbl, c in _label_colors_iter(leaves):
            label_color[lbl] = tuple(int(round(v)) for v in c)
        for lbl in unique_labels:
            is_sentinel = lbl in SENTINEL_LABELS
            if is_sentinel and not show_sentinel_labels:
                continue
            mx, my = merged_pos[lbl]
            _draw_leaf_label(lbl, label_color[lbl], mx * scale, my * scale)
    else:
        for leaf in leaves:
            is_sentinel = leaf.label in SENTINEL_LABELS
            if is_sentinel and not show_sentinel_labels:
                continue
            col = tuple(int(round(v)) for v in leaf._fcolor)
            _draw_leaf_label(leaf.label, col, leaf._fx * scale, leaf._fy * scale)

    # Output label(s) — direction-aware placement
    if show_output_label:
        out_x = out_x_logical * scale
        out_y = out_y_logical * scale

        def _draw_output_text(label_text, ox, oy):
            bbox = draw.textbbox((0, 0), label_text, font=font_output)
            tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
            if direction in ("down", "up"):
                draw.text((ox - tw / 2, oy - th / 2), label_text,
                          fill=(34, 34, 34), font=font_output)
            elif direction == "right":
                draw.text((ox, oy - th / 2), label_text,
                          fill=(34, 34, 34), font=font_output)
            else:  # left
                draw.text((ox - tw, oy - th / 2), label_text,
                          fill=(34, 34, 34), font=font_output)

        if multi_output and len(output_labels) > 1:
            n = len(output_labels)
            spread = max(60.0, output_font_size * 1.2 *
                         max(len(s) for s in output_labels)) * scale
            for i, lbl in enumerate(output_labels):
                offset = (i - (n - 1) / 2) * spread
                if direction in ("down", "up"):
                    ox, oy = out_x + offset, out_y
                else:
                    ox, oy = out_x, out_y + offset
                # connecting curve from root to this output position
                pts = _curve_pts_for_pil(node._fx, node._fy,
                                         ox / scale, oy / scale)
                draw.line(pts, fill=tuple(int(round(v)) for v in node._fcolor),
                          width=ew, joint="curve")
                _draw_output_text(lbl, ox, oy)
        else:
            _draw_output_text(output_labels[0], out_x, out_y)

    buf = BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()
