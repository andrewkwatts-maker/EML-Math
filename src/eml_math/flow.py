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

# Every leaf in the EML graph is a real input — there are no sentinels.
# Variables (a, M, …) and arbitrary numeric constants (2, 4, 0.5, …) get
# palette colours via the equal-labels-same-colour rule.
#
# The two special constants 0 and 1 are *always* drawn as a short stub at
# the leaf's branch endpoint with the number written right there — they
# don't go all the way up to the top input row. Their colours are fixed:
#   0 → pure black
#   1 → pure white  (the branch is effectively invisible on a white
#                    canvas, indicating ln(1)=0 contributes nothing)
SENTINEL_COLOR: Tuple[int, int, int] = (160, 160, 160)   # legacy export, unused
SENTINEL_LABELS: frozenset = frozenset()
FIXED_COLORS: dict = {
    "0": (140, 140, 140),   # default: medium grey (override per call to taste)
    "1": (200, 200, 200),   # default: light grey
}
# Label-text colour overrides for the FIXED_COLORS entries — used when the
# branch colour is too pale to read as text.
FIXED_LABEL_COLORS: dict = {
    "0": (60,  60,  60),
    "1": (60,  60,  60),
}


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
    *,
    bypass_identity: bool = True,
) -> Tuple[float, float, float]:
    """Recursively colour every node by averaging its children's colours.

    When ``bypass_identity=True`` (the default), an `eml(L, R)` junction
    excludes from the average:
      * the L child if its label is ``"0"`` — because ``exp(0) = 0`` in
        our pure-eml convention so the L leg contributes nothing to the
        value, and shouldn't muddy the colour either;
      * the R child if its label is ``"1"`` — because ``ln(1) = 0`` so
        the R leg likewise vanishes.
    The excluded child still gets coloured (its short-stub branch is
    drawn) but doesn't blend up the tree.
    """
    if not node.children:
        rgb = leaf_color[id(node)]
        node._fcolor = rgb
        return rgb

    child_rgbs = [_assign_colors(c, leaf_color, bypass_identity=bypass_identity)
                  for c in node.children]

    contributing = list(child_rgbs)
    if bypass_identity and len(node.children) == 2:
        L, R = node.children
        # Drop L's colour if L is the bottom-sentinel '0' (exp leg vanishes).
        # Drop R's colour if R is '1' (ln leg vanishes).
        keep = []
        for child, rgb, is_L in (
            (L, child_rgbs[0], True),
            (R, child_rgbs[1], False),
        ):
            label = (child.label or "")
            if is_L and label == "0":
                continue
            if (not is_L) and label == "1":
                continue
            keep.append(rgb)
        if keep:
            contributing = keep

    avg = (
        sum(c[0] for c in contributing) / len(contributing),
        sum(c[1] for c in contributing) / len(contributing),
        sum(c[2] for c in contributing) / len(contributing),
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


def _stub_inline_leaves(
    node: "EMLTreeNode",
    *,
    direction: str,
    fixed_labels,           # iterable of labels that should become stubs
    inline_constants: bool,
    label_font_size: int,
) -> None:
    """Reposition inline-constant leaves so they sit *next to* their parent
    junction as a short stub instead of extending all the way up to the
    leaf row. Operates after :func:`_to_screen` has set _fx/_fy.

    Direction → which axis the stub sits on:
      down/up    : stub is offset along the cross (x) axis from the parent
      right/left : stub is offset along the cross (y) axis
    """
    stub_set = set(fixed_labels)
    if not stub_set and not inline_constants:
        return

    def _is_inline(leaf):
        if leaf.label in stub_set:
            return True
        if inline_constants and _is_numeric_label(leaf.label):
            return True
        return False

    # Distance from parent to stub leaf, in the same screen-units _to_screen used.
    stub_offset = max(28.0, float(label_font_size) * 1.6)

    def _walk(parent):
        if not parent.children:
            return
        n = len(parent.children)
        for i, child in enumerate(parent.children):
            if not child.children and _is_inline(child):
                # Cross-axis offset: spread children evenly around parent.
                # For binary parent: L-child gets -ve offset, R-child +ve.
                t = (i / (n - 1) - 0.5) if n > 1 else 0.0
                cross_off = t * 2 * stub_offset    # left-of-parent for L, right for R
                primary_off = stub_offset * 0.6   # short way back along the primary axis
                if direction == "down":
                    child._fx = parent._fx + cross_off
                    child._fy = parent._fy - primary_off
                elif direction == "up":
                    child._fx = parent._fx + cross_off
                    child._fy = parent._fy + primary_off
                elif direction == "right":
                    child._fx = parent._fx - primary_off
                    child._fy = parent._fy + cross_off
                else:  # left
                    child._fx = parent._fx + primary_off
                    child._fy = parent._fy + cross_off
            _walk(child)

    _walk(node)


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
    fixed_colors: Optional[dict] = None,
    bypass_identity_blend: bool = True,
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

    # Colour assignment: equal labels share a colour. Fixed-colour labels
    # (e.g. 0 and 1) are pinned by the caller-supplied (or default) map.
    fc = FIXED_COLORS if fixed_colors is None else fixed_colors
    label_to_color: dict[str, Tuple[int, int, int]] = {}
    next_palette_idx = 0
    for leaf in leaves:
        if leaf.label in fc:
            label_to_color.setdefault(leaf.label, tuple(fc[leaf.label]))
            continue
        if leaf.label not in label_to_color:
            label_to_color[leaf.label] = tuple(palette[next_palette_idx % len(palette)])
            next_palette_idx += 1
    leaf_color = {id(l): label_to_color[l.label] for l in leaves}
    _assign_colors(node, leaf_color, bypass_identity=bypass_identity_blend)
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
    inline_constants: bool = False,          # numeric leaves render at their branch endpoint
    fixed_colors: Optional[dict] = None,     # override colours for special labels (0, 1, …)
    fixed_label_colors: Optional[dict] = None,
    bypass_identity_blend: bool = True,      # skip L=0 and R=1 from the colour blend
    omit_identity_labels: bool = True,       # don't print the number for L=0 / R=1 — the grey stub implies it
    show_sentinel_labels: bool = False,      # legacy, no-op (every leaf is a real input)
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
    # When merging, push the leaves further into the canvas so the merged
    # input labels at the very lead-end have vertical room to slerp down
    # to each usage point in the tree.
    if merge_inputs:
        # Big multiplier so the redirector curves from the merged-input
        # row down to each leaf's tree position have real vertical room
        # to slerp — without it they end up looking horizontal.
        primary_label_size *= 5.5

    fc = FIXED_COLORS if fixed_colors is None else fixed_colors
    flc = FIXED_LABEL_COLORS if fixed_label_colors is None else fixed_label_colors

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_lead=primary_label_size,
        margin_trail=primary_output_size,
        margin_cross=cross_margin,
        palette=palette, direction=direction,
        expand_symbols=expand_symbols,
        fixed_colors=fc,
        bypass_identity_blend=bypass_identity_blend,
    )

    # Reposition 0/1 (and other inline) leaves to sit RIGHT NEXT TO their
    # parent junction so they don't draw long branches all the way up.
    _stub_inline_leaves(node, direction=direction, fixed_labels=fc.keys(),
                         inline_constants=inline_constants,
                         label_font_size=label_font_size)

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

    # ── inline constants — render at the branch endpoint, not at the lead end ──
    # Two cases trigger inline rendering:
    #   * `inline_constants=True` and the leaf parses as any number (2, 4, 0.5 …)
    #   * the leaf is `0` or `1` — these are *always* short stubs with fixed
    #     colours (grey by default), regardless of `inline_constants`.
    # When omit_identity_labels=True (default), the *number* is suppressed
    # for L=0 and R=1 because their meaning is already inferable from the
    # grey colour and the L/R position; only the short grey stub is drawn.
    inline_leaves = []
    parent_of: dict = {}
    def _track_parents(p):
        for c in (p.children or []):
            parent_of[id(c)] = p
            _track_parents(c)
    _track_parents(node)

    for l in leaves:
        if l.label in fc:
            inline_leaves.append(l)
        elif inline_constants and _is_numeric_label(l.label):
            inline_leaves.append(l)
    inline_leaf_set = {id(l) for l in inline_leaves}

    def _is_identity_position(leaf) -> bool:
        """True if leaf is L=0 (exp leg vanishes) or R=1 (ln leg vanishes)
        of its parent eml junction — those are the cases the grey stub
        implies on its own without needing the number printed."""
        p = parent_of.get(id(leaf))
        if p is None or len(p.children) != 2:
            return False
        which = "L" if p.children[0] is leaf else "R" if p.children[1] is leaf else None
        if which == "L" and leaf.label == "0":
            return True
        if which == "R" and leaf.label == "1":
            return True
        return False

    for leaf in inline_leaves:
        if omit_identity_labels and _is_identity_position(leaf):
            continue   # the grey stub alone communicates 0-on-L / 1-on-R
        # Use the fixed-label colour where given (so a near-white '1' stays
        # readable), otherwise the branch colour.
        text_col = _rgb_hex(flc.get(leaf.label, leaf._fcolor))
        if direction == "down":
            tx, ty = leaf._fx, leaf._fy - 4
            anchor = "middle"
        elif direction == "up":
            tx, ty = leaf._fx, leaf._fy + label_font_size
            anchor = "middle"
        elif direction == "right":
            tx, ty = leaf._fx + 4, leaf._fy + label_font_size * 0.35
            anchor = "start"
        else:  # left
            tx, ty = leaf._fx - 4, leaf._fy + label_font_size * 0.35
            anchor = "end"
        parts.append(
            f'<text x="{tx:.1f}" y="{ty:.1f}" fill="{text_col}" text-anchor="{anchor}" '
            f'font-weight="700" font-size="{label_font_size}">{_esc(leaf.label)}</text>'
        )

    # ── input labels at the LEAD end ────────────────────────────────────────
    # Skip leaves that have already been drawn inline.
    leaves_for_top = [l for l in leaves if id(l) not in inline_leaf_set]
    if merge_inputs:
        # Group leaves by label; render ONE label per unique input and draw
        # 1-to-N redirector curves from the merged-input position to each
        # usage point (the leaf's screen position in the tree).
        # Inline constants are excluded from the top labels.
        unique_labels: List[str] = []
        for leaf in leaves_for_top:
            if leaf.label not in unique_labels:
                unique_labels.append(leaf.label)
        n_uniq = len(unique_labels)

        merged_pos: dict[str, Tuple[float, float]] = {}
        # Place the merged inputs at the very edge of the canvas — well
        # ahead of where the leaves sit (which got pushed inward by the 3×
        # margin bump) so the redirector curves have room to slerp.
        merged_offset = label_font_size * 1.6   # distance from the canvas edge
        for i, lbl in enumerate(unique_labels):
            cross_t = 0.5 if n_uniq == 1 else i / (n_uniq - 1)
            if direction == "down":
                mx = cross_margin + cross_t * (width - 2 * cross_margin)
                my = merged_offset
            elif direction == "up":
                mx = cross_margin + cross_t * (width - 2 * cross_margin)
                my = height - merged_offset
            elif direction == "right":
                my = cross_margin + cross_t * (height - 2 * cross_margin)
                mx = merged_offset
            else:  # left
                my = cross_margin + cross_t * (height - 2 * cross_margin)
                mx = width - merged_offset
            merged_pos[lbl] = (mx, my)

        # Redirector curves: from each merged-input position to every leaf's tree position.
        # Strong vertical bias (0.85) so the curves leave the merged-input
        # row going firmly DOWN instead of flattening into long horizontal runs.
        for leaf in leaves_for_top:
            mx, my = merged_pos[leaf.label]
            col = _rgb_hex(leaf._fcolor)
            parts.append(_curve_d(mx, my, leaf._fx, leaf._fy, direction, col,
                                   edge_width, vertical_bias=0.85))

        # Render one label per unique input — every leaf is a real input.
        label_color = {l: c for l, c in _label_colors_iter(leaves_for_top)}
        for lbl in unique_labels:
            mx, my = merged_pos[lbl]
            col = _rgb_hex(label_color[lbl])
            lx, ly = _label_offset(mx, my, direction, label_font_size, "lead")
            anchor = _text_anchor(direction, "lead")
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{col}" text-anchor="{anchor}" '
                f'font-weight="700" font-size="{label_font_size}">{_esc(lbl)}</text>'
            )
    else:
        for leaf in leaves_for_top:
            col = _rgb_hex(leaf._fcolor)
            lx, ly = _label_offset(leaf._fx, leaf._fy, direction, label_font_size, "lead")
            anchor = _text_anchor(direction, "lead")
            parts.append(
                f'<text x="{lx:.1f}" y="{ly:.1f}" fill="{col}" text-anchor="{anchor}" '
                f'font-weight="700" font-size="{label_font_size}">{_esc(leaf.label)}</text>'
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


def _is_numeric_label(label: str) -> bool:
    """True if *label* parses as a finite real number (e.g. '1', '-2.5', '0.5')."""
    try:
        float(label)
        return True
    except (ValueError, TypeError):
        return False


def _label_colors_iter(leaves):
    """Yield (label, color) pairs in iteration order (after _layout has set _fcolor)."""
    seen = set()
    for l in leaves:
        if l.label in seen:
            continue
        seen.add(l.label)
        yield l.label, l._fcolor


def _curve_d(cx: float, cy: float, px: float, py: float, direction: str,
             stroke: str, edge_width: float, *,
             vertical_bias: float = 0.5) -> str:
    """Cubic-Bezier path string from (cx, cy) to (px, py).

    `vertical_bias` ∈ [0, 1] controls how far along the primary axis the
    control points sit. 0.5 (the default) puts them at the midpoint —
    smooth S-curve. Higher values push the control points closer to the
    *opposite* endpoint along the primary axis, forcing the curve to
    leave each endpoint more vertically before bending — useful for
    merge-input redirector curves where horizontal travel >> vertical
    and the default S-curve flattens out.
    """
    vertical_bias = max(0.05, min(0.95, vertical_bias))
    if direction in ("down", "up"):
        # Control y for child end and parent end.
        c_ctrl_y = cy + (py - cy) * vertical_bias
        p_ctrl_y = py - (py - cy) * vertical_bias
        d = (f"M{cx:.1f},{cy:.1f} C{cx:.1f},{c_ctrl_y:.1f} "
             f"{px:.1f},{p_ctrl_y:.1f} {px:.1f},{py:.1f}")
    else:  # right or left
        c_ctrl_x = cx + (px - cx) * vertical_bias
        p_ctrl_x = px - (px - cx) * vertical_bias
        d = (f"M{cx:.1f},{cy:.1f} C{c_ctrl_x:.1f},{cy:.1f} "
             f"{p_ctrl_x:.1f},{py:.1f} {px:.1f},{py:.1f}")
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
    inline_constants: bool = False,
    fixed_colors: Optional[dict] = None,
    fixed_label_colors: Optional[dict] = None,
    bypass_identity_blend: bool = True,
    omit_identity_labels: bool = True,
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
    if merge_inputs:
        # Big multiplier so the redirector curves from the merged-input
        # row down to each leaf's tree position have real vertical room
        # to slerp — without it they end up looking horizontal.
        primary_label_size *= 5.5   # room above leaves for slerp from merged inputs

    preview_root = _binarize(_expand_symbols_in_tree(node) if expand_symbols else node)
    leaves_preview = _collect_leaves(preview_root)
    max_label_len  = max((len(l.label) for l in leaves_preview), default=1)
    half_label_w   = 0.5 * 0.6 * label_font_size * max_label_len
    cross_margin   = max(40.0, half_label_w + 12.0)

    fc = FIXED_COLORS if fixed_colors is None else fixed_colors
    flc = FIXED_LABEL_COLORS if fixed_label_colors is None else fixed_label_colors

    node, leaves = _layout(
        node,
        width=width, height=height,
        margin_lead=primary_label_size,
        margin_trail=primary_output_size,
        margin_cross=cross_margin,
        palette=palette, direction=direction,
        expand_symbols=expand_symbols,
        fixed_colors=fc,
        bypass_identity_blend=bypass_identity_blend,
    )

    _stub_inline_leaves(node, direction=direction, fixed_labels=fc.keys(),
                         inline_constants=inline_constants,
                         label_font_size=label_font_size)

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

    def _curve_pts_for_pil(c_x, c_y, p_x, p_y, vertical_bias: float = 0.5):
        vb = max(0.05, min(0.95, vertical_bias))
        p0 = (c_x * scale, c_y * scale)
        p3 = (p_x * scale, p_y * scale)
        if direction in ("down", "up"):
            c_y_ctrl = p0[1] + (p3[1] - p0[1]) * vb
            p_y_ctrl = p3[1] - (p3[1] - p0[1]) * vb
            p1 = (p0[0], c_y_ctrl); p2 = (p3[0], p_y_ctrl)
        else:  # right or left
            c_x_ctrl = p0[0] + (p3[0] - p0[0]) * vb
            p_x_ctrl = p3[0] - (p3[0] - p0[0]) * vb
            p1 = (c_x_ctrl, p0[1]); p2 = (p_x_ctrl, p3[1])
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

    # Inline pass: 0 and 1 (or whatever fixed_colors keys you passed) are
    # *always* drawn as short stubs at their branch endpoints. Other
    # numeric leaves only become inline when inline_constants=True.
    inline_constant_leaves = []
    parent_of_pil: dict = {}
    def _track_p(p):
        for c in (p.children or []):
            parent_of_pil[id(c)] = p
            _track_p(c)
    _track_p(node)

    for l in leaves:
        if l.label in fc:
            inline_constant_leaves.append(l)
        elif inline_constants and _is_numeric_label(l.label):
            inline_constant_leaves.append(l)
    inline_constant_set = {id(l) for l in inline_constant_leaves}

    def _is_identity_pos(leaf) -> bool:
        p = parent_of_pil.get(id(leaf))
        if p is None or len(p.children) != 2:
            return False
        if p.children[0] is leaf and leaf.label == "0":
            return True
        if p.children[1] is leaf and leaf.label == "1":
            return True
        return False

    for leaf in inline_constant_leaves:
        if omit_identity_labels and _is_identity_pos(leaf):
            continue
        text_rgb = tuple(int(round(v)) for v in flc.get(leaf.label, leaf._fcolor))
        bbox = draw.textbbox((0, 0), leaf.label, font=font_label)
        tw = bbox[2] - bbox[0]; th = bbox[3] - bbox[1]
        lx = leaf._fx * scale; ly = leaf._fy * scale
        if direction == "down":
            tx = lx - tw / 2;        ty = ly - th - 2 * scale
        elif direction == "up":
            tx = lx - tw / 2;        ty = ly + 2 * scale
        elif direction == "right":
            tx = lx + 4 * scale;     ty = ly - th / 2
        else:  # left
            tx = lx - tw - 4 * scale; ty = ly - th / 2
        draw.text((tx, ty), leaf.label, fill=text_rgb, font=font_label)

    leaves_for_top = [l for l in leaves if id(l) not in inline_constant_set]

    if merge_inputs:
        # One input position per unique leaf label; redirector curves from
        # the merged position to every usage point in the tree.
        unique_labels = []
        for l in leaves_for_top:
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

        # Redirector curves (strong vertical bias so they don't look horizontal)
        for leaf in leaves_for_top:
            mx, my = merged_pos[leaf.label]
            col = tuple(int(round(v)) for v in leaf._fcolor)
            pts = _curve_pts_for_pil(mx, my, leaf._fx, leaf._fy, vertical_bias=0.85)
            draw.line(pts, fill=col, width=ew, joint="curve")

        # One label per unique input — every leaf is a real input.
        label_color = {}
        for lbl, c in _label_colors_iter(leaves_for_top):
            label_color[lbl] = tuple(int(round(v)) for v in c)
        for lbl in unique_labels:
            mx, my = merged_pos[lbl]
            _draw_leaf_label(lbl, label_color[lbl], mx * scale, my * scale)
    else:
        for leaf in leaves_for_top:
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
