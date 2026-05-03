"""
Tidy tree layout for the EML formula renderer.

Algorithm: classic Reingold-Tilford "tidy tree" — two passes over a binary
or n-ary tree:

    1. Post-order: assign each subtree a *contour* (left/right extent at
       each depth) and a relative x-offset; pack siblings against each
       other's contours so subtrees never overlap.
    2. Pre-order: accumulate parent offsets to produce final coordinates.

The output is the **layout dict**, the canonical contract every renderer
consumes:

    {
      "schema": "eml-layout/v1",
      "canvas": {"width": W, "height": H},
      "direction": "down",
      "edge_style": "curve",
      "nodes": [{"id", "label", "kind", "x", "y", "color", "is_leaf", ...}],
      "edges": [{"from", "to", "style", "color"}, ...]
    }

Pure stdlib.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from eml_math.render.palette import (
    DEFAULT_PALETTE,
    blend,
    pastel_for_label,
)

__all__ = [
    "compute_layout",
    "DIRECTIONS",
    "EDGE_STYLES",
    "LAYOUT_SCHEMA",
]

DIRECTIONS = ("down", "up", "right", "left")
EDGE_STYLES = ("straight", "curve", "spline")
LAYOUT_SCHEMA = "eml-layout/v1"


# ── Reingold-Tilford internals ───────────────────────────────────────────────

class _LNode:
    """A working node for the tidy-tree algorithm."""
    __slots__ = (
        "id", "label", "kind", "children",
        "x", "y", "depth", "mod", "shift",
        "color", "raw",
    )

    def __init__(self, raw: Dict[str, Any], idx: List[int]) -> None:
        self.raw = raw
        self.id = f"n{idx[0]}"
        idx[0] += 1
        self.label = str(raw.get("label", ""))
        self.kind = str(raw.get("kind", "unknown"))
        self.children: List[_LNode] = [
            _LNode(c, idx) for c in raw.get("children", []) or []
        ]
        self.x: float = 0.0      # logical x (relative units, set by RT)
        self.y: float = 0.0      # logical y (= depth in RT)
        self.depth: int = 0
        self.mod: float = 0.0    # accumulated subtree shift
        self.shift: float = 0.0  # this node's relative position within siblings
        self.color: Tuple[int, int, int] = (200, 200, 200)


def _measure_depth(n: _LNode, d: int = 0) -> int:
    n.depth = d
    if not n.children:
        return d
    return max(_measure_depth(c, d + 1) for c in n.children)


def _contour(n: _LNode, mod_acc: float, side: str,
             out: Dict[int, float]) -> None:
    """Walk the left or right contour of subtree *n*, updating *out* in place.

    ``out[depth]`` becomes the extreme x at that depth (min for "left",
    max for "right"). *mod_acc* is the accumulated parent ``mod`` shift.
    """
    x = n.shift + mod_acc
    if side == "left":
        if n.depth not in out or x < out[n.depth]:
            out[n.depth] = x
    else:
        if n.depth not in out or x > out[n.depth]:
            out[n.depth] = x
    for c in n.children:
        _contour(c, mod_acc + n.mod, side, out)


def _layout_pass1(n: _LNode, sibling_spacing: float,
                  subtree_spacing: float) -> None:
    """First pass: assign each node its ``shift`` relative to its parent.

    Recursive post-order. Two siblings are pushed apart by the difference
    of their contours plus *sibling_spacing*.
    """
    for c in n.children:
        _layout_pass1(c, sibling_spacing, subtree_spacing)

    if not n.children:
        n.shift = 0.0
        return

    # Centre children around 0; then push right siblings further to clear
    # left siblings' right-contour.
    n.children[0].shift = 0.0
    for i in range(1, len(n.children)):
        left = n.children[i - 1]
        right = n.children[i]
        rcontour: Dict[int, float] = {}
        _contour(left, 0.0, "right", rcontour)
        lcontour: Dict[int, float] = {}
        _contour(right, 0.0, "left", lcontour)
        # Required gap: at every depth where both contours exist, the
        # right-contour of left + spacing must be ≤ left-contour of right.
        push = sibling_spacing
        for d in rcontour:
            if d in lcontour:
                gap = lcontour[d] - rcontour[d]
                need = sibling_spacing - gap
                if need > push:
                    push = need
        right.shift = left.shift + push + subtree_spacing * 0.0   # subtree_spacing reserved

    # Centre the parent over its children.
    mid = (n.children[0].shift + n.children[-1].shift) / 2.0
    n.shift = mid
    n.mod = -mid
    # Re-anchor children so the parent sits at shift=mid (no actual move,
    # we just propagate via mod).


def _layout_pass2(n: _LNode, parent_x: float = 0.0) -> None:
    """Second pass: accumulate ``mod`` shifts into final ``x``."""
    n.x = parent_x + n.shift
    for c in n.children:
        _layout_pass2(c, n.x + n.mod)


# ── Colour assignment ────────────────────────────────────────────────────────

def _assign_colors(
    n: _LNode,
    leaf_idx: List[int],
    palette: Sequence[Tuple[int, int, int]],
    random_palette: bool,
    fixed_colors: Dict[str, Tuple[int, int, int]],
    bypass_identity_blend: bool,
) -> Tuple[int, int, int]:
    """Recursively colour leaves from *palette* and junctions as the blend
    of their children."""
    if not n.children:
        if n.label in fixed_colors:
            n.color = fixed_colors[n.label]
        elif random_palette:
            n.color = pastel_for_label(n.label)
        else:
            n.color = palette[leaf_idx[0] % len(palette)]
            leaf_idx[0] += 1
        return n.color

    child_cols = [
        _assign_colors(c, leaf_idx, palette, random_palette,
                       fixed_colors, bypass_identity_blend)
        for c in n.children
    ]
    if bypass_identity_blend:
        # Drop fixed-colour leaves (0, 1) from the blend so an eml(x, 1)
        # junction takes x's colour directly.
        live = [
            col for c, col in zip(n.children, child_cols)
            if not (not c.children and c.label in fixed_colors)
        ]
        if live:
            child_cols = live
    n.color = blend(*child_cols)
    return n.color


# ── Public API ───────────────────────────────────────────────────────────────

def compute_layout(
    formula: Dict[str, Any],
    *,
    direction: str = "down",
    canvas: Tuple[int, int] = (720, 440),
    margin: float = 40.0,
    sibling_spacing: float = 40.0,
    subtree_spacing: float = 12.0,
    edge_style: str = "curve",
    palette: Sequence[Tuple[int, int, int]] = DEFAULT_PALETTE,
    random_palette: bool = False,
    fixed_colors: Optional[Dict[str, Tuple[int, int, int]]] = None,
    bypass_identity_blend: bool = True,
    auto_canvas: bool = True,
    min_layer_height: float = 56.0,
) -> Dict[str, Any]:
    """Lay out a raw EML formula JSON tree.

    Parameters
    ----------
    formula : dict
        Output of ``EMLTreeNode.to_dict()`` — must have ``label``, ``kind``,
        and optional ``children``. The optional ``schema`` field is ignored.
    direction : {"down", "up", "right", "left"}
        Which way the formula flows. ``"down"`` puts inputs at top and the
        output at bottom (default).
    canvas : (width, height)
        Pixel size of the rendering surface. With *auto_canvas* on (default),
        the primary axis grows to fit deep trees so node spacing stays
        readable.
    sibling_spacing : float
        Minimum cross-axis gap between sibling subtrees (Reingold-Tilford
        guarantees no overlap below this).
    edge_style : {"straight", "curve", "spline"}
        Default style stamped on every edge. Renderers honour this.
    palette : sequence of (r, g, b)
        Per-leaf colour cycle (recycled if there are more leaves than
        entries).
    random_palette : bool
        If True, ignore *palette* and assign each unique label a
        deterministic pastel hue (so equal labels share a colour).
    fixed_colors : dict label → (r, g, b)
        Override leaf colours for specific labels — defaults pin ``"0"`` and
        ``"1"`` to black so identity arms stay visually quiet.

    Returns
    -------
    dict
        A layout dict keyed by the schema ``"eml-layout/v1"``.
    """
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    if edge_style not in EDGE_STYLES:
        raise ValueError(f"edge_style must be one of {EDGE_STYLES}, got {edge_style!r}")

    if fixed_colors is None:
        fixed_colors = {"0": (0, 0, 0), "1": (0, 0, 0)}

    width, height = canvas

    # 1) Build working tree & lay out in logical units.
    idx = [0]
    root = _LNode(formula, idx)
    max_depth = _measure_depth(root, 0)
    _layout_pass1(root, sibling_spacing, subtree_spacing)
    _layout_pass2(root, 0.0)

    # 2) Assign colours.
    _assign_colors(root, [0], palette, random_palette,
                   fixed_colors, bypass_identity_blend)

    # 3) Find logical x-range (min/max across the tree).
    nodes_flat: List[_LNode] = []

    def _flatten(n: _LNode) -> None:
        nodes_flat.append(n)
        for c in n.children:
            _flatten(c)

    _flatten(root)
    min_x = min(n.x for n in nodes_flat)
    max_x = max(n.x for n in nodes_flat)
    span = max_x - min_x if max_x > min_x else 1.0

    # 4) Auto-grow canvas along the primary axis if requested.
    if auto_canvas:
        primary_needed = (max_depth + 1) * min_layer_height + 2 * margin
        if direction in ("down", "up"):
            height = max(height, int(primary_needed))
        else:
            width = max(width, int(primary_needed))

    # 5) Project (logical_x, depth) → (screen x, y) for the chosen direction.
    if direction in ("down", "up"):
        cross = width
        primary = height
    else:
        cross = height
        primary = width

    cross_lo = margin
    cross_hi = cross - margin
    cross_span = cross_hi - cross_lo

    primary_lo = margin
    primary_hi = primary - margin
    primary_span = primary_hi - primary_lo
    layer_h = primary_span / max(1, max_depth)

    # Centre logical x within [cross_lo, cross_hi].
    def _to_cross(lx: float) -> float:
        if span == 0:
            return (cross_lo + cross_hi) / 2.0
        t = (lx - min_x) / span
        return cross_lo + t * cross_span

    def _to_primary(d: int) -> float:
        # depth d=0 at root, d=max_depth at deepest leaf.
        # For "down": leaves at top → primary_lo at deepest; root at primary_hi.
        if direction == "down":
            return primary_lo + (max_depth - d) * layer_h
        if direction == "up":
            return primary_hi - (max_depth - d) * layer_h
        if direction == "right":
            return primary_lo + (max_depth - d) * layer_h
        # left
        return primary_hi - (max_depth - d) * layer_h

    nodes_out: List[Dict[str, Any]] = []
    edges_out: List[Dict[str, Any]] = []
    for n in nodes_flat:
        cx = _to_cross(n.x)
        cy = _to_primary(n.depth)
        if direction in ("down", "up"):
            sx, sy = cx, cy
        else:
            sx, sy = cy, cx
        nodes_out.append({
            "id": n.id,
            "label": n.label,
            "kind": n.kind,
            "x": round(sx, 2),
            "y": round(sy, 2),
            "color": list(n.color),
            "is_leaf": not n.children,
            "depth": n.depth,
        })
        for c in n.children:
            edges_out.append({
                "from": c.id,   # child end
                "to": n.id,     # parent (junction) end
                "style": edge_style,
                "color": list(c.color),
            })

    return {
        "schema": LAYOUT_SCHEMA,
        "canvas": {"width": int(width), "height": int(height)},
        "direction": direction,
        "edge_style": edge_style,
        "nodes": nodes_out,
        "edges": edges_out,
    }
