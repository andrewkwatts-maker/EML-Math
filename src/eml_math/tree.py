"""
eml_math.tree — EML operator-tree renderer.

Parses an ``eml_description`` string into a labeled :class:`EMLTreeNode` tree
by walking Python's own ``ast`` module (so ops.* call names are preserved as
node labels).  Renders as:

* **ASCII** — ``node.ascii()``   indented box-and-line tree, e.g.::

      mul  [exp(ln·ln)]
      ├── V_cb
      └── pow  [exp(n·ln)]
          ├── lambda_wolfenstein
          └── 2.0

* **SVG** — ``node.svg()``  self-contained SVG string (no external deps).
  Compound ops get a teal box, EML primitives get an amber box,
  leaf scalars/vecs get coloured circles.

* **dict/JSON** — ``node.to_dict()``  JSON-serializable tree for web / D3.

Quick start::

    from eml_math.tree import parse_eml_tree

    desc = "EML: ops.mul(eml_vec('V_cb'), ops.pow(eml_vec('lambda'), eml_scalar(2.0))) — V_cb·λ²"
    tree = parse_eml_tree(desc)
    print(tree.ascii())      # ASCII tree
    svg  = tree.svg()        # SVG string ready to embed in HTML
    data = tree.to_dict()    # JSON dict for web renderer
"""
from __future__ import annotations

import ast
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "EMLTreeNode",
    "NodeKind",
    "EML_EXPANSIONS",
    "parse_eml_tree",
]


# ── EML primitive expansions ──────────────────────────────────────────────────

EML_EXPANSIONS: Dict[str, str] = {
    # EML primitives — atomic
    "eml":    "exp(x) − ln(y)",
    "exp":    "eml(x, 1)",
    "ln":     "eml(1, eml(eml(1,x), 1))",
    # Arithmetic — compound
    "add":    "sub(a, neg(b))",
    "sub":    "a.t − b.t",
    "neg":    "0 − x",
    "inv":    "exp(−ln(x))",
    "mul":    "exp(ln(a) + ln(b))",
    "div":    "exp(ln(a) − ln(b))",
    "sqrt":   "exp(½·ln(x))",
    "sqr":    "exp(2·ln(x))",
    "pow":    "exp(n·ln(x))",
    "pow_fn": "exp(n·ln(x))",
    # Trig — compound (immediate eval pending Table-1 depth-15 chain)
    "sin":    "depth-15 EML",
    "cos":    "depth-15 EML",
    "tan":    "sin/cos",
    "asin":   "depth-15 EML",
    "acos":   "depth-15 EML",
    "atan":   "depth-15 EML",
    "sinh":   "½(eˣ−e⁻ˣ)",
    "cosh":   "½(eˣ+e⁻ˣ)",
    "tanh":   "sinh/cosh",
    # Other
    "abs":    "|x|",
    "log_fn": "ln(x)/ln(b)",
    "sqr":    "exp(2·ln(x))",
}

_PRIMITIVES: frozenset = frozenset({"eml", "exp", "ln"})


class NodeKind:
    COMPOUND  = "compound"   # ops.mul, ops.sin, ops.pow …  (higher-level box)
    PRIMITIVE = "primitive"  # ops.exp, ops.ln  (raw EML)
    SCALAR    = "scalar"     # eml_scalar(x)
    VEC       = "vec"        # eml_vec('name')
    PI        = "pi"         # eml_pi()
    CONST     = "const"      # bare numeric literal
    UNKNOWN   = "unknown"


# ── Tree node ─────────────────────────────────────────────────────────────────

@dataclass
class EMLTreeNode:
    """
    One node in the parsed EML operator tree.

    Attributes
    ----------
    label :
        Display name: ``"mul"``, ``"2.718"``, ``"lambda_wolfenstein"`` …
    kind :
        One of the :class:`NodeKind` constants.
    children :
        Ordered child nodes (operands).
    eml_form :
        Brief annotation showing how this operator expands into EML primitives,
        e.g. ``"exp(ln·ln)"`` for *mul*.
    """
    label:    str
    kind:     str                    = NodeKind.UNKNOWN
    children: List["EMLTreeNode"]    = field(default_factory=list)
    eml_form: str                    = ""

    # layout scratch-space (set by _layout / _assign_pos)
    _px: float = field(default=0.0, repr=False, compare=False)
    _py: float = field(default=0.0, repr=False, compare=False)
    _sw: float = field(default=0.0, repr=False, compare=False)  # subtree width

    # ------------------------------------------------------------------
    # ASCII renderer
    # ------------------------------------------------------------------

    def ascii(self, *, _pfx: str = "", _last: bool = True) -> str:
        """Return a multi-line ASCII art representation of the tree."""
        conn  = "└── " if _last else "├── "
        badge = f"  [{self.eml_form}]" if self.eml_form else ""
        lines = [_pfx + conn + self.label + badge]

        child_pfx = _pfx + ("    " if _last else "│   ")
        for i, child in enumerate(self.children):
            lines.append(child.ascii(_pfx=child_pfx, _last=(i == len(self.children) - 1)))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.ascii(_pfx="", _last=True)

    # ------------------------------------------------------------------
    # JSON / dict
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """JSON-serializable tree dict for web renderers (D3 etc.)."""
        d: Dict[str, Any] = {"label": self.label, "kind": self.kind}
        if self.eml_form:
            d["eml_form"] = self.eml_form
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    # ------------------------------------------------------------------
    # SVG renderer
    # ------------------------------------------------------------------

    # Layout constants
    _NW   = 130   # node box width
    _NH   = 46    # node box height
    _HGAP = 18    # horizontal gap between siblings
    _VGAP = 72    # vertical gap between levels
    _PAD  = 24    # canvas padding

    # Colour palette per kind
    _COLORS = {
        NodeKind.COMPOUND:  ("#2E86C1", "#EBF5FB"),   # teal border, light fill
        NodeKind.PRIMITIVE: ("#D35400", "#FEF0E7"),   # amber
        NodeKind.SCALAR:    ("#1E8449", "#EAFAF1"),   # green
        NodeKind.VEC:       ("#7D3C98", "#F5EEF8"),   # purple
        NodeKind.PI:        ("#7D3C98", "#F5EEF8"),   # purple
        NodeKind.CONST:     ("#1E8449", "#EAFAF1"),   # green
        NodeKind.UNKNOWN:   ("#7F8C8D", "#F2F3F4"),   # grey
    }

    def svg(self, *, max_width: int = 1000) -> str:
        """
        Return a self-contained SVG string.

        Parameters
        ----------
        max_width :
            Maximum canvas width in pixels; tree scales to fit.
        """
        _compute_subtree_width(self)
        canvas_w = max(max_width, int(self._sw + 2 * self._PAD))
        _assign_positions(self, x_left=self._PAD, level=0)
        depth    = _tree_depth(self)
        canvas_h = self._PAD * 2 + depth * (self._NH + self._VGAP) + self._NH

        parts: List[str] = []
        parts.append(
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_w}" height="{canvas_h}" '
            f'font-family="monospace" font-size="12">'
        )
        parts.append(
            '<style>'
            '.node-box{rx:8;ry:8;stroke-width:2}'
            '.node-label{text-anchor:middle;dominant-baseline:central;font-weight:bold}'
            '.eml-badge{text-anchor:middle;dominant-baseline:central;font-size:10;fill:#666}'
            '.edge{stroke:#AAB7B8;stroke-width:1.5;fill:none}'
            '</style>'
        )
        # Draw edges first (so boxes paint on top)
        _emit_edges(self, parts)
        # Draw boxes
        _emit_nodes(self, parts)
        parts.append("</svg>")
        return "\n".join(parts)


# ── Layout helpers ────────────────────────────────────────────────────────────

def _compute_subtree_width(node: EMLTreeNode) -> float:
    NW, HGAP = EMLTreeNode._NW, EMLTreeNode._HGAP
    if not node.children:
        node._sw = float(NW)
    else:
        kids_w  = sum(_compute_subtree_width(c) for c in node.children)
        gaps    = (len(node.children) - 1) * HGAP
        node._sw = max(float(NW), kids_w + gaps)
    return node._sw


def _assign_positions(node: EMLTreeNode, x_left: float, level: int) -> None:
    NW, NH, HGAP, VGAP, PAD = (
        EMLTreeNode._NW, EMLTreeNode._NH,
        EMLTreeNode._HGAP, EMLTreeNode._VGAP, EMLTreeNode._PAD,
    )
    node._py = PAD + level * (NH + VGAP)
    node._px = x_left + (node._sw - NW) / 2.0
    cursor = x_left
    for child in node.children:
        _assign_positions(child, cursor, level + 1)
        cursor += child._sw + HGAP


def _tree_depth(node: EMLTreeNode) -> int:
    if not node.children:
        return 0
    return 1 + max(_tree_depth(c) for c in node.children)


def _emit_edges(node: EMLTreeNode, out: List[str]) -> None:
    NW, NH = EMLTreeNode._NW, EMLTreeNode._NH
    px = node._px + NW / 2
    py = node._py + NH
    for child in node.children:
        cx = child._px + NW / 2
        cy = child._py
        mid_y = (py + cy) / 2
        out.append(
            f'<path class="edge" d="M{px:.1f},{py:.1f} '
            f'C{px:.1f},{mid_y:.1f} {cx:.1f},{mid_y:.1f} {cx:.1f},{cy:.1f}"/>'
        )
        _emit_edges(child, out)


def _emit_nodes(node: EMLTreeNode, out: List[str]) -> None:
    NW, NH = EMLTreeNode._NW, EMLTreeNode._NH
    stroke, fill = EMLTreeNode._COLORS.get(
        node.kind, EMLTreeNode._COLORS[NodeKind.UNKNOWN]
    )
    x, y = node._px, node._py

    out.append(
        f'<rect class="node-box" x="{x:.1f}" y="{y:.1f}" '
        f'width="{NW}" height="{NH}" '
        f'stroke="{stroke}" fill="{fill}"/>'
    )
    # Label — truncate long names
    label = node.label if len(node.label) <= 18 else node.label[:16] + "…"
    cx, cy = x + NW / 2, y + NH / 2

    if node.eml_form:
        # Split vertically: label top, eml_form bottom
        out.append(
            f'<text class="node-label" x="{cx:.1f}" y="{y + NH*0.38:.1f}" '
            f'fill="{stroke}">{_esc(label)}</text>'
        )
        badge = node.eml_form if len(node.eml_form) <= 22 else node.eml_form[:20] + "…"
        out.append(
            f'<text class="eml-badge" x="{cx:.1f}" y="{y + NH*0.72:.1f}">'
            f'{_esc(badge)}</text>'
        )
    else:
        out.append(
            f'<text class="node-label" x="{cx:.1f}" y="{cy:.1f}" '
            f'fill="{stroke}">{_esc(label)}</text>'
        )

    for child in node.children:
        _emit_nodes(child, out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── AST → EMLTreeNode parser ─────────────────────────────────────────────────

def parse_eml_tree(eml_description: str) -> EMLTreeNode:
    """
    Parse an ``eml_description`` string into an :class:`EMLTreeNode` tree.

    Parameters
    ----------
    eml_description :
        A string starting with ``"EML: "`` in the standard format.
        The expression is parsed via :mod:`ast` so ``ops.*`` call names
        are preserved as tree-node labels.

    Returns
    -------
    EMLTreeNode
        Root of the operator tree.

    Example
    -------
    >>> t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_vec('b3')) — 3·b₃")
    >>> print(t.ascii())
    └── mul  [exp(ln·ln)]
        ├── 3.0
        └── b3
    """
    from eml_math.evaluator import EMLEvaluator
    expr = EMLEvaluator._parse(eml_description)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return EMLTreeNode(label=f"<parse error: {exc}>", kind=NodeKind.UNKNOWN)
    return _ast_to_node(tree.body)


def _ast_to_node(node: ast.expr) -> EMLTreeNode:  # noqa: C901
    """Recursively convert a Python AST expression node to an EMLTreeNode."""

    # ── function call ──────────────────────────────────────────────────
    if isinstance(node, ast.Call):
        func_name = _func_name(node.func)
        children  = [_ast_to_node(a) for a in node.args]

        # --- eml_scalar(x) ---
        if func_name == "eml_scalar":
            val = _literal_value(node.args[0]) if node.args else "?"
            return EMLTreeNode(label=_fmt_num(val), kind=NodeKind.SCALAR)

        # --- eml_pi() ---
        if func_name == "eml_pi":
            return EMLTreeNode(label="π", kind=NodeKind.PI)

        # --- eml_vec('name') ---
        if func_name == "eml_vec":
            name = _literal_value(node.args[0]) if node.args else "?"
            return EMLTreeNode(label=str(name), kind=NodeKind.VEC)

        # --- ops.exp / ops.ln / ops.eml (primitives) ---
        if func_name in _PRIMITIVES:
            eml_form = EML_EXPANSIONS.get(func_name, "")
            return EMLTreeNode(label=func_name, kind=NodeKind.PRIMITIVE,
                               children=children, eml_form=eml_form)

        # --- ops.* compound operators ---
        eml_form = EML_EXPANSIONS.get(func_name, "")
        return EMLTreeNode(label=func_name, kind=NodeKind.COMPOUND,
                           children=children, eml_form=eml_form)

    # ── attribute access: ops.mul → "mul" ─────────────────────────────
    if isinstance(node, ast.Attribute):
        return EMLTreeNode(label=node.attr, kind=NodeKind.UNKNOWN)

    # ── bare name ─────────────────────────────────────────────────────
    if isinstance(node, ast.Name):
        n = node.id
        if n == "math":
            return EMLTreeNode(label="math", kind=NodeKind.UNKNOWN)
        return EMLTreeNode(label=n, kind=NodeKind.VEC)

    # ── numeric / string constant ──────────────────────────────────────
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, (int, float)):
            return EMLTreeNode(label=_fmt_num(v), kind=NodeKind.CONST)
        return EMLTreeNode(label=repr(v), kind=NodeKind.CONST)

    # ── unary negation (e.g. -1.0) ────────────────────────────────────
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_to_node(node.operand)
        return EMLTreeNode(label=f"-{inner.label}", kind=inner.kind)

    # ── tuple / list (rare) ───────────────────────────────────────────
    if isinstance(node, (ast.Tuple, ast.List)):
        children = [_ast_to_node(e) for e in node.elts]
        return EMLTreeNode(label="(…)", kind=NodeKind.UNKNOWN, children=children)

    # ── fallback ──────────────────────────────────────────────────────
    return EMLTreeNode(label=ast.dump(node)[:30], kind=NodeKind.UNKNOWN)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _func_name(node: ast.expr) -> str:
    """Extract the bare function name from a Call's func node."""
    if isinstance(node, ast.Attribute):
        return node.attr          # ops.mul → "mul"
    if isinstance(node, ast.Name):
        return node.id            # eml_scalar → "eml_scalar"
    return ast.dump(node)[:20]


def _literal_value(node: ast.expr) -> Any:
    """Extract the Python literal value from a Constant or simple Name node."""
    if isinstance(node, ast.Constant):
        return node.value
    try:
        return ast.literal_eval(node)
    except Exception:
        return ast.dump(node)[:20]


def _fmt_num(v: Any) -> str:
    """Format a numeric value compactly for display."""
    if not isinstance(v, (int, float)):
        return str(v)
    if v == math.pi:
        return "π"
    if v == math.e:
        return "e"
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e6):
        return str(int(v))
    # scientific if very large/small
    if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.3e}"
    return f"{v:.6g}"
