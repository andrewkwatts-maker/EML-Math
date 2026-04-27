"""
eml_math.tree — EML operator-tree renderer.

Parses an ``eml_description`` string into a labeled :class:`EMLTreeNode` tree.
By default (``expand_eml=True``) compound ops are *fully expanded* into their
underlying EML primitives so the rendered tree consists only of:

* **exp** / **ln** — EML primitives  (amber boxes)
* **add** / **sub** / **neg** / **scale** — structural helpers  (grey boxes)
* leaf scalars, vecs, pi  (green / purple circles)

Example::

    from eml_math.tree import parse_eml_tree

    desc = "EML: ops.mul(eml_vec('A'), ops.pow(eml_vec('lam'), eml_scalar(2.0)))"
    print(parse_eml_tree(desc).ascii())

    └── exp
        └── add
            ├── ln
            │   └── A
            └── ln
                └── exp
                    └── ×2.0
                        └── ln
                            └── lam

Pass ``expand_eml=False`` to keep the compact ops-level tree (``mul``, ``pow``
shown as single boxes with EML-form annotations).

Render targets
--------------
``node.ascii()``  — indented Unicode art tree
``node.svg()``    — self-contained inline SVG
``node.to_dict()`` — JSON dict for web renderers / D3
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


# ── EML expansion annotations (compact ops-level view) ───────────────────────

EML_EXPANSIONS: Dict[str, str] = {
    "eml":    "exp(x) − ln(y)",
    "exp":    "eml(x, 1)",
    "ln":     "eml(1,eml(eml(1,x),1))",
    "add":    "sub(a, neg(b))",
    "sub":    "a − b",
    "neg":    "0 − x",
    "inv":    "exp(−ln(x))",
    "mul":    "exp(ln(a)+ln(b))",
    "div":    "exp(ln(a)−ln(b))",
    "sqrt":   "exp(½·ln(x))",
    "sqr":    "exp(2·ln(x))",
    "pow":    "exp(n·ln(x))",
    "pow_fn": "exp(n·ln(x))",
    "sin":    "depth-15 EML",
    "cos":    "depth-15 EML",
    "tan":    "sin/cos",
    "asin":   "depth-15 EML",
    "acos":   "depth-15 EML",
    "atan":   "depth-15 EML",
    "sinh":   "½(eˣ−e⁻ˣ)",
    "cosh":   "½(eˣ+e⁻ˣ)",
    "tanh":   "sinh/cosh",
    "abs":    "|x|",
    "log_fn": "ln(x)/ln(b)",
}

_PRIMITIVES: frozenset = frozenset({"eml", "exp", "ln"})


class NodeKind:
    PRIMITIVE  = "primitive"   # exp, ln  (EML atomic)
    STRUCTURAL = "structural"  # add, sub, neg, scale  (EML scaffolding)
    COMPOUND   = "compound"    # sin, cos … (no EML expansion defined yet)
    SCALAR     = "scalar"      # eml_scalar(x)
    VEC        = "vec"         # eml_vec('name')
    PI         = "pi"          # eml_pi()
    CONST      = "const"       # bare numeric literal
    UNKNOWN    = "unknown"


# ── Tree node ─────────────────────────────────────────────────────────────────

@dataclass
class EMLTreeNode:
    """One node in the EML operator tree."""
    label:    str
    kind:     str                 = NodeKind.UNKNOWN
    children: List["EMLTreeNode"] = field(default_factory=list)
    eml_form: str                 = ""   # annotation (compact mode only)

    # layout scratch (set by layout helpers)
    _px: float = field(default=0.0, repr=False, compare=False)
    _py: float = field(default=0.0, repr=False, compare=False)
    _sw: float = field(default=0.0, repr=False, compare=False)

    # ------------------------------------------------------------------
    # ASCII
    # ------------------------------------------------------------------

    def ascii(self, *, _pfx: str = "", _last: bool = True) -> str:
        conn  = "└── " if _last else "├── "
        badge = f"  [{self.eml_form}]" if self.eml_form else ""
        lines = [_pfx + conn + self.label + badge]
        child_pfx = _pfx + ("    " if _last else "│   ")
        for i, c in enumerate(self.children):
            lines.append(c.ascii(_pfx=child_pfx, _last=(i == len(self.children) - 1)))
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.ascii()

    # ------------------------------------------------------------------
    # JSON / dict
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"label": self.label, "kind": self.kind}
        if self.eml_form:
            d["eml_form"] = self.eml_form
        if self.children:
            d["children"] = [c.to_dict() for c in self.children]
        return d

    # ------------------------------------------------------------------
    # SVG
    # ------------------------------------------------------------------

    _NW   = 120
    _NH   = 44
    _HGAP = 16
    _VGAP = 68
    _PAD  = 24

    _COLORS = {
        NodeKind.PRIMITIVE:  ("#D35400", "#FEF0E7"),   # amber — exp / ln
        NodeKind.STRUCTURAL: ("#5D6D7E", "#F2F3F4"),   # slate — add/sub/neg/scale
        NodeKind.COMPOUND:   ("#2E86C1", "#EBF5FB"),   # blue  — sin/cos/…
        NodeKind.SCALAR:     ("#1E8449", "#EAFAF1"),   # green
        NodeKind.VEC:        ("#7D3C98", "#F5EEF8"),   # purple
        NodeKind.PI:         ("#7D3C98", "#F5EEF8"),   # purple
        NodeKind.CONST:      ("#1E8449", "#EAFAF1"),   # green
        NodeKind.UNKNOWN:    ("#7F8C8D", "#F2F3F4"),
    }

    def svg(self, *, max_width: int = 1000) -> str:
        _compute_width(self)
        canvas_w = max(max_width, int(self._sw + 2 * self._PAD))
        _assign_pos(self, x_left=self._PAD, level=0)
        depth    = _depth(self)
        canvas_h = self._PAD * 2 + (depth + 1) * (self._NH + self._VGAP)
        parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" '
            f'width="{canvas_w}" height="{canvas_h}" '
            f'font-family="monospace" viewBox="0 0 {canvas_w} {canvas_h}">',
        ]
        _emit_edges(self, parts)
        _emit_nodes(self, parts)
        parts.append("</svg>")
        return "\n".join(parts)


# ── Layout helpers ────────────────────────────────────────────────────────────

def _compute_width(n: EMLTreeNode) -> float:
    NW, HGAP = EMLTreeNode._NW, EMLTreeNode._HGAP
    if not n.children:
        n._sw = float(NW); return n._sw
    cw   = sum(_compute_width(c) for c in n.children)
    gaps = (len(n.children) - 1) * HGAP
    n._sw = max(float(NW), cw + gaps)
    return n._sw


def _assign_pos(n: EMLTreeNode, x_left: float, level: int) -> None:
    NW, NH, HGAP, VGAP, PAD = (
        EMLTreeNode._NW, EMLTreeNode._NH,
        EMLTreeNode._HGAP, EMLTreeNode._VGAP, EMLTreeNode._PAD,
    )
    n._py = PAD + level * (NH + VGAP)
    n._px = x_left + (n._sw - NW) / 2.0
    cursor = x_left
    for c in n.children:
        _assign_pos(c, cursor, level + 1)
        cursor += c._sw + HGAP


def _depth(n: EMLTreeNode) -> int:
    if not n.children: return 0
    return 1 + max(_depth(c) for c in n.children)


def _emit_edges(n: EMLTreeNode, out: List[str]) -> None:
    NW, NH = EMLTreeNode._NW, EMLTreeNode._NH
    px, py = n._px + NW / 2, n._py + NH
    for c in n.children:
        cx, cy = c._px + NW / 2, c._py
        my = (py + cy) / 2
        out.append(
            f'<path d="M{px:.1f},{py:.1f} C{px:.1f},{my:.1f} '
            f'{cx:.1f},{my:.1f} {cx:.1f},{cy:.1f}" '
            f'stroke="#AAB7B8" stroke-width="1.5" fill="none"/>'
        )
        _emit_edges(c, out)


def _emit_nodes(n: EMLTreeNode, out: List[str]) -> None:
    NW, NH = EMLTreeNode._NW, EMLTreeNode._NH
    stroke, fill = EMLTreeNode._COLORS.get(n.kind, EMLTreeNode._COLORS[NodeKind.UNKNOWN])
    x, y = n._px, n._py
    out.append(
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{NW}" height="{NH}" '
        f'rx="8" ry="8" stroke="{stroke}" stroke-width="2" fill="{fill}"/>'
    )
    label = n.label if len(n.label) <= 16 else n.label[:14] + "…"
    cx = x + NW / 2
    if n.eml_form:
        badge = n.eml_form if len(n.eml_form) <= 20 else n.eml_form[:18] + "…"
        out.append(
            f'<text x="{cx:.1f}" y="{y+NH*0.36:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-weight="bold" font-size="12" '
            f'fill="{stroke}">{_esc(label)}</text>'
        )
        out.append(
            f'<text x="{cx:.1f}" y="{y+NH*0.72:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-size="10" fill="#666">'
            f'{_esc(badge)}</text>'
        )
    else:
        out.append(
            f'<text x="{cx:.1f}" y="{y+NH/2:.1f}" text-anchor="middle" '
            f'dominant-baseline="central" font-weight="bold" font-size="12" '
            f'fill="{stroke}">{_esc(label)}</text>'
        )
    for c in n.children:
        _emit_nodes(c, out)


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# ── AST → EMLTreeNode (with optional EML expansion) ─────────────────────────

def parse_eml_tree(eml_description: str, *, expand_eml: bool = True) -> EMLTreeNode:
    """
    Parse an ``eml_description`` string into an :class:`EMLTreeNode` tree.

    Parameters
    ----------
    eml_description :
        String starting with ``"EML: "``.
    expand_eml :
        If *True* (default) compound ops are expanded to their EML primitive
        form — ``mul(a,b)`` becomes ``exp → add → [ln(a), ln(b)]``.
        If *False* compact ops-level tree is returned (``mul``, ``pow`` etc.
        shown as single nodes with EML-form annotations).

    Example
    -------
    >>> t = parse_eml_tree("EML: ops.mul(eml_vec('A'), eml_vec('B'))")
    >>> print(t.ascii())
    └── exp
        └── add
            ├── ln
            │   └── A
            └── ln
                └── B
    """
    from eml_math.evaluator import EMLEvaluator
    expr = EMLEvaluator._parse(eml_description)
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        return EMLTreeNode(label=f"<parse error: {exc}>", kind=NodeKind.UNKNOWN)
    return _ast_to_node(tree.body, expand_eml=expand_eml)


# ── Primitive node builders ───────────────────────────────────────────────────

def _prim(label: str, *children: EMLTreeNode) -> EMLTreeNode:
    return EMLTreeNode(label=label, kind=NodeKind.PRIMITIVE, children=list(children))

def _struct(label: str, *children: EMLTreeNode) -> EMLTreeNode:
    return EMLTreeNode(label=label, kind=NodeKind.STRUCTURAL, children=list(children))

def _exp(child: EMLTreeNode) -> EMLTreeNode:
    return _prim("exp", child)

def _ln(child: EMLTreeNode) -> EMLTreeNode:
    return _prim("ln", child)

def _add(a: EMLTreeNode, b: EMLTreeNode) -> EMLTreeNode:
    return _struct("add", a, b)

def _sub(a: EMLTreeNode, b: EMLTreeNode) -> EMLTreeNode:
    return _struct("sub", a, b)

def _neg(child: EMLTreeNode) -> EMLTreeNode:
    return _struct("neg", child)

def _scale(n: float, child: EMLTreeNode) -> EMLTreeNode:
    return _struct(f"×{_fmt_num(n)}", child)


# ── Compound-op expansion ─────────────────────────────────────────────────────

def _expand(op: str, args: List[ast.expr], expand_eml: bool) -> EMLTreeNode:
    """Expand a compound ops.* call to its EML primitive tree."""
    def p(a: ast.expr) -> EMLTreeNode:
        return _ast_to_node(a, expand_eml=expand_eml)

    # ── arithmetic ──────────────────────────────────────────────────────
    if op in ("mul", "mul_n") and len(args) == 2:
        return _exp(_add(_ln(p(args[0])), _ln(p(args[1]))))

    if op in ("div", "div_n") and len(args) == 2:
        return _exp(_sub(_ln(p(args[0])), _ln(p(args[1]))))

    if op in ("sqrt", "sqrt_n") and len(args) == 1:
        return _exp(_scale(0.5, _ln(p(args[0]))))

    if op == "sqr" and len(args) == 1:
        return _exp(_scale(2.0, _ln(p(args[0]))))

    if op in ("pow", "pow_fn") and len(args) == 2:
        x = p(args[0])
        n_val = _extract_scalar(args[1])
        if n_val is not None:
            return _exp(_scale(n_val, _ln(x)))
        # symbolic exponent: exp(add(ln(n), ln(x)))
        return _exp(_add(_ln(p(args[1])), _ln(x)))

    if op in ("inv",) and len(args) == 1:
        return _exp(_neg(_ln(p(args[0]))))

    if op in ("neg", "neg_n") and len(args) == 1:
        return _neg(p(args[0]))

    if op in ("add", "add_n") and len(args) == 2:
        return _add(p(args[0]), p(args[1]))

    if op in ("sub", "sub_n") and len(args) == 2:
        return _sub(p(args[0]), p(args[1]))

    # ── EML primitives (already at the base level) ───────────────────────
    if op == "exp" and len(args) == 1:
        return _exp(p(args[0]))

    if op == "ln" and len(args) == 1:
        return _ln(p(args[0]))

    if op == "eml" and len(args) == 2:
        node = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE,
                           children=[p(args[0]), p(args[1])],
                           eml_form="exp(x)−ln(y)")
        return node

    # ── trig — no closed-form EML expansion yet ──────────────────────────
    children = [p(a) for a in args]
    eml_form = EML_EXPANSIONS.get(op, "")
    return EMLTreeNode(label=op, kind=NodeKind.COMPOUND,
                       children=children, eml_form=eml_form)


# ── Main AST walker ───────────────────────────────────────────────────────────

def _ast_to_node(node: ast.expr, *, expand_eml: bool = True) -> EMLTreeNode:  # noqa: C901
    # ── function call ──────────────────────────────────────────────────
    if isinstance(node, ast.Call):
        op = _func_name(node.func)

        if op == "eml_scalar":
            v = _literal_value(node.args[0]) if node.args else "?"
            return EMLTreeNode(label=_fmt_num(v), kind=NodeKind.SCALAR)

        if op == "eml_pi":
            return EMLTreeNode(label="π", kind=NodeKind.PI)

        if op == "eml_vec":
            name = _literal_value(node.args[0]) if node.args else "?"
            return EMLTreeNode(label=str(name), kind=NodeKind.VEC)

        if expand_eml:
            return _expand(op, node.args, expand_eml=True)

        # compact mode — keep the ops label
        children = [_ast_to_node(a, expand_eml=False) for a in node.args]
        if op in _PRIMITIVES:
            return EMLTreeNode(label=op, kind=NodeKind.PRIMITIVE,
                               children=children,
                               eml_form=EML_EXPANSIONS.get(op, ""))
        eml_form = EML_EXPANSIONS.get(op, "")
        return EMLTreeNode(label=op, kind=NodeKind.COMPOUND,
                           children=children, eml_form=eml_form)

    # ── attribute: ops.mul → "mul" (function without call, shouldn't happen) ─
    if isinstance(node, ast.Attribute):
        return EMLTreeNode(label=node.attr, kind=NodeKind.UNKNOWN)

    # ── bare name ─────────────────────────────────────────────────────
    if isinstance(node, ast.Name):
        return EMLTreeNode(label=node.id, kind=NodeKind.VEC)

    # ── constant ──────────────────────────────────────────────────────
    if isinstance(node, ast.Constant):
        v = node.value
        if isinstance(v, (int, float)):
            return EMLTreeNode(label=_fmt_num(v), kind=NodeKind.CONST)
        return EMLTreeNode(label=repr(v), kind=NodeKind.CONST)

    # ── unary negation -x ─────────────────────────────────────────────
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = _ast_to_node(node.operand, expand_eml=expand_eml)
        return EMLTreeNode(label=f"-{inner.label}", kind=inner.kind)

    # ── tuple / list (rare) ───────────────────────────────────────────
    if isinstance(node, (ast.Tuple, ast.List)):
        children = [_ast_to_node(e, expand_eml=expand_eml) for e in node.elts]
        return EMLTreeNode(label="(…)", kind=NodeKind.UNKNOWN, children=children)

    return EMLTreeNode(label=ast.dump(node)[:30], kind=NodeKind.UNKNOWN)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _extract_scalar(node: ast.expr) -> Optional[float]:
    """Return float if node is a numeric literal or eml_scalar(x), else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return float(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        v = _extract_scalar(node.operand)
        return -v if v is not None else None
    if isinstance(node, ast.Call):
        name = _func_name(node.func)
        if name == "eml_scalar" and node.args:
            v = _extract_scalar(node.args[0])
            return v
    return None


def _func_name(node: ast.expr) -> str:
    if isinstance(node, ast.Attribute): return node.attr
    if isinstance(node, ast.Name):      return node.id
    return ast.dump(node)[:20]


def _literal_value(node: ast.expr) -> Any:
    if isinstance(node, ast.Constant): return node.value
    try:    return ast.literal_eval(node)
    except: return ast.dump(node)[:20]  # noqa: E722


def _fmt_num(v: Any) -> str:
    if not isinstance(v, (int, float)): return str(v)
    if v == math.pi:  return "π"
    if v == math.e:   return "e"
    if isinstance(v, int) or (isinstance(v, float) and v == int(v) and abs(v) < 1e6):
        return str(int(v))
    if abs(v) >= 1e6 or (abs(v) < 1e-3 and v != 0):
        return f"{v:.3e}"
    return f"{v:.6g}"
