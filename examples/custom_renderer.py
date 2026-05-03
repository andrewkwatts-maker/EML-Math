"""
Bring-your-own renderer for the abstracted EML render pipeline.

Three things this script demonstrates:

1. The **raw bifurcating-formula JSON** has zero positional fields —
   ``EMLTreeNode.to_dict()`` is purely structural.
2. The **layout dict** (output of ``compute_layout``) is the renderer
   contract — anything that can read ``{"nodes": [...], "edges": [...]}``
   can render an EML formula.
3. A **custom renderer** drops in via :func:`eml_math.render.register`.

Run::

    python examples/custom_renderer.py
"""
from __future__ import annotations

import json
from pathlib import Path

from eml_math import parse_eml_tree
from eml_math.render import (
    SVGRenderer,
    compute_layout,
    register,
    render_with,
)


# ── A custom renderer: dump as a 1-line ASCII tree ───────────────────────────

class AsciiTreeRenderer:
    """Pretty-print the layout as nested ASCII boxes — no SVG, no fonts.

    Implements the :class:`eml_math.render.Renderer` protocol implicitly by
    providing a ``.render(layout, **opts) -> str`` method.
    """

    def render(self, layout, **opts) -> str:
        nodes = {n["id"]: n for n in layout["nodes"]}
        children: dict[str, list[str]] = {n["id"]: [] for n in layout["nodes"]}
        for e in layout["edges"]:
            children[e["to"]].append(e["from"])
        # Find root: node with no out-edges as a child.
        is_child = {e["from"] for e in layout["edges"]}
        roots = [nid for nid in nodes if nid not in is_child]

        def _draw(nid: str, depth: int) -> list[str]:
            n = nodes[nid]
            head = f"{'  ' * depth}- {n['label']} ({n['kind']})"
            kids = children.get(nid, [])
            return [head, *(line for k in kids for line in _draw(k, depth + 1))]

        out: list[str] = []
        for r in roots:
            out.extend(_draw(r, 0))
        return "\n".join(out)


# ── A custom renderer: emit Graphviz DOT ─────────────────────────────────────

class DotRenderer:
    """Emit a Graphviz DOT description of the layout."""

    def render(self, layout, **opts) -> str:
        lines = ["digraph G {", "  node [shape=circle, style=filled];"]
        for n in layout["nodes"]:
            r, g, b = n["color"]
            color = f'"#{r:02x}{g:02x}{b:02x}"'
            lines.append(
                f'  {n["id"]} [label="{n["label"]}", '
                f'fillcolor={color}, fontsize=10];'
            )
        for e in layout["edges"]:
            r, g, b = e["color"]
            color = f'"#{r:02x}{g:02x}{b:02x}"'
            lines.append(f'  {e["to"]} -> {e["from"]} [color={color}];')
        lines.append("}")
        return "\n".join(lines)


def main() -> None:
    # 1) Build a small EML tree: mul(add(a, b), c).
    desc = "EML: ops.mul(ops.add(eml_vec('a'), eml_vec('b')), eml_vec('c'))"
    tree = parse_eml_tree(desc, expand_eml=False)

    # 2) Show the raw bifurcating-formula JSON has zero positional data.
    raw = tree.to_dict()
    print("── raw formula JSON (structure only) ──")
    print(json.dumps(raw, indent=2))
    raw_text = json.dumps(raw)
    for forbidden in ('"x":', '"y":', '"color":', '"canvas":', '"width":'):
        assert forbidden not in raw_text, f"unexpected layout field: {forbidden}"
    assert raw["schema"] == "eml-formula/v1"

    # 3) Compute the layout dict (the renderer contract).
    layout = compute_layout(raw, edge_style="curve")
    print("\n── layout dict (renderer contract) ──")
    print(f"  schema={layout['schema']}, canvas={layout['canvas']}")
    print(f"  nodes={len(layout['nodes'])}, edges={len(layout['edges'])}")

    # 4) Built-in SVG renderer.
    svg = SVGRenderer().render(layout)
    print(f"\n── built-in SVG renderer: {len(svg)} chars ──")

    # 5) Register two custom renderers and use them.
    register("ascii-tree", AsciiTreeRenderer())
    register("dot", DotRenderer())

    print("\n── custom AsciiTreeRenderer ──")
    print(render_with("ascii-tree", layout))

    print("\n── custom DotRenderer ──")
    print(render_with("dot", layout))

    # 6) Same render via the EMLTreeNode shortcut.
    svg2 = tree.render("svg", layout_opts={"edge_style": "spline"})
    print(f"\n── tree.render('svg', edge_style='spline'): {len(svg2)} chars ──")

    # 7) Save outputs next to this script.
    out_dir = Path(__file__).resolve().parent
    (out_dir / "custom_renderer_out.svg").write_text(svg, encoding="utf-8")
    (out_dir / "custom_renderer_out.dot").write_text(
        render_with("dot", layout), encoding="utf-8"
    )
    print(f"\nWrote: {out_dir / 'custom_renderer_out.svg'}")
    print(f"Wrote: {out_dir / 'custom_renderer_out.dot'}")


if __name__ == "__main__":
    main()
