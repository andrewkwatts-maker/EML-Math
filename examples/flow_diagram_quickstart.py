"""
Quickstart for the EML flow-diagram renderer.

Run this script to produce three artefacts in the current directory:
    flow_demo.svg   — inline SVG, embeddable in any HTML page
    flow_demo.png   — rasterised PNG (uses cairosvg if installed, else Pillow)
    flow_demo.html  — standalone HTML page using the bundled JS renderer
"""
from pathlib import Path

from eml_math.tree import parse_eml_tree, to_compact
from eml_math.web import get_flow_js


def main() -> None:
    out_dir = Path.cwd()

    # 1. Parse an EML expression — this could be any formula.
    desc = (
        "EML: ops.mul(eml_vec('A'), "
        "ops.pow(eml_vec('lambda'), eml_scalar(2.0)))"
    )
    tree = parse_eml_tree(desc, expand_eml=False)
    output_name = "V_cb"

    # 2. SVG — fastest path; drop the string into a page.
    svg = tree.flow_svg(width=720, height=420, output_label=output_name)
    (out_dir / "flow_demo.svg").write_text(svg, encoding="utf-8")

    # 3. PNG — rasterised at 2× scale for crisp display.
    png = tree.flow_png(width=720, height=420, output_label=output_name)
    (out_dir / "flow_demo.png").write_bytes(png)

    # 4. Standalone HTML using the shipped JS renderer.
    #    The compact tree is shipped as JSON so the JS can re-render at
    #    any size without re-fetching from the server.
    js = get_flow_js()
    compact = to_compact(tree)
    import json
    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>EML flow demo</title>
<style>body{{font-family:sans-serif;max-width:840px;margin:2em auto;padding:0 1em}}</style>
</head><body>
<h1>EML flow diagram (rendered in-browser)</h1>
<p>Tree shipped as compact JSON, rendered by <code>EmlFlow.renderFlowSvg</code>.</p>
<div id="diagram"></div>
<script>{js}</script>
<script>
  const tree = {json.dumps(compact)};
  document.getElementById('diagram').innerHTML =
      EmlFlow.renderFlowSvg(tree, {{
        width: 720, height: 420, outputLabel: {json.dumps(output_name)},
      }});
</script>
</body></html>
"""
    (out_dir / "flow_demo.html").write_text(html, encoding="utf-8")

    print("Wrote:")
    print(f"  {out_dir / 'flow_demo.svg'}")
    print(f"  {out_dir / 'flow_demo.png'}  ({len(png):,} bytes)")
    print(f"  {out_dir / 'flow_demo.html'}  (open in any browser)")


if __name__ == "__main__":
    main()
