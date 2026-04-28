"""
Render every entry of eml_math.famous as PNG + PDF + a single HTML viewer.

Run from the EML-Math repo root:
    python examples/famous_gallery.py [output_dir]

Default output dir: ./famous_gallery/
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from eml_math.famous import all_equations, by_category
from eml_math.tree import to_compact
from eml_math.web import get_flow_js


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "famous_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "png").mkdir(exist_ok=True)
    (out_dir / "pdf").mkdir(exist_ok=True)

    eqs = all_equations()
    print(f"Rendering {len(eqs)} famous equations -> {out_dir}/")

    # Per-equation PNG + PDF, in both standard and merged-inputs variants.
    for eq in eqs:
        png      = eq.flow_png(width=720, height=440, merge_inputs=False)
        png_merg = eq.flow_png(width=720, height=440, merge_inputs=True)
        pdf      = eq.flow_pdf(width=720, height=440, merge_inputs=False)
        pdf_merg = eq.flow_pdf(width=720, height=440, merge_inputs=True)
        (out_dir / "png" / f"{eq.name}.png").write_bytes(png)
        (out_dir / "png" / f"{eq.name}_merged.png").write_bytes(png_merg)
        (out_dir / "pdf" / f"{eq.name}.pdf").write_bytes(pdf)
        (out_dir / "pdf" / f"{eq.name}_merged.pdf").write_bytes(pdf_merg)
        print(f"  {eq.category:9s} {eq.name:24s}  PNG {len(png):>6d} / merged {len(png_merg):>6d}")

    # Standalone HTML viewer that re-renders client-side via the bundled JS
    js  = get_flow_js()
    rows = []
    for cat in ("physics", "geometry", "math"):
        cat_eqs = by_category(cat)
        rows.append(f'<h2>{cat.capitalize()}</h2>')
        for eq in cat_eqs:
            tree_json = json.dumps(to_compact(eq.parse()))
            label_json = json.dumps(eq.output)
            rows.append(f"""
              <article>
                <h3>{_esc(eq.title)} <code class="hint">{_esc(eq.description)}</code></h3>
                <p class="meta"><strong>Inputs:</strong> {", ".join(eq.inputs) or "<em>(none)</em>"}
                   &nbsp;&nbsp;<strong>Output:</strong> {eq.output}</p>
                {f'<p class="notes">{_esc(eq.notes)}</p>' if eq.notes else ''}
                <div id="d_{eq.name}"></div>
                <script>document.getElementById("d_{eq.name}").innerHTML =
                    EmlFlow.renderFlowSvg({tree_json},
                        {{ width: 720, height: 380, outputLabel: {label_json} }});</script>
              </article>""")

    html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>EML-Math: famous equations gallery</title>
<style>
  body  {{ font: 16px/1.5 system-ui, sans-serif; max-width: 880px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1    {{ border-bottom: 2px solid #444; padding-bottom: .25em; }}
  h2    {{ margin-top: 2em; color: #444; border-bottom: 1px solid #ccc; }}
  article {{ margin: 1.5em 0 2em; padding: 1em; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }}
  article h3 {{ margin: 0 0 .25em; }}
  .hint {{ font-weight: normal; color: #666; font-size: .85em; }}
  .meta {{ font-size: .85em; color: #555; margin: .25em 0 .5em; }}
  .notes {{ background: #fff8e1; border-left: 3px solid #f5a623; padding: .5em .75em; font-size: .85em; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
</style>
</head><body>
<h1>EML-Math — Famous equations</h1>
<p>Each equation is parsed from a single <code>EML: …</code> string and rendered
client-side as a top-down flow diagram. Inputs are at the top, the output sits
at the bottom; every junction is the binary EML primitive
<code>eml(L, R) = exp(L) − ln(R)</code>.</p>
{"".join(rows)}
<script>{js}</script>
</body></html>
"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\nHTML viewer: {out_dir / 'index.html'}")


def _esc(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


if __name__ == "__main__":
    main()
