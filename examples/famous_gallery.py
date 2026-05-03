"""
Render every entry of eml_math.famous in two layout styles.

Both styles render the same tree with two simplifying conventions on:
constants are collapsed into single SCALAR leaves (each variable-free
subtree → one leaf labelled with its evaluated value), and those leaves
plus all 0/1 sentinels render as short black stubs at their parent
junction. The only "real" leaves of the tree are then the variables.

    famous_gallery/
        formal/        — default columnar layout
        organic/       — branches tilt as they grow up
        index.html     — landing page

Run from the EML-Math repo root:
    python examples/famous_gallery.py [output_dir]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from eml_math.famous import all_equations, by_category
from eml_math.tree import to_compact
from eml_math.web import get_flow_js
from eml_math.flow_layout import (
    to_layout, render_png, render_pdf,
    fit_to_canvas, organic_layout,
)


def _pipe(*fns):
    def composed(L):
        for fn in fns:
            L = fn(L)
        return L
    return composed


# Two clean styles. Both apply constant-collapse + inline-constants at
# tree-build time (handled by `to_layout`); the post-processes only
# adjust geometry.
STYLES = {
    "formal":  fit_to_canvas,
    "organic": _pipe(organic_layout, fit_to_canvas),
}

WIDTH, HEIGHT = 720, 440


def _gen_layout(eq):
    """Build the layout dict for a famous equation. Constants are
    collapsed into single SCALAR leaves and rendered as inline stubs."""
    return to_layout(
        eq.parse(),
        width=WIDTH, height=HEIGHT,
        output_label=eq.output,
        collapse_constants=True,
        inline_constants=True,
    )


def main() -> None:
    out_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "famous_gallery")
    out_dir.mkdir(parents=True, exist_ok=True)
    for style in STYLES:
        (out_dir / style).mkdir(exist_ok=True)
        (out_dir / style / "png").mkdir(exist_ok=True)
        (out_dir / style / "pdf").mkdir(exist_ok=True)

    eqs = all_equations()
    print(f"Rendering {len(eqs)} equations x {len(STYLES)} styles -> {out_dir}/")

    for eq in eqs:
        layout = _gen_layout(eq)
        for style, post in STYLES.items():
            L = post(layout) if post else layout
            png = render_png(L)
            pdf = render_pdf(L)
            (out_dir / style / "png" / f"{eq.name}.png").write_bytes(png)
            (out_dir / style / "pdf" / f"{eq.name}.pdf").write_bytes(pdf)
        print(f"  {eq.category:9s} {eq.name}")

    js = get_flow_js()
    for style, post in STYLES.items():
        rows = []
        for cat in ("physics", "geometry", "math"):
            rows.append(f"<h2>{cat.capitalize()}</h2>")
            for eq in by_category(cat):
                rows.append(f'''
                  <article>
                    <h3>{_esc(eq.title)} <code class="hint">{_esc(eq.description)}</code></h3>
                    <p class="meta"><strong>Inputs:</strong> {", ".join(eq.inputs) or "<em>(none)</em>"}
                       &nbsp;&nbsp;<strong>Output:</strong> {eq.output}</p>
                    <p class="meta"><strong>Style:</strong> {style}
                       &nbsp;<a href="png/{eq.name}.png">PNG</a>
                       &nbsp;<a href="pdf/{eq.name}.pdf">PDF</a></p>
                    <img src="png/{eq.name}.png" alt="{eq.name} ({style})" style="max-width:100%">
                  </article>''')
        html = f"""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8"><title>EML-Math gallery — {style}</title>
<style>
  body {{ font: 16px/1.5 system-ui, sans-serif; max-width: 880px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1   {{ border-bottom: 2px solid #444; padding-bottom: .25em; }}
  h2   {{ margin-top: 2em; color: #444; border-bottom: 1px solid #ccc; }}
  article {{ margin: 1.5em 0 2em; padding: 1em; border: 1px solid #e0e0e0; border-radius: 8px; background: #fafafa; }}
  article h3 {{ margin: 0 0 .25em; }}
  .hint {{ font-weight: normal; color: #666; font-size: .85em; }}
  .meta {{ font-size: .85em; color: #555; margin: .25em 0; }}
  code {{ background: #f0f0f0; padding: 1px 5px; border-radius: 3px; }}
  nav  {{ margin-bottom: 2em; padding: .5em 1em; background: #fafafa; border: 1px solid #ddd; border-radius: 6px; }}
</style>
</head><body>
<h1>EML-Math — {style} style</h1>
<nav>Other styles: {' &middot; '.join(f'<a href="../{s}/index.html">{s}</a>' for s in STYLES if s != style)}</nav>
{"".join(rows)}
</body></html>
"""
        (out_dir / style / "index.html").write_text(html, encoding="utf-8")

    top_index = "<!doctype html><html><head><meta charset='utf-8'><title>EML-Math gallery</title></head>"
    top_index += "<body style='font:16px/1.5 system-ui;max-width:680px;margin:2em auto;padding:0 1em'>"
    top_index += "<h1>EML-Math famous-equations gallery</h1><ul>"
    for s in STYLES:
        top_index += f"<li><a href='{s}/index.html'>{s}</a></li>"
    top_index += "</ul></body></html>"
    (out_dir / "index.html").write_text(top_index, encoding="utf-8")

    print(f"\nIndex: {out_dir / 'index.html'}")


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


if __name__ == "__main__":
    main()
