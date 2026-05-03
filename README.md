# EML-Math

**EML Mathematics** — a universal real-valued foundation for elementary mathematics, built from a single binary operator.

[![PyPI](https://img.shields.io/pypi/v/eml-math)](https://pypi.org/project/eml-math/)
[![Python 3.11+](https://img.shields.io/pypi/pyversions/eml-math)](https://pypi.org/project/eml-math/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Repo:** <https://github.com/andrewkwatts-maker/EML-Math>

Created by **Andrew K Watts**. Based on the EML Sheffer operator established by
Andrzej Odrzywolek: [arXiv:2603.21852v2](https://arxiv.org/html/2603.21852v2)
(CC BY 4.0).

---

## The core idea

A single binary operator generates every elementary function:

```
eml(x, y) = exp(x) − ln(y)
```

This is the **EML Sheffer operator** — the continuous analog of the NAND
gate for Boolean logic. From it the 36 standard elementary functions
(`+`, `−`, `×`, `/`, `exp`, `ln`, `sin`, `cos`, `tan`, `π`, `e`, …) can
all be reconstructed as composed expression trees.

`EMLPoint` is the operator's computation node — simultaneously a
mathematical state and a composable expression-tree leaf:

```python
from eml_math import EMLPoint
import math

EMLPoint(1, 1).tension()                                  # e   = eml(1, 1)
EMLPoint(2, 1).tension()                                  # exp(2)
EMLPoint(1, EMLPoint(EMLPoint(1, math.e), 1)).tension()   # ln(e) = 1.0
```

---

## What's in v1.2.0 — the slim core

`eml-math` v1.2.0 is the **pure-EML universal-math toolkit**: the
operator, expression trees, symbolic regression, the elementary-function
operator library, the famous-equations registry, and the flow-diagram
renderer. Nothing else.

| Module | Purpose |
|---|---|
| `EMLPoint`, `_VarNode` | The EML node, with variable-leaf support for symbolic work |
| `tree` | Expression-tree parser, renderer, JSON-array compact form |
| `operators` | The 36 elementary functions as ready-made EML trees |
| `evaluator` | Parse and evaluate EML formula strings |
| `symbols` | Named-symbol registry (e, π, φ, √2, …) |
| `discover` | `compress`, `recognize`, `Searcher` — symbolic regression |
| `famous` | Registered classic equations (Pythagoras, Euler, Einstein, …) |
| `flow` + `flow_layout` | Legacy SVG / PNG / PDF / HTML flow-diagram renderer |
| `render` | New abstracted renderer — raw JSON → layout dict → pluggable Renderer (`SVGRenderer`, `HTMLRenderer`, `PNGRenderer`, `PDFRenderer`, BYO). Three edge styles (`straight` · `curve` · `spline`), Reingold-Tilford tidy-tree layout, MathJax/MathML output via `decompress(r, fmt='mathjax')`. |
| `web` | Bundled `eml_flow.js` UMD bundle for browser-side rendering |

> **Algebras and physics are now in [eml-spectral](https://github.com/andrewkwatts-maker/EML-Spectral)** —
> the sister package, **v1.0.0 release**. EML-tree representations of Clifford
> algebras, octonions, exceptional algebras (E7/E8/Freudenthal), Lorentz-invariant
> spacetime ops, named GR metrics, and the spectral-flow operator Φ all live there.
> Same zero-deps philosophy, optional Rust acceleration, optional C API.
>
> ```bash
> pip install eml-spectral   # transitively installs eml-math >= 1.2.0
> ```

---

## Installation

```bash
pip install eml-math               # core
pip install eml-math[ext]          # + numpy, sympy
pip install eml-math[precision]    # + mpmath
pip install eml-math[dev]          # + pytest, ruff, mypy
```

For the algebras / physics layer:

```bash
pip install eml-spectral
# transitively pulls in eml-math >= 1.2.0
```

---

## Quickstart — symbolic regression

`Searcher` finds an EML expression that matches a target numeric value:

```python
from eml_math import Searcher

s = Searcher(target=2.71828)
result = s.search()
print(result.formula)      # 'eml(1, 1)'   (i.e. e)
```

`compress` and `recognize` go in the other direction:

```python
from eml_math import compress, recognize

print(recognize(3.14159))         # ('π', 3.141592653589793, 0.0)
print(compress("exp(x) - ln(y)")) # SearchResult: matches the EML primitive itself
```

---

## Generating equation graphs (SVG / PNG / PDF / HTML)

Every formula in the famous-equations registry can be rendered to
SVG / PNG / PDF / HTML directly. The same primitives work on any
EML expression you build.

### One-liner from a famous equation

```python
from eml_math.famous import get

einstein = get("einstein_e_mc2")

# Bytes → write yourself
open("einstein.svg", "w", encoding="utf-8").write(einstein.flow_svg(width=900, height=600))
open("einstein.png", "wb").write(einstein.flow_png(width=900, height=600))
open("einstein.pdf", "wb").write(einstein.flow_pdf(width=900, height=600))

# Self-contained interactive HTML (UMD bundle inlined)
open("einstein.html", "w", encoding="utf-8").write(einstein.parse().flow_html(width=900, height=600))
```

### Direct from an EML expression

```python
from eml_math import parse_eml_tree

tree = parse_eml_tree(
    "EML: ops.sqrt(ops.add(ops.pow(eml_vec('a'), eml_scalar(2.0)), "
    "ops.pow(eml_vec('b'), eml_scalar(2.0))))",
    pure_eml=True,
)

open("pythagoras.svg", "w", encoding="utf-8").write(tree.flow_svg(width=1200, height=900))
open("pythagoras.png", "wb").write(tree.flow_png(width=1200, height=900))
```

### Render-time options

Useful kwargs accepted by `flow_svg` / `flow_png` / `flow_pdf`:

| Argument | Default | Effect |
|---|---|---|
| `width`, `height` | `800` × `600` | SVG / PNG canvas in px |
| `direction` | `"down"` | growth direction: `"down"` · `"up"` · `"left"` · `"right"` |
| `auto_height` | `True` | grow the canvas vertically for deep trees instead of cropping |
| `min_layer_height` | `38.0` | minimum vertical gap per tree level |
| `palette` | built-in | sequence of `(r, g, b)` colours for variable inputs |
| `output_label` | `"Out"` | label drawn at the root of the tree |
| `show_output_label` | `True` | hide the root label by passing `False` |
| `inline_constants` | `False` | render small constants (`2`, `0.5`) on the edge instead of as a leaf |
| `merge_inputs` | `False` | combine repeated variable leaves into a single fan-out node |
| `expand_symbols` | `False` | expand named symbols (π, φ, …) to their pure-EML form |
| `edge_width` | `3.0` | stroke width for edges |
| `junction_radius` | `4.0` | dot radius at each internal node |
| `label_font_size` / `output_font_size` | `18` / `22` | label sizes |
| `background` | `None` | SVG background colour (`None` = transparent) |

For a different *visual style*, post-process the layout JSON before
rendering — see the next section.

### Layout-intermediate JSON pipeline (this is how styles are applied)

For finer control — or to render the same geometry from JavaScript /
post-process it programmatically — go through the layout-intermediate
form. `to_layout` returns a JSON-serialisable dict; the post-processes
return a transformed copy; `render_layout_svg` / `render_layout_png` /
`render_layout_pdf` produce final output.

The two named styles in the gallery are post-process pipelines applied
to the same `to_layout(tree)` output:

| Style | Post-process pipeline |
|---|---|
| `formal` | `fit_to_canvas(L)` — symmetric, top-down, the default |
| `organic` | `organic_layout(L, branch_angle=24, length_scale=44, length_decay=0.97, min_length=24)` → `fit_to_canvas` — branching, tree-like; `min_length` floor stops deep leaves clumping |

```python
import json
from eml_math import (
    to_layout, render_layout_svg, render_layout_png, render_layout_pdf,
    organic_layout, fit_to_canvas,
)

layout = to_layout(tree, width=1200, height=900)

# Apply organic style
layout = organic_layout(layout,
                        branch_angle=24.0, length_scale=44.0,
                        length_decay=0.97, min_length=24.0,
                        branch_jitter=0.12, trunk_pull=0.35,
                        balance="subtree_size")
layout = fit_to_canvas(layout, margin=20)

# Inspect / save the geometry before rendering
with open("pythagoras_layout.json", "w", encoding="utf-8") as f:
    json.dump(layout, f, indent=2)

# Render to whatever you need
open("pythagoras.svg", "w", encoding="utf-8").write(render_layout_svg(layout))
open("pythagoras.png", "wb").write(render_layout_png(layout))
open("pythagoras.pdf", "wb").write(render_layout_pdf(layout))
```

Knobs on `organic_layout` worth knowing:

- `length_decay` (default `0.97`) — how aggressively branches shrink
  per generation. Closer to `1.0` keeps deep leaves spaciously apart.
- `min_length` (default `22`) — floor on per-generation branch length.
  Even after `length_decay` reduces the geometric length, no branch
  shrinks below this. Set to `0` for the pure geometric decay look.
- `trunk_pull` (default `0.35`) — how strongly each child's growing
  direction is pulled back toward the global trunk axis. Prevents long
  chains from curling into a tight spiral.
- `balance` (default `"subtree_size"`) — assigns the larger subtree to
  the outside of each fork so the tree visually balances.

### Compact JSON for the tree itself

The expression tree (independent of layout) round-trips through a tiny
JSON-friendly array. Useful for storage, transport, or rendering on
the JS side without re-parsing the EML string:

```python
import json
from eml_math import to_compact, from_compact

compact = to_compact(tree)             # list of nested arrays
encoded = json.dumps(compact)          # plain JSON, ~7× smaller than the dict form
rebuilt = from_compact(json.loads(encoded))   # exact structural round-trip
```

### Browser-side rendering (`eml_flow.js`)

`eml_math.web` ships a UMD bundle that renders the same layout JSON in
the browser. Two ways to use it:

```python
from eml_math import flow_html, get_flow_js, FLOW_JS_PATH

# Self-contained HTML (the JS bundle is inlined — open the file, no server needed)
open("einstein.html", "w", encoding="utf-8").write(
    tree.flow_html(width=900, height=600)
)

# Or grab the bundle to host yourself
open("static/eml_flow.js", "w", encoding="utf-8").write(get_flow_js())
print("Bundle path on disk:", FLOW_JS_PATH)
```

In your own page:

```html
<script src="eml_flow.js"></script>
<script>
  const layout = /* JSON dumped from to_layout(...) */;
  document.body.innerHTML = EMLFlow.renderSVG(layout);
</script>
```

### Batch generation — the famous-equations gallery

To regenerate every equation in the registry across all styles
(produces `formal/`, `gentle/`, `organic/`, `tree/` folders with PNG +
PDF for each, plus a self-contained `index.html`):

```bash
python examples/famous_gallery.py [output_dir]   # default: ./famous_gallery
```

Output structure:

```
famous_gallery/
├── index.html        # browse every style of every equation
├── formal/
│   ├── png/einstein_e_mc2.png
│   └── pdf/einstein_e_mc2.pdf
├── organic/   (same layout)
├── gentle/    (same layout)
└── tree/      (same layout)
```

---

## Famous-equations registry

```python
from eml_math import all_famous_equations, get_famous

print([eq.name for eq in all_famous_equations()])
# ['Pythagoras', 'Euler identity', 'E = mc²', 'Schrödinger', ...]

einstein = get_famous("einstein_e_mc2")
print(einstein.eml_formula)    # the EML-tree form
```

---

## Rust accelerator (optional)

The wheel bundles a Rust extension exposing Rayon-parallel batch
operators (`exp_n`, `ln_n`, `add_n`, `mul_n`, `sin_n`, `tension_n`, …):

```python
from eml_math.eml_core import tension_n
import numpy as np

xs = np.linspace(0, 5, 1_000_000)
ys = np.ones_like(xs)
out = tension_n(xs.tolist(), ys.tolist())   # parallel
```

A C/C++/Rust shared-library API lives under [`c_api/`](c_api/) for
embedding the operator into other languages. Build it with:

```bash
cargo build --release -p eml_c_api
```

---

## Project layout

```
eml-math/                           # this repo
├─ src/eml_math/                    # Python sources (the slim core)
│  ├─ render/                       # Abstracted renderer pipeline
│  │  ├─ layout.py                  # Reingold-Tilford tidy-tree
│  │  ├─ edges.py                   # straight / curve / spline path generators
│  │  ├─ palette.py                 # palette helpers
│  │  └─ renderers/                 # SVG, HTML, PNG, PDF — pluggable Renderer protocol
│  └─ ...
├─ rust/eml_core/                   # Rust accelerator (PyO3 module)
├─ c_api/                           # C/C++/Rust shared-library bindings
├─ docs/                            # Static-HTML site (index/api/guide/concepts)
└─ tests/                           # pytest suite (2077 tests, 0 required deps)
```

---

## License

MIT, © Andrew K Watts.
