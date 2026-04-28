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
| `flow` + `flow_layout` | SVG / PNG / PDF / HTML flow-diagram renderer |
| `web` | Bundled `eml_flow.js` UMD bundle for browser-side rendering |

> **Algebras and physics are now in [eml-spectral](https://github.com/andrewkwatts-maker/EML-Spectral)** —
> the sister package. EML-tree representations of Clifford algebras,
> octonions, exceptional algebras (E7/E8/Freudenthal), Lorentz-invariant
> spacetime ops, and named GR metrics all live there. v1.2.0 split them
> out so the `eml-math` core has zero physics narrative.

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
pip install eml-spectral           # transitively pulls in eml-math
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

## Quickstart — flow-diagram renderer

```python
from eml_math import EMLPoint, flow_svg, flow_png

# c² = a² + b² — Pythagoras
expr = "sqrt(add(pow(a, 2), pow(b, 2)))"

flow_png(expr, "pythagoras.png", style="organic", width=1200, height=900)
flow_svg(expr, "pythagoras.svg", style="formal")
```

Available styles:

| Style | Path | Joint | Look |
|---|---|---|---|
| `formal` | continuous | continuous | symmetric tree, top-down |
| `organic` | continuous | continuous | branching, depth-aware angle |
| `gentle` | continuous (large bend) | continuous | long flowing sweeps |
| `tree` | continuous (small bend) | continuous | balanced upright |

The **layout-intermediate JSON** lets you inspect or post-process the
geometry before rendering:

```python
from eml_math import to_layout, render_layout_svg, organic_layout

layout = to_layout(expr, width=1200, height=900)
layout = organic_layout(layout, branch_jitter=0.15, trunk_pull=0.3)
svg = render_layout_svg(layout)
```

The bundled UMD bundle (`eml_math.web.FLOW_JS_PATH`) renders the same
JSON in a browser:

```html
<script src="eml_flow.js"></script>
<script>
  const layout = /* JSON from to_layout(...) */;
  document.body.innerHTML = EMLFlow.renderSVG(layout);
</script>
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
├─ rust/eml_core/                   # Rust accelerator (PyO3 module)
├─ c_api/                           # C/C++/Rust shared-library bindings
└─ tests/                           # pytest suite (~1500 tests)
```

---

## License

MIT, © Andrew K Watts.
