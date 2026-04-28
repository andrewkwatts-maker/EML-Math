# EML Flow Diagrams

The `eml_math.flow` module renders an EML expression tree as a top-down
**flow diagram** — inputs at the top, an output at the bottom, with every
internal junction representing the binary primitive
**eml(L, R) = exp(L) − ln(R)**.

Because every join is the *same* operator there are no internal labels —
the structure of the tree itself communicates the formula.  Branch colours
are inherited from each leaf input (one palette colour per input) and
blended at every junction so related sub-expressions share visual tone all
the way down to the output.

## Quick start

```python
from eml_math.tree import parse_eml_tree

# Take any EML expression string
tree = parse_eml_tree(
    "EML: ops.mul(eml_vec('A'), ops.pow(eml_vec('lambda'), eml_scalar(2.0)))",
    expand_eml=False,
)

# Render to inline SVG — embed directly in any HTML page
svg = tree.flow_svg(width=720, height=420, output_label="V_cb")

# Or to a PNG file (uses cairosvg if installed, else Pillow)
png = tree.flow_png(width=720, height=420, output_label="V_cb")
open("V_cb.png", "wb").write(png)

# Or to a self-contained HTML <div> snippet
html = tree.flow_html(output_label="V_cb")
```

## Reading the diagram

```
A_wolfenstein     lambda_wolfenstein     2          ←  inputs (each in its own colour)
     |                |   |
     |                 \ /
     |              [junction]                     ←  binary join (no label — implicit eml)
     |                |
      \              /
        [junction]                                 ←  output junction
            |
          V_cb                                     ←  output label
```

* **Left branch** of every junction is the *exp-side* (L) input;
  **right branch** is the *ln-side* (R) input.
* Junction colour = average of the two child branch colours.
* The output label is whatever you pass as `output_label` (default `"Out"`).

## Binarisation

Real-world expressions sometimes contain unary operators (`sqrt`, `exp`,
`sin`, …) or n-ary ones (`std(a, b, c)`, `sum_n(...)`).  The renderer
**binarises** the tree before drawing so every junction is a true binary
merge:

* a unary internal node is **collapsed** — its single child becomes the
  result, eliminating "1-into-1" passthrough junctions;
* an n-ary node is **left-folded** into nested binary, so `std(a, b, c)`
  draws as `[[a, b], c]` with two binary merges.

For the most uniform diagrams use `pure_eml=True` when parsing — every
internal node is then guaranteed to be the binary `eml` primitive
already, with `⊥` (the bottom sentinel where exp(⊥)=0) and `1` as the
only special leaves.

## In-browser rendering

The package ships a UMD-style JavaScript port of the renderer at
`eml_math/web/eml_flow.js`.  Get the path or the contents from Python:

```python
from eml_math.web import FLOW_JS_PATH, get_flow_js
print(FLOW_JS_PATH)             # absolute path to the .js file
script_source = get_flow_js()   # for embedding in a generated page
```

In the browser, render from the **compact** tree form (see below):

```html
<script src="path/to/eml_flow.js"></script>
<div id="diagram"></div>
<script>
  const treeArr = ["mul","c",["a","v"],["b","v"]];   // from to_compact()
  document.getElementById('diagram').innerHTML =
      EmlFlow.renderFlowSvg(treeArr, { outputLabel: "V_cb" });
</script>
```

## Compact tree serialisation

To save bandwidth when shipping many trees to the browser, use
`to_compact` / `from_compact`:

```python
from eml_math.tree import to_compact, from_compact

arr = tree.to_compact()
# leaf     = [label, kind_char]
# internal = [label, kind_char, child1, child2, …]

restored = from_compact(arr)
```

Compact form is roughly **7× smaller than the dict form** in JSON, and the
JavaScript renderer accepts it directly.

## Named symbols

Common irrational constants have ready-made EML constructions in
`eml_math.symbols`.  Use them in place of decimal literals so diagrams
stay symbolic:

```python
from eml_math.symbols import lookup, construct

print(lookup("e").value)        # 2.718281828459045
print(lookup("phi").value)      # 1.618033988749…

e_tree = construct("e")         # eml(1, 1)  (== exp(1))
phi_tree = construct("phi")     # (1 + √5) / 2 as an EML chain
```

The built-in registry includes `e`, `pi`, `tau`, `gamma_em`, `phi`,
`sqrt2`, `sqrt3`, `sqrt5`, `ln2`.  Add your own with `register(Symbol(…))`.

Symbols whose value has no closed form in the elementary EML primitives
(`pi`, `tau`, `gamma_em`) carry `tree=None` and are rendered as a single
labelled leaf.

## Customisation

All rendering options have sensible defaults but can be overridden:

```python
tree.flow_svg(
    width=1000, height=600,
    output_label="my_param",
    palette=[(255,0,0), (0,128,255), (0,200,0)],   # custom colours
    label_font_size=20, output_font_size=24,
    edge_width=4, junction_radius=5,
    background="#fafafa",
    show_output_label=True,
)
```

## See also

* `eml_math.tree.parse_eml_tree` — parse an `eml_description` string
* `eml_math.tree.EMLTreeNode.to_latex` — render the same tree as LaTeX
* `eml_math.tree.EMLTreeNode.svg` — the *operator-tree* SVG (boxes,
  not flow)
* `eml_math.evaluator.EMLEvaluator` — numerically evaluate the same
  expression against a parameter context
