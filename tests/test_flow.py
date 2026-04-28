"""
Tests for eml_math.flow — flow-diagram renderer (SVG, HTML, PNG).
"""
import pytest

from eml_math.tree import parse_eml_tree, EMLTreeNode, NodeKind
from eml_math.flow import (
    flow_svg, flow_html, flow_png, DEFAULT_PALETTE,
    _collect_leaves, _height, _layout, _rgb_hex, _binarize,
)


# ---------------------------------------------------------------------------
# Layout helpers
# ---------------------------------------------------------------------------

class TestLayoutHelpers:
    def test_collect_leaves_single(self):
        n = EMLTreeNode(label="x", kind=NodeKind.VEC)
        assert _collect_leaves(n) == [n]

    def test_collect_leaves_left_to_right(self):
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        c = EMLTreeNode(label="c", kind=NodeKind.VEC)
        root = EMLTreeNode(
            label="eml", kind=NodeKind.PRIMITIVE,
            children=[
                EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[a, b]),
                c,
            ],
        )
        assert [l.label for l in _collect_leaves(root)] == ["a", "b", "c"]

    def test_height_leaf(self):
        n = EMLTreeNode(label="x", kind=NodeKind.VEC)
        assert _height(n) == 0

    def test_height_recursive(self):
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        node = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[a, b])
        assert _height(node) == 1
        deeper = EMLTreeNode(
            label="eml", kind=NodeKind.PRIMITIVE,
            children=[node, EMLTreeNode(label="c", kind=NodeKind.VEC)],
        )
        assert _height(deeper) == 2

    def test_layout_assigns_xy_and_color(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        _, leaves = _layout(
            t, width=400, height=300, margin_top=40, margin_bottom=40,
            margin_lr=40, palette=DEFAULT_PALETTE,
        )
        # every leaf gets _fx, _fy, _fcolor
        for l in leaves:
            assert hasattr(l, "_fx")
            assert hasattr(l, "_fy")
            assert hasattr(l, "_fcolor")
            assert 0 <= l._fx <= 400
            assert 0 <= l._fy <= 300


class TestBinarize:
    """The flow renderer treats the diagram as a binary tree.
    _binarize() collapses unary internals and left-folds n-ary."""

    def test_leaf_unchanged(self):
        n = EMLTreeNode(label="x", kind=NodeKind.VEC)
        assert _binarize(n) is n

    def test_unary_collapses(self):
        # exp(x) — the exp node should be skipped, leaving just the leaf x
        leaf = EMLTreeNode(label="x", kind=NodeKind.VEC)
        unary = EMLTreeNode(label="exp", kind=NodeKind.PRIMITIVE, children=[leaf])
        result = _binarize(unary)
        assert result is leaf

    def test_unary_chain_collapses(self):
        # exp(neg(x)) — both unaries collapse, leaving just x
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        neg = EMLTreeNode(label="neg", kind=NodeKind.STRUCTURAL, children=[x])
        exp = EMLTreeNode(label="exp", kind=NodeKind.PRIMITIVE, children=[neg])
        assert _binarize(exp) is x

    def test_binary_passthrough(self):
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        root = EMLTreeNode(label="mul", kind=NodeKind.COMPOUND, children=[a, b])
        result = _binarize(root)
        assert len(result.children) == 2

    def test_nary_left_folds(self):
        # std(a, b, c) → [[a, b], c] — N-1 binary joins
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        c = EMLTreeNode(label="c", kind=NodeKind.VEC)
        root = EMLTreeNode(label="std", kind=NodeKind.COMPOUND, children=[a, b, c])
        result = _binarize(root)
        assert len(result.children) == 2
        # left child should itself be a binary node containing a and b
        left = result.children[0]
        assert len(left.children) == 2
        assert left.children[0].label == "a"
        assert left.children[1].label == "b"
        assert result.children[1].label == "c"

    def test_unary_inside_binary_collapses(self):
        # mul(sqrt(x), y) — sqrt collapses, mul becomes (x, y)
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        y = EMLTreeNode(label="y", kind=NodeKind.VEC)
        sqrt = EMLTreeNode(label="sqrt", kind=NodeKind.COMPOUND, children=[x])
        mul = EMLTreeNode(label="mul", kind=NodeKind.COMPOUND, children=[sqrt, y])
        result = _binarize(mul)
        assert [c.label for c in result.children] == ["x", "y"]


class TestRgbHex:
    def test_basic_colors(self):
        assert _rgb_hex((255, 0, 0)) == "#FF0000"
        assert _rgb_hex((0, 255, 0)) == "#00FF00"
        assert _rgb_hex((0, 0, 255)) == "#0000FF"

    def test_clamps(self):
        assert _rgb_hex((300, -10, 128)) == "#FF0080"

    def test_floats_round(self):
        assert _rgb_hex((127.4, 127.6, 127.5)) == "#7F8080"


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------

class TestFlowSVG:
    def test_returns_svg_string(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        s = t.flow_svg()
        assert isinstance(s, str)
        assert s.startswith("<svg")
        assert s.endswith("</svg>")

    def test_contains_input_labels(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('alpha'), eml_vec('beta'))", pure_eml=True
        )
        svg = t.flow_svg()
        assert "alpha" in svg
        assert "beta" in svg

    def test_contains_output_label(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        svg = t.flow_svg(output_label="Result")
        assert "Result" in svg

    def test_no_internal_eml_labels(self):
        # The flow diagram should NOT show 'eml' text — every internal
        # node is implicitly the binary primitive.
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        svg = t.flow_svg()
        # 'eml' should not appear as a text label
        assert "<text" in svg
        assert ">eml<" not in svg

    def test_contains_curves(self):
        t = parse_eml_tree(
            "EML: ops.div(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        svg = t.flow_svg()
        assert "<path" in svg

    def test_contains_junctions(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        svg = t.flow_svg()
        assert "<circle" in svg

    def test_dimensions(self):
        t = parse_eml_tree("EML: eml_scalar(1.0)")
        s = t.flow_svg(width=500, height=400)
        assert 'width="500"' in s
        assert 'height="400"' in s

    def test_background(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        svg = t.flow_svg(background="#222")
        assert 'fill="#222"' in svg

    def test_unicode_leaf_labels(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_pi(), eml_scalar(2.0))", pure_eml=True
        )
        svg = t.flow_svg()
        # Pi should appear as the unicode π
        assert "π" in svg or "&#960;" in svg

    def test_deeply_nested_does_not_crash(self):
        # mul(a, mul(b, mul(c, d)))
        desc = "EML: ops.mul(eml_vec('a'), ops.mul(eml_vec('b'), ops.mul(eml_vec('c'), eml_vec('d'))))"
        t = parse_eml_tree(desc, pure_eml=True)
        svg = t.flow_svg()
        assert "<svg" in svg

    def test_color_blending(self):
        # internal node colour should be average of children's colours
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        root = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[a, b])
        binarised, leaves = _layout(root, width=200, height=200, margin_top=20,
                                     margin_bottom=20, margin_lr=20,
                                     palette=[(200, 0, 0), (0, 0, 200)])
        # root colour = average of (200,0,0) and (0,0,200) = (100,0,100)
        assert binarised._fcolor == pytest.approx((100, 0, 100))


# ---------------------------------------------------------------------------
# HTML wrapper
# ---------------------------------------------------------------------------

class TestFlowHTML:
    def test_wraps_svg_in_div(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        html = t.flow_html()
        assert html.startswith("<div")
        assert "<svg" in html
        assert html.rstrip().endswith("</div>")

    def test_container_class(self):
        t = parse_eml_tree("EML: eml_vec('x')")
        html = t.flow_html(container_class="my-flow")
        assert 'class="my-flow"' in html

    def test_container_id(self):
        t = parse_eml_tree("EML: eml_vec('x')")
        html = t.flow_html(container_id="formula-42")
        assert 'id="formula-42"' in html

    def test_inline_style(self):
        t = parse_eml_tree("EML: eml_vec('x')")
        html = t.flow_html(inline_style="border: 1px solid red")
        assert 'style="border: 1px solid red"' in html


# ---------------------------------------------------------------------------
# PNG renderer
# ---------------------------------------------------------------------------

class TestFlowPNG:
    def test_returns_bytes(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        png = t.flow_png(width=200, height=150)
        assert isinstance(png, bytes)
        assert len(png) > 100

    def test_png_magic_header(self):
        t = parse_eml_tree("EML: eml_vec('x')")
        png = t.flow_png(width=100, height=100)
        # PNG signature: 89 50 4E 47 0D 0A 1A 0A
        assert png[:8] == b"\x89PNG\r\n\x1a\n"

    def test_scale_increases_size(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        small = t.flow_png(width=200, height=150, scale=1.0)
        big   = t.flow_png(width=200, height=150, scale=3.0)
        # The 3x version should produce more pixels and a larger file
        assert len(big) > len(small)

    def test_deep_tree_renders(self):
        desc = (
            "EML: ops.mul(eml_vec('a'), ops.mul(eml_vec('b'), "
            "ops.mul(eml_vec('c'), eml_vec('d'))))"
        )
        t = parse_eml_tree(desc, pure_eml=True)
        png = t.flow_png(width=300, height=300)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
