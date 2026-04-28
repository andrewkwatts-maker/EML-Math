"""
Tests for eml_math.flow — flow-diagram renderer (SVG, HTML, PNG).
"""
import pytest

from eml_math.tree import parse_eml_tree, EMLTreeNode, NodeKind
from eml_math.flow import (
    flow_svg, flow_html, flow_png, DEFAULT_PALETTE, DIRECTIONS,
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
            t, width=400, height=300, margin_lead=40, margin_trail=40,
            margin_cross=40, palette=DEFAULT_PALETTE,
            direction="down", expand_symbols=False,
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
        binarised, leaves = _layout(root, width=200, height=200, margin_lead=20,
                                     margin_trail=20, margin_cross=20,
                                     palette=[(200, 0, 0), (0, 0, 200)],
                                     direction="down", expand_symbols=False)
        # root colour = average of (200,0,0) and (0,0,200) = (100,0,100)
        assert binarised._fcolor == pytest.approx((100, 0, 100))


# ---------------------------------------------------------------------------
# Direction (orientation) — down (default), up, right, left
# ---------------------------------------------------------------------------

class TestDirection:
    DESC = "EML: ops.mul(eml_vec('a'), eml_vec('b'))"

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_renders(self, direction):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        svg = t.flow_svg(direction=direction, output_label="C")
        assert svg.startswith("<svg")
        assert "C" in svg
        assert "a" in svg and "b" in svg

    def test_invalid_direction_raises(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        with pytest.raises(ValueError, match="direction"):
            t.flow_svg(direction="diagonal")

    def test_down_leaves_at_top(self):
        # In 'down' mode the leaves should be visually above the root
        t = parse_eml_tree(self.DESC, expand_eml=False)
        from eml_math.flow import _layout
        bnode, leaves = _layout(
            t, width=400, height=300,
            margin_lead=40, margin_trail=40, margin_cross=40,
            palette=DEFAULT_PALETTE, direction="down", expand_symbols=False,
        )
        # Leaf y should be smaller (higher on screen) than root y
        for leaf in leaves:
            assert leaf._fy < bnode._fy

    def test_right_leaves_at_left(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        from eml_math.flow import _layout
        bnode, leaves = _layout(
            t, width=400, height=300,
            margin_lead=40, margin_trail=40, margin_cross=40,
            palette=DEFAULT_PALETTE, direction="right", expand_symbols=False,
        )
        # Leaf x should be smaller (further left) than root x
        for leaf in leaves:
            assert leaf._fx < bnode._fx

    def test_up_leaves_at_bottom(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        from eml_math.flow import _layout
        bnode, leaves = _layout(
            t, width=400, height=300,
            margin_lead=40, margin_trail=40, margin_cross=40,
            palette=DEFAULT_PALETTE, direction="up", expand_symbols=False,
        )
        for leaf in leaves:
            assert leaf._fy > bnode._fy

    def test_left_leaves_at_right(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        from eml_math.flow import _layout
        bnode, leaves = _layout(
            t, width=400, height=300,
            margin_lead=40, margin_trail=40, margin_cross=40,
            palette=DEFAULT_PALETTE, direction="left", expand_symbols=False,
        )
        for leaf in leaves:
            assert leaf._fx > bnode._fx

    def test_png_renders_in_each_direction(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        for d in DIRECTIONS:
            png = t.flow_png(direction=d, width=400, height=300)
            assert png[:8] == b"\x89PNG\r\n\x1a\n", f"{d} PNG malformed"

    def test_pdf_renders_in_each_direction(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        for d in DIRECTIONS:
            pdf = t.flow_pdf(direction=d, width=400, height=300)
            assert pdf[:5] == b"%PDF-", f"{d} PDF malformed"


# ---------------------------------------------------------------------------
# Multi-output (e.g. quadratic ±)
# ---------------------------------------------------------------------------

class TestMultiOutput:
    DESC = ("EML: ops.div(ops.add(ops.neg(eml_vec('b')), "
            "ops.mul(eml_vec('sign'), ops.sqrt(ops.sub(ops.pow(eml_vec('b'), eml_scalar(2.0)), "
            "ops.mul(eml_scalar(4.0), ops.mul(eml_vec('a'), eml_vec('c'))))))), "
            "ops.mul(eml_scalar(2.0), eml_vec('a')))")

    def test_list_output_label_renders_both(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        svg = t.flow_svg(output_label=["x_+", "x_-"])
        assert "x_+" in svg
        assert "x_-" in svg
        # ± indicator should appear
        assert "±" in svg

    def test_tuple_output_label_works(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        svg = t.flow_svg(output_label=("root1", "root2"))
        assert "root1" in svg
        assert "root2" in svg

    def test_single_string_no_indicator(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        svg = t.flow_svg(output_label="x")
        assert "±" not in svg

    def test_png_multi_output(self):
        t = parse_eml_tree(self.DESC, expand_eml=False)
        png = t.flow_png(output_label=["x_+", "x_-"], width=600, height=400)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---------------------------------------------------------------------------
# Equal labels share a colour (same-input deduplication)
# ---------------------------------------------------------------------------

class TestEqualLabelsSameColor:
    def test_pythagoras_two_2s_same_color(self):
        # Pythagoras has two `2` literals (one in each pow) — they should
        # be the same colour.
        t = parse_eml_tree(
            "EML: ops.sqrt(ops.add(ops.pow(eml_vec('a'), eml_scalar(2.0)), "
            "ops.pow(eml_vec('b'), eml_scalar(2.0))))",
            expand_eml=False,
        )
        from eml_math.flow import _layout
        _, leaves = _layout(t, width=400, height=300, margin_lead=40,
                            margin_trail=40, margin_cross=40,
                            palette=DEFAULT_PALETTE, direction="down",
                            expand_symbols=False)
        twos = [l for l in leaves if l.label == "2"]
        assert len(twos) == 2
        assert twos[0]._fcolor == twos[1]._fcolor

    def test_distinct_labels_distinct_colors(self):
        t = parse_eml_tree(
            "EML: ops.add(eml_vec('a'), ops.add(eml_vec('b'), eml_vec('c')))",
            expand_eml=False,
        )
        from eml_math.flow import _layout
        _, leaves = _layout(t, width=400, height=300, margin_lead=40,
                            margin_trail=40, margin_cross=40,
                            palette=DEFAULT_PALETTE, direction="down",
                            expand_symbols=False)
        cols = [l._fcolor for l in leaves]
        # all three different
        assert len(set(cols)) == 3

    def test_identity_children_bypass_color_blend(self):
        """L=0 (exp leg vanishes) and R=1 (ln leg vanishes) should not
        contribute to the parent junction's blended colour."""
        from eml_math.tree import EMLTreeNode, NodeKind
        from eml_math.flow import _layout, FIXED_COLORS, DEFAULT_PALETTE

        # eml(x, 1) — R=1 is identity; node colour should equal x's colour.
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        one = EMLTreeNode(label="1", kind=NodeKind.SCALAR)
        root = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[x, one])
        rt, _ = _layout(root, width=300, height=200, margin_lead=20,
                        margin_trail=20, margin_cross=20,
                        palette=[(255, 0, 0), (0, 0, 255)],
                        direction="down", expand_symbols=False,
                        bypass_identity_blend=True)
        # x got palette[0] = (255, 0, 0); root colour should equal x's
        assert rt._fcolor == pytest.approx((255, 0, 0))

        # eml(0, y) — L=0 is identity; node colour should equal y's colour.
        zero = EMLTreeNode(label="0", kind=NodeKind.BOTTOM)
        y = EMLTreeNode(label="y", kind=NodeKind.VEC)
        root2 = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[zero, y])
        rt2, _ = _layout(root2, width=300, height=200, margin_lead=20,
                         margin_trail=20, margin_cross=20,
                         palette=[(0, 200, 0)],   # only y picks one up
                         direction="down", expand_symbols=False,
                         bypass_identity_blend=True)
        # y got palette[0] = (0, 200, 0); root colour should equal y's
        assert rt2._fcolor == pytest.approx((0, 200, 0))

    def test_blend_without_bypass_includes_identity(self):
        """With bypass_identity_blend=False the L=0 / R=1 colours DO blend
        into the parent (legacy behaviour)."""
        from eml_math.tree import EMLTreeNode, NodeKind
        from eml_math.flow import _layout

        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        one = EMLTreeNode(label="1", kind=NodeKind.SCALAR)
        root = EMLTreeNode(label="eml", kind=NodeKind.PRIMITIVE, children=[x, one])
        # Force palette where x gets red, then 1 picks up the FIXED grey.
        from eml_math.flow import FIXED_COLORS
        rt, _ = _layout(root, width=300, height=200, margin_lead=20,
                        margin_trail=20, margin_cross=20,
                        palette=[(255, 0, 0)],
                        direction="down", expand_symbols=False,
                        bypass_identity_blend=False)
        # Average of x's red and 1's grey should NOT equal red.
        assert rt._fcolor != pytest.approx((255, 0, 0))

    def test_all_leaves_get_real_palette_color(self):
        # No sentinels — every distinct label gets its own palette colour;
        # equal labels (including ⊥, 1) share that colour.
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True
        )
        from eml_math.flow import _layout
        _, leaves = _layout(t, width=400, height=300, margin_lead=40,
                            margin_trail=40, margin_cross=40,
                            palette=DEFAULT_PALETTE, direction="down",
                            expand_symbols=False)
        for label in {"⊥", "1", "a", "b"}:
            same = [l for l in leaves if l.label == label]
            if not same:
                continue
            shared = same[0]._fcolor
            for s in same[1:]:
                assert s._fcolor == shared, f"{label} leaves disagree on colour"


# ---------------------------------------------------------------------------
# merge_inputs — deduplicate identical inputs
# ---------------------------------------------------------------------------

class TestMergeInputs:
    PYTHAG = ("EML: ops.sqrt(ops.add(ops.pow(eml_vec('a'), eml_scalar(2.0)), "
              "ops.pow(eml_vec('b'), eml_scalar(2.0))))")

    def test_merged_renders(self):
        t = parse_eml_tree(self.PYTHAG, expand_eml=False)
        svg = t.flow_svg(merge_inputs=True, output_label="c")
        assert svg.startswith("<svg")
        assert "c" in svg

    def test_merged_has_one_label_per_unique_input(self):
        # Without merge: 4 input labels (a, 2, b, 2). With merge: 3 (a, 2, b).
        t = parse_eml_tree(self.PYTHAG, expand_eml=False)
        svg_off = t.flow_svg(merge_inputs=False, output_label="c")
        svg_on  = t.flow_svg(merge_inputs=True,  output_label="c")
        # Count <text> tags that contain just '2' between > and < — works both ways.
        # Simpler: count occurrences of '>2<' which appear as label text only.
        c_off = svg_off.count(">2<")
        c_on  = svg_on.count(">2<")
        assert c_off == 2, f"un-merged should show '2' twice, got {c_off}"
        assert c_on  == 1, f"merged should show '2' once, got {c_on}"

    def test_merged_has_extra_redirector_curves(self):
        t = parse_eml_tree(self.PYTHAG, expand_eml=False)
        svg_off = t.flow_svg(merge_inputs=False, output_label="c")
        svg_on  = t.flow_svg(merge_inputs=True,  output_label="c")
        # Merged version draws redirector curves from each unique input to
        # every usage point in the tree → strictly more <path> elements.
        assert svg_on.count("<path") > svg_off.count("<path")


# ---------------------------------------------------------------------------
# Symbol expansion (expand_symbols=True)
# ---------------------------------------------------------------------------

class TestExpandSymbols:
    def test_e_expands(self):
        # A formula that uses 'e' as a leaf-name vec.
        t = parse_eml_tree("EML: ops.mul(eml_vec('e'), eml_scalar(2.0))",
                           expand_eml=False)
        # Without expansion: the leaf is just labeled 'e'
        svg_off = t.flow_svg(expand_symbols=False)
        # With expansion: 'e' is replaced by eml(1, 1) → tree gets bigger
        svg_on  = t.flow_svg(expand_symbols=True)
        # The expanded version should have more SVG paths (more nodes)
        assert svg_on.count("<path") > svg_off.count("<path")

    def test_unknown_symbol_unchanged(self):
        t = parse_eml_tree("EML: eml_vec('totally_unknown_var')",
                           expand_eml=False)
        svg = t.flow_svg(expand_symbols=True)
        assert "totally_unknown_var" in svg

    def test_pi_kept_as_leaf(self):
        # π is a primitive symbol with no closed-form construction
        t = parse_eml_tree("EML: ops.mul(eml_pi(), eml_vec('r'))", expand_eml=False)
        svg = t.flow_svg(expand_symbols=True)
        # π should still appear as a leaf
        assert "π" in svg


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
