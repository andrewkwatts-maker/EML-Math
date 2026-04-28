"""
Tests for eml_math.tree — EML operator-tree parser and renderers.

Tests cover both modes:
  expand_eml=True  (default) — compound ops expanded to exp/ln/add/sub/scale trees
  expand_eml=False           — compact ops-level tree (mul, pow, … as single nodes)
"""
import json
import math
import pytest

from eml_math.tree import (
    EMLTreeNode,
    NodeKind,
    EML_EXPANSIONS,
    parse_eml_tree,
    to_compact,
    from_compact,
    KIND_CHAR,
    CHAR_KIND,
)


# ---------------------------------------------------------------------------
# Expanded mode (default) — all compound ops unfold to exp/ln/…
# ---------------------------------------------------------------------------

class TestExpandedMode:
    def test_mul_becomes_exp_add_ln(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b')) — a*b")
        # mul(a,b) = exp(add(ln(a), ln(b)))
        assert t.label == "exp"
        assert t.kind == NodeKind.PRIMITIVE
        add = t.children[0]
        assert add.label == "add"
        assert add.kind == NodeKind.STRUCTURAL
        # add's children are ln(a) and ln(b), not the leaves directly
        ln_a, ln_b = add.children
        assert ln_a.label == "ln" and ln_a.children[0].label == "a"
        assert ln_b.label == "ln" and ln_b.children[0].label == "b"

    def test_mul_wraps_children_in_ln(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_vec('x')) — 3x")
        add = t.children[0]
        # children of add should be ln nodes wrapping leaves
        assert add.children[0].label == "ln"
        assert add.children[0].children[0].label == "3"
        assert add.children[1].label == "ln"
        assert add.children[1].children[0].label == "x"

    def test_div_becomes_exp_sub_ln(self):
        t = parse_eml_tree("EML: ops.div(eml_vec('a'), eml_vec('b')) — a/b")
        assert t.label == "exp"
        sub = t.children[0]
        assert sub.label == "sub"
        assert sub.kind == NodeKind.STRUCTURAL

    def test_sqrt_becomes_exp_scale_ln(self):
        t = parse_eml_tree("EML: ops.sqrt(eml_vec('x')) — sqrt(x)")
        assert t.label == "exp"
        scale = t.children[0]
        assert scale.label == "×0.5"
        assert scale.kind == NodeKind.STRUCTURAL
        ln = scale.children[0]
        assert ln.label == "ln"

    def test_pow_scalar_exponent(self):
        t = parse_eml_tree("EML: ops.pow(eml_vec('x'), eml_scalar(3.0)) — x^3")
        assert t.label == "exp"
        scale = t.children[0]
        assert scale.label == "×3"
        ln = scale.children[0]
        assert ln.label == "ln"
        assert ln.children[0].label == "x"

    def test_pow_integer_exponent(self):
        t = parse_eml_tree("EML: ops.pow(eml_vec('lam'), eml_scalar(2.0)) — lam^2")
        scale = t.children[0]
        assert scale.label == "×2"

    def test_inv_becomes_exp_neg_ln(self):
        t = parse_eml_tree("EML: ops.inv(eml_vec('x')) — 1/x")
        assert t.label == "exp"
        neg = t.children[0]
        assert neg.label == "neg"
        assert neg.children[0].label == "ln"

    def test_exp_primitive_preserved(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(1.0)) — e")
        assert t.label == "exp"
        assert t.kind == NodeKind.PRIMITIVE
        assert t.children[0].label == "1"

    def test_ln_primitive_preserved(self):
        t = parse_eml_tree("EML: ops.ln(eml_vec('x')) — ln(x)")
        assert t.label == "ln"
        assert t.kind == NodeKind.PRIMITIVE
        assert t.children[0].label == "x"

    def test_add_structural(self):
        t = parse_eml_tree("EML: ops.add(eml_scalar(1.0), eml_scalar(2.0)) — 1+2")
        assert t.label == "add"
        assert t.kind == NodeKind.STRUCTURAL

    def test_sub_structural(self):
        t = parse_eml_tree("EML: ops.sub(eml_vec('a'), eml_vec('b')) — a-b")
        assert t.label == "sub"
        assert t.kind == NodeKind.STRUCTURAL

    def test_neg_structural(self):
        t = parse_eml_tree("EML: ops.neg(eml_vec('x')) — -x")
        assert t.label == "neg"
        assert t.kind == NodeKind.STRUCTURAL

    def test_sin_unexpandable_stays_compound(self):
        t = parse_eml_tree("EML: ops.sin(eml_pi()) — sin(pi)")
        assert t.label == "sin"
        assert t.kind == NodeKind.COMPOUND
        assert t.eml_form != ""  # annotation preserved

    def test_leaves_preserved(self):
        t = parse_eml_tree("EML: eml_scalar(42.0) — 42")
        assert t.label == "42"
        assert t.kind == NodeKind.SCALAR

    def test_vec_preserved(self):
        t = parse_eml_tree("EML: eml_vec('alpha_inv') — alpha")
        assert t.label == "alpha_inv"
        assert t.kind == NodeKind.VEC

    def test_pi_preserved(self):
        t = parse_eml_tree("EML: eml_pi() — pi")
        assert t.label == "π"
        assert t.kind == NodeKind.PI


# ---------------------------------------------------------------------------
# Compact mode (expand_eml=False) — ops.* kept as single nodes
# ---------------------------------------------------------------------------

class TestCompactMode:
    def test_mul_stays_single_node(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_scalar(3.0), eml_scalar(4.0)) — 3x4",
            expand_eml=False,
        )
        assert t.label == "mul"
        assert t.kind == NodeKind.COMPOUND
        assert len(t.children) == 2
        assert t.children[0].label == "3"
        assert t.children[1].label == "4"

    def test_pow_stays_single_node(self):
        t = parse_eml_tree(
            "EML: ops.pow(eml_vec('x'), eml_scalar(2.0)) — x^2",
            expand_eml=False,
        )
        assert t.label == "pow"
        assert len(t.children) == 2

    def test_mul_has_eml_form_annotation(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_scalar(2.0), eml_vec('x')) — 2x",
            expand_eml=False,
        )
        assert t.eml_form != ""
        assert "ln" in t.eml_form.lower() or "exp" in t.eml_form.lower()

    def test_exp_is_primitive_in_compact(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(1.0)) — e", expand_eml=False)
        assert t.label == "exp"
        assert t.kind == NodeKind.PRIMITIVE

    def test_compound_kinds_in_compact(self):
        for op in ["mul", "div", "sqrt", "sin", "cos", "pow"]:
            t = parse_eml_tree(
                f"EML: ops.{op}(eml_scalar(1.0), eml_scalar(2.0)) — test",
                expand_eml=False,
            )
            assert t.kind == NodeKind.COMPOUND, f"ops.{op} should be COMPOUND in compact mode"

    def test_nested_compact(self):
        desc = "EML: ops.mul(eml_vec('A'), ops.pow(eml_vec('lam'), eml_scalar(2.0))) — A*lam^2"
        t = parse_eml_tree(desc, expand_eml=False)
        assert t.label == "mul"
        pow_node = t.children[1]
        assert pow_node.label == "pow"


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------

class TestASCII:
    def test_leaf_ascii(self):
        t = parse_eml_tree("EML: eml_scalar(7.0) — seven")
        out = t.ascii()
        assert "7" in out
        assert "└──" in out

    def test_expanded_mul_ascii(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b')) — a*b")
        out = t.ascii()
        assert "exp" in out
        assert "add" in out
        assert "ln" in out
        assert "a" in out
        assert "b" in out

    def test_compact_mul_ascii(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_scalar(3.0), eml_scalar(4.0)) — 3x4",
            expand_eml=False,
        )
        out = t.ascii()
        assert "mul" in out
        assert "3" in out
        assert "4" in out
        assert "├──" in out

    def test_ascii_has_tree_chars(self):
        t = parse_eml_tree("EML: ops.add(eml_vec('a'), eml_vec('b')) — a+b")
        out = t.ascii()
        assert "├──" in out or "└──" in out

    def test_str_delegates_to_ascii(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        assert str(t) == t.ascii()

    def test_eml_form_badge_in_compact(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_scalar(2.0), eml_scalar(3.0)) — 2*3",
            expand_eml=False,
        )
        out = t.ascii()
        assert "[" in out   # badge brackets


# ---------------------------------------------------------------------------
# to_dict / JSON
# ---------------------------------------------------------------------------

class TestToDict:
    def test_leaf_dict(self):
        t = parse_eml_tree("EML: eml_scalar(42.0) — 42")
        d = t.to_dict()
        assert d["label"] == "42"
        assert d["kind"] == NodeKind.SCALAR
        assert "children" not in d

    def test_expanded_mul_dict(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b')) — a*b")
        d = t.to_dict()
        assert d["label"] == "exp"
        assert d["kind"] == NodeKind.PRIMITIVE
        assert "children" in d

    def test_compact_mul_dict(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b')) — a*b",
            expand_eml=False,
        )
        d = t.to_dict()
        assert d["label"] == "mul"
        assert len(d["children"]) == 2

    def test_dict_is_json_serializable(self):
        t = parse_eml_tree("EML: ops.div(eml_vec('b3'), eml_pi()) — b3/pi")
        s = json.dumps(t.to_dict())
        assert "exp" in s
        assert "b3" in s


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------

class TestSVG:
    def test_svg_is_string(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_vec('x')) — 3x")
        assert isinstance(t.svg(), str)

    def test_svg_starts_with_tag(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        svg = t.svg()
        assert svg.strip().startswith("<svg")
        assert "</svg>" in svg

    def test_svg_contains_primitive_labels(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('alpha'), eml_scalar(2.0)) — 2*alpha")
        svg = t.svg()
        assert "exp" in svg
        assert "ln" in svg
        assert "alpha" in svg

    def test_svg_contains_rect(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(1.0)) — e")
        assert "<rect" in t.svg()

    def test_svg_contains_edges(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(2.0), eml_scalar(3.0)) — 2*3")
        assert "<path" in t.svg()

    def test_svg_max_width(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        assert 'width="400"' in t.svg(max_width=400)

    def test_deep_tree_svg(self):
        desc = (
            "EML: ops.mul(eml_vec('A'), "
            "ops.pow(eml_vec('lam'), eml_scalar(2.0))) — A*lam^2"
        )
        svg = parse_eml_tree(desc).svg()
        assert "</svg>" in svg
        assert svg.count("<rect") >= 5


# ---------------------------------------------------------------------------
# Compact serialization (to_compact / from_compact)
# ---------------------------------------------------------------------------

class TestCompactSerialization:
    def test_kind_maps_are_inverse(self):
        for k, v in KIND_CHAR.items():
            assert CHAR_KIND[v] == k

    def test_leaf_compact(self):
        n = EMLTreeNode(label="x", kind=NodeKind.VEC)
        assert to_compact(n) == ["x", "v"]

    def test_internal_compact(self):
        a = EMLTreeNode(label="a", kind=NodeKind.VEC)
        b = EMLTreeNode(label="b", kind=NodeKind.VEC)
        n = EMLTreeNode(label="mul", kind=NodeKind.COMPOUND, children=[a, b])
        assert to_compact(n) == ["mul", "c", ["a", "v"], ["b", "v"]]

    def test_round_trip_leaf(self):
        n = EMLTreeNode(label="42", kind=NodeKind.SCALAR)
        n2 = from_compact(to_compact(n))
        assert n2.label == "42"
        assert n2.kind == NodeKind.SCALAR

    def test_round_trip_internal(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))", expand_eml=False)
        arr = to_compact(t)
        t2 = from_compact(arr)
        assert t2.label == t.label
        assert t2.kind == t.kind
        assert len(t2.children) == 2

    def test_method_aliases(self):
        n = EMLTreeNode(label="x", kind=NodeKind.VEC)
        assert n.to_compact() == ["x", "v"]
        n2 = EMLTreeNode.from_compact(["x", "v"])
        assert n2.label == "x"

    def test_nested_round_trip(self):
        # 7-deep pure-eml tree
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        arr = to_compact(t)
        t2 = from_compact(arr)
        # tree depth should match
        def depth(n):
            return 0 if not n.children else 1 + max(depth(c) for c in n.children)
        assert depth(t) == depth(t2)

    def test_compact_size_smaller_than_dict(self):
        import json
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        compact_size = len(json.dumps(to_compact(t)))
        dict_size    = len(json.dumps(t.to_dict()))
        assert compact_size < dict_size  # always smaller


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------

class TestToLatex:
    # Compact mode (preserves ops labels) — straightforward
    def test_compact_mul(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))", expand_eml=False
        )
        s = t.to_latex()
        assert s == r"a \cdot b"

    def test_compact_div(self):
        t = parse_eml_tree(
            "EML: ops.div(eml_vec('a'), eml_vec('b'))", expand_eml=False
        )
        s = t.to_latex()
        assert s == r"\frac{a}{b}"

    def test_compact_pow(self):
        t = parse_eml_tree(
            "EML: ops.pow(eml_vec('x'), eml_scalar(2.0))", expand_eml=False
        )
        s = t.to_latex()
        assert s == r"x^{2}"

    def test_compact_sqrt(self):
        t = parse_eml_tree(
            "EML: ops.sqrt(eml_vec('x'))", expand_eml=False
        )
        s = t.to_latex()
        assert s == r"\sqrt{x}"

    def test_compact_sin(self):
        t = parse_eml_tree(
            "EML: ops.sin(eml_pi())", expand_eml=False
        )
        s = t.to_latex()
        assert r"\sin" in s and r"\pi" in s

    def test_greek_substitution(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_scalar(2.0), eml_vec('phi'))", expand_eml=False
        )
        s = t.to_latex()
        assert r"\varphi" in s

    def test_subscript_underscore(self):
        t = parse_eml_tree(
            "EML: eml_vec('V_cb')", expand_eml=False
        )
        s = t.to_latex()
        assert s == "V_{cb}"

    def test_dotted_path_strips_prefix(self):
        t = parse_eml_tree(
            "EML: eml_vec('ckm.V_us')", expand_eml=False
        )
        s = t.to_latex()
        assert s == "V_{us}"

    # Expanded mode — recognise compact patterns
    def test_expanded_mul_collapses(self):
        # exp(add(ln a, ln b)) should render as a · b, not literal exp/ln
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))")
        s = t.to_latex()
        assert s == r"a \cdot b"

    def test_expanded_div_collapses(self):
        t = parse_eml_tree("EML: ops.div(eml_vec('a'), eml_vec('b'))")
        s = t.to_latex()
        assert s == r"\frac{a}{b}"

    def test_expanded_pow_collapses(self):
        t = parse_eml_tree("EML: ops.pow(eml_vec('x'), eml_scalar(3.0))")
        s = t.to_latex()
        assert s == r"x^{3}"

    def test_expanded_sqrt_collapses(self):
        t = parse_eml_tree("EML: ops.sqrt(eml_vec('y'))")
        s = t.to_latex()
        assert s == r"\sqrt{y}"

    def test_expanded_inv_collapses(self):
        t = parse_eml_tree("EML: ops.inv(eml_vec('x'))")
        s = t.to_latex()
        assert s == r"\frac{1}{x}"

    # Pure-eml mode — recognise the sentinel patterns
    def test_pure_exp(self):
        # eml(x, 1) → e^x
        t = parse_eml_tree(
            "EML: ops.exp(eml_scalar(1.0))", pure_eml=True
        )
        s = t.to_latex()
        assert s == r"e^{1}"

    def test_pure_ln(self):
        # 3-nested eml encoding → ln(y)
        t = parse_eml_tree(
            "EML: ops.ln(eml_vec('y'))", pure_eml=True
        )
        s = t.to_latex()
        assert r"\ln" in s and "y" in s

    def test_pure_neg(self):
        # eml(⊥, eml(x, 1)) → -x
        t = parse_eml_tree(
            "EML: ops.neg(eml_vec('x'))", pure_eml=True
        )
        s = t.to_latex()
        assert "-" in s and "x" in s


# ---------------------------------------------------------------------------
# Pure EML mode — every internal node is the binary primitive eml(L, R)
# ---------------------------------------------------------------------------

class TestPureEmlMode:
    @staticmethod
    def _all_internal_eml(node):
        if not node.children:
            return True
        if node.label != "eml":
            return False
        return all(TestPureEmlMode._all_internal_eml(c) for c in node.children)

    def test_exp_becomes_eml_x_one(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(2.0))", pure_eml=True)
        assert t.label == "eml"
        assert len(t.children) == 2
        assert t.children[0].label == "2"
        assert t.children[1].label == "1"
        assert t.children[1].kind == NodeKind.SCALAR

    def test_ln_uses_three_nested_eml(self):
        t = parse_eml_tree("EML: ops.ln(eml_vec('y'))", pure_eml=True)
        # ln(y) = eml(⊥, eml(eml(⊥, y), 1))
        assert t.label == "eml"
        assert t.children[0].kind == NodeKind.BOTTOM
        mid = t.children[1]
        assert mid.label == "eml"
        inner = mid.children[0]
        assert inner.label == "eml"
        assert inner.children[0].kind == NodeKind.BOTTOM
        assert inner.children[1].label == "y"
        assert mid.children[1].label == "1"

    def test_neg_uses_two_nested_eml(self):
        t = parse_eml_tree("EML: ops.neg(eml_vec('x'))", pure_eml=True)
        # neg(x) = eml(⊥, eml(x, 1))
        assert t.label == "eml"
        assert t.children[0].kind == NodeKind.BOTTOM
        inner = t.children[1]
        assert inner.label == "eml"
        assert inner.children[0].label == "x"
        assert inner.children[1].label == "1"

    def test_sub_pure_form(self):
        # sub(u, v): outer eml( pure_ln(u), eml(v, 1) )
        t = parse_eml_tree("EML: ops.sub(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        assert t.label == "eml"
        # right leg is exp(v) = eml(v, 1)
        right = t.children[1]
        assert right.label == "eml"
        assert right.children[0].label == "b"
        assert right.children[1].label == "1"

    def test_mul_all_internal_eml(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        assert self._all_internal_eml(t)

    def test_div_all_internal_eml(self):
        t = parse_eml_tree("EML: ops.div(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        assert self._all_internal_eml(t)

    def test_pow_all_internal_eml(self):
        t = parse_eml_tree(
            "EML: ops.pow(eml_vec('x'), eml_scalar(2.0))", pure_eml=True
        )
        assert self._all_internal_eml(t)

    def test_sqrt_all_internal_eml(self):
        t = parse_eml_tree("EML: ops.sqrt(eml_vec('x'))", pure_eml=True)
        assert self._all_internal_eml(t)

    def test_inv_all_internal_eml(self):
        t = parse_eml_tree("EML: ops.inv(eml_vec('x'))", pure_eml=True)
        assert self._all_internal_eml(t)

    def test_leaves_preserved(self):
        t = parse_eml_tree("EML: eml_scalar(7.0)", pure_eml=True)
        assert t.label == "7"
        assert t.kind == NodeKind.SCALAR

    def test_pure_implies_expand(self):
        # pure_eml=True overrides expand_eml=False
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('a'), eml_vec('b'))",
            pure_eml=True,
            expand_eml=False,
        )
        assert t.label == "eml"

    def test_to_dict_serializable(self):
        import json
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))", pure_eml=True)
        s = json.dumps(t.to_dict())
        assert '"label": "eml"' in s
        assert '"bottom"' in s


# ---------------------------------------------------------------------------
# Parse error handling
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_bad_syntax_returns_error_node(self):
        t = parse_eml_tree("EML: (((broken syntax")
        assert t.kind == NodeKind.UNKNOWN
        assert "parse error" in t.label.lower()

    def test_missing_prefix_raises(self):
        from eml_math.evaluator import ParseError
        with pytest.raises(ParseError):
            parse_eml_tree("ops.mul(eml_scalar(1), eml_scalar(2))")
