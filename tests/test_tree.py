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
