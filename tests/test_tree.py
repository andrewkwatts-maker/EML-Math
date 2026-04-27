"""
Tests for eml_math.tree — EML operator-tree parser and renderers.
"""
import math
import pytest

from eml_math.tree import (
    EMLTreeNode,
    NodeKind,
    EML_EXPANSIONS,
    parse_eml_tree,
)


# ---------------------------------------------------------------------------
# parse_eml_tree — basic structure
# ---------------------------------------------------------------------------

class TestParseBasic:
    def test_scalar_leaf(self):
        t = parse_eml_tree("EML: eml_scalar(42.0) — forty-two")
        assert t.label == "42"
        assert t.kind == NodeKind.SCALAR
        assert t.children == []

    def test_pi_leaf(self):
        t = parse_eml_tree("EML: eml_pi() — pi")
        assert t.label == "π"
        assert t.kind == NodeKind.PI

    def test_vec_leaf(self):
        t = parse_eml_tree("EML: eml_vec('alpha_inv') — α⁻¹")
        assert t.label == "alpha_inv"
        assert t.kind == NodeKind.VEC

    def test_mul_structure(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_scalar(4.0)) — 3×4")
        assert t.label == "mul"
        assert t.kind == NodeKind.COMPOUND
        assert len(t.children) == 2
        assert t.children[0].label == "3"
        assert t.children[1].label == "4"

    def test_div_structure(self):
        t = parse_eml_tree("EML: ops.div(eml_vec('b3'), eml_pi()) — b₃/π")
        assert t.label == "div"
        assert t.kind == NodeKind.COMPOUND
        assert t.children[0].label == "b3"
        assert t.children[1].label == "π"

    def test_nested(self):
        desc = "EML: ops.mul(eml_vec('A'), ops.pow(eml_vec('lam'), eml_scalar(2.0))) — A·λ²"
        t = parse_eml_tree(desc)
        assert t.label == "mul"
        pow_node = t.children[1]
        assert pow_node.label == "pow"
        assert pow_node.children[0].label == "lam"
        assert pow_node.children[1].label == "2"

    def test_depth_three(self):
        desc = "EML: ops.exp(ops.mul(eml_scalar(2.0), ops.ln(eml_vec('x')))) — exp(2·ln(x))"
        t = parse_eml_tree(desc)
        assert t.label == "exp"
        assert t.kind == NodeKind.PRIMITIVE
        mul = t.children[0]
        assert mul.label == "mul"
        ln = mul.children[1]
        assert ln.label == "ln"
        assert ln.kind == NodeKind.PRIMITIVE


# ---------------------------------------------------------------------------
# EML expansion annotations
# ---------------------------------------------------------------------------

class TestEMLExpansions:
    def test_mul_has_expansion(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(2.0), eml_scalar(3.0)) — 2×3")
        assert t.eml_form != ""
        assert "ln" in t.eml_form.lower() or "exp" in t.eml_form.lower()

    def test_exp_has_expansion(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(1.0)) — e")
        assert t.eml_form == EML_EXPANSIONS["exp"]

    def test_pow_has_expansion(self):
        t = parse_eml_tree("EML: ops.pow(eml_vec('x'), eml_scalar(3.0)) — x³")
        assert t.eml_form != ""

    def test_leaf_no_expansion(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        assert t.eml_form == ""

    def test_vec_no_expansion(self):
        t = parse_eml_tree("EML: eml_vec('b3') — b₃")
        assert t.eml_form == ""


# ---------------------------------------------------------------------------
# NodeKind assignment
# ---------------------------------------------------------------------------

class TestNodeKind:
    def test_compound_kinds(self):
        for op in ["mul", "div", "add", "sub", "sqrt", "sin", "cos", "pow"]:
            t = parse_eml_tree(f"EML: ops.{op}(eml_scalar(1.0), eml_scalar(2.0)) — test")
            assert t.kind == NodeKind.COMPOUND, f"ops.{op} should be COMPOUND"

    def test_primitive_kinds(self):
        for op in ["exp", "ln"]:
            t = parse_eml_tree(f"EML: ops.{op}(eml_scalar(1.0)) — test")
            assert t.kind == NodeKind.PRIMITIVE, f"ops.{op} should be PRIMITIVE"

    def test_scalar_kind(self):
        t = parse_eml_tree("EML: eml_scalar(5.0) — five")
        assert t.kind == NodeKind.SCALAR

    def test_vec_kind(self):
        t = parse_eml_tree("EML: eml_vec('x') — x")
        assert t.kind == NodeKind.VEC

    def test_pi_kind(self):
        t = parse_eml_tree("EML: eml_pi() — pi")
        assert t.kind == NodeKind.PI


# ---------------------------------------------------------------------------
# ASCII renderer
# ---------------------------------------------------------------------------

class TestASCII:
    def test_leaf_ascii(self):
        t = parse_eml_tree("EML: eml_scalar(7.0) — seven")
        out = t.ascii()
        assert "7" in out
        assert "└──" in out

    def test_mul_ascii_has_children(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_scalar(4.0)) — 3×4")
        out = t.ascii()
        assert "mul" in out
        assert "3" in out
        assert "4" in out
        assert "├──" in out
        assert "└──" in out

    def test_ascii_indentation(self):
        desc = "EML: ops.mul(eml_vec('A'), ops.pow(eml_vec('lam'), eml_scalar(2.0))) — test"
        t = parse_eml_tree(desc)
        lines = t.ascii().splitlines()
        # pow node should start later (more leading whitespace) than mul (root)
        pow_line = next(l for l in lines if "pow" in l)
        mul_line = lines[0]
        pow_indent = len(pow_line) - len(pow_line.lstrip())
        mul_indent = len(mul_line) - len(mul_line.lstrip())
        assert pow_indent > mul_indent

    def test_str_delegates_to_ascii(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        assert str(t) == t.ascii()

    def test_eml_form_in_ascii(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(2.0), eml_scalar(3.0)) — 2×3")
        out = t.ascii()
        assert "[" in out   # badge brackets present


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

    def test_compound_dict_has_children(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(2.0), eml_vec('x')) — 2x")
        d = t.to_dict()
        assert "children" in d
        assert len(d["children"]) == 2

    def test_dict_is_json_serializable(self):
        import json
        t = parse_eml_tree(
            "EML: ops.div(eml_vec('b3'), eml_pi()) — b₃/π"
        )
        s = json.dumps(t.to_dict())
        assert "div" in s
        assert "b3" in s


# ---------------------------------------------------------------------------
# SVG renderer
# ---------------------------------------------------------------------------

class TestSVG:
    def test_svg_is_string(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(3.0), eml_vec('x')) — 3x")
        svg = t.svg()
        assert isinstance(svg, str)

    def test_svg_starts_with_tag(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        svg = t.svg()
        assert svg.strip().startswith("<svg")
        assert "</svg>" in svg

    def test_svg_contains_node_labels(self):
        t = parse_eml_tree(
            "EML: ops.mul(eml_vec('alpha'), eml_scalar(2.0)) — 2·alpha"
        )
        svg = t.svg()
        assert "mul" in svg
        assert "alpha" in svg
        assert "2" in svg

    def test_svg_contains_rect(self):
        t = parse_eml_tree("EML: ops.exp(eml_scalar(1.0)) — e")
        svg = t.svg()
        assert "<rect" in svg

    def test_svg_contains_edges(self):
        t = parse_eml_tree("EML: ops.mul(eml_scalar(2.0), eml_scalar(3.0)) — 2×3")
        svg = t.svg()
        assert "<path" in svg

    def test_svg_max_width_respected(self):
        t = parse_eml_tree("EML: eml_scalar(1.0) — one")
        svg = t.svg(max_width=400)
        assert 'width="400"' in svg

    def test_deep_tree_svg(self):
        desc = (
            "EML: ops.mul(eml_vec('A'), "
            "ops.pow(eml_vec('lam'), "
            "ops.add(eml_scalar(2.0), eml_scalar(1.0)))) — A·λ³"
        )
        t = parse_eml_tree(desc)
        svg = t.svg()
        assert "</svg>" in svg
        assert svg.count("<rect") >= 5  # root + 2 children + 2 grandchildren


# ---------------------------------------------------------------------------
# Parse error handling
# ---------------------------------------------------------------------------

class TestParseErrors:
    def test_bad_syntax_returns_error_node(self):
        t = parse_eml_tree("EML: (((broken syntax")
        assert t.kind == NodeKind.UNKNOWN
        assert "parse error" in t.label.lower()

    def test_missing_prefix_raises(self):
        # parse_eml_tree delegates prefix-stripping to EMLEvaluator._parse
        # which raises ParseError — we let it propagate
        from eml_math.evaluator import ParseError
        with pytest.raises(ParseError):
            parse_eml_tree("ops.mul(eml_scalar(1), eml_scalar(2))")
