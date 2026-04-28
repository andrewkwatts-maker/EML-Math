"""
Tests for eml_math.symbols — named EML constructions for common math symbols.
"""
import math
import pytest

from eml_math.symbols import (
    Symbol, SYMBOLS, lookup, construct, register,
    pure_exp, pure_ln, pure_neg, pure_inv,
)
from eml_math.tree import EMLTreeNode, NodeKind


class TestRegistry:
    def test_builtins_present(self):
        for name in ("e", "pi", "tau", "phi", "sqrt2", "sqrt3", "sqrt5", "ln2", "gamma_em"):
            assert name in SYMBOLS

    def test_lookup_returns_symbol(self):
        e = lookup("e")
        assert isinstance(e, Symbol)
        assert e.name == "e"
        assert e.value == pytest.approx(math.e, rel=1e-12)

    def test_lookup_unknown_returns_none(self):
        assert lookup("not_a_symbol") is None

    def test_construct_returns_tree(self):
        t = construct("e")
        assert isinstance(t, EMLTreeNode)
        assert t.label == "eml"
        assert t.kind == NodeKind.PRIMITIVE

    def test_construct_raises_on_primitive(self):
        with pytest.raises(ValueError, match="no closed-form"):
            construct("pi")

    def test_construct_raises_on_unknown(self):
        with pytest.raises(KeyError):
            construct("not_a_symbol")

    def test_register_custom(self):
        my_sym = Symbol(
            name="my_constant", latex=r"\xi", description="test",
            value=42.0, tree=None,
        )
        register(my_sym)
        assert lookup("my_constant") is my_sym
        # cleanup
        del SYMBOLS["my_constant"]


class TestEConstruction:
    def test_e_value(self):
        assert lookup("e").value == pytest.approx(math.e, rel=1e-12)

    def test_e_tree_is_eml_one_one(self):
        # e = exp(1) = eml(1, 1)
        t = construct("e")
        assert t.label == "eml"
        assert len(t.children) == 2
        assert t.children[0].label == "1"
        assert t.children[1].label == "1"


class TestPiConstruction:
    def test_pi_value(self):
        assert lookup("pi").value == pytest.approx(math.pi, rel=1e-14)

    def test_pi_has_no_construction(self):
        # π has no closed-form in elementary EML primitives
        assert lookup("pi").tree is None


class TestPhiConstruction:
    def test_phi_value(self):
        assert lookup("phi").value == pytest.approx((1 + math.sqrt(5)) / 2, rel=1e-12)

    def test_phi_tree_exists(self):
        t = construct("phi")
        assert t is not None
        # Tree mentions sqrt5 sub-construction
        assert any("eml" in str(c.label) or "add" in str(c.label) or "div" in str(c.label)
                   for c in [t] + t.children)


class TestSqrtConstructions:
    def test_sqrt2(self):
        assert lookup("sqrt2").value == pytest.approx(math.sqrt(2), rel=1e-12)
        t = construct("sqrt2")
        # Tree should be exp(scale * ln(2))-style — top-level eml
        assert t.label == "eml"

    def test_sqrt3(self):
        assert lookup("sqrt3").value == pytest.approx(math.sqrt(3), rel=1e-12)

    def test_sqrt5(self):
        assert lookup("sqrt5").value == pytest.approx(math.sqrt(5), rel=1e-12)


class TestPureBuilders:
    def test_pure_exp_shape(self):
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        t = pure_exp(x)
        assert t.label == "eml"
        assert t.children[0] is x
        assert t.children[1].label == "1"

    def test_pure_ln_three_nested(self):
        y = EMLTreeNode(label="y", kind=NodeKind.VEC)
        t = pure_ln(y)
        assert t.label == "eml"
        # outer eml: ⊥, eml(eml(⊥, y), 1)
        assert t.children[0].kind == NodeKind.BOTTOM
        mid = t.children[1]
        assert mid.label == "eml"
        assert mid.children[0].label == "eml"

    def test_pure_neg_shape(self):
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        t = pure_neg(x)
        # neg(x) = eml(⊥, eml(x, 1))
        assert t.label == "eml"
        assert t.children[0].kind == NodeKind.BOTTOM

    def test_pure_inv_shape(self):
        x = EMLTreeNode(label="x", kind=NodeKind.VEC)
        t = pure_inv(x)
        assert t.label == "eml"
