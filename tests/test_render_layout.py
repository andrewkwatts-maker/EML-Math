"""
Layout-pass tests for the abstracted render pipeline.

Covers:
  - The raw formula JSON is structure-only (no positions).
  - Reingold-Tilford produces no overlapping siblings.
  - All four directions project consistently.
  - Canvas auto-grows for deep trees.
  - Layout JSON round-trips through json.dumps cleanly.
"""
from __future__ import annotations

import json
import math
import pytest

from eml_math import parse_eml_tree
from eml_math.render import compute_layout, DIRECTIONS, EDGE_STYLES
from eml_math.render.layout import LAYOUT_SCHEMA


def _tree(desc: str = "EML: ops.mul(eml_vec('a'), eml_vec('b'))",
          *, expand: bool = False):
    return parse_eml_tree(desc, expand_eml=expand)


# ── Raw JSON has zero positional data ────────────────────────────────────────

class TestRawFormulaJson:

    def test_no_positional_fields(self):
        d = _tree().to_dict()
        s = json.dumps(d)
        for forbidden in ("x", "y", "color", "vertical_bias", "render_hints",
                          "canvas", "width", "height"):
            assert f'"{forbidden}"' not in s, f"forbidden field {forbidden!r} found"

    def test_schema_present(self):
        d = _tree().to_dict()
        assert d["schema"] == "eml-formula/v1"

    def test_schema_omitted_when_disabled(self):
        d = _tree().to_dict(schema=False)
        assert "schema" not in d

    def test_only_structural_keys(self):
        d = _tree().to_dict()
        allowed = {"schema", "label", "kind", "eml_form", "children"}
        assert set(d.keys()) <= allowed
        # Recursively check every node.
        def _walk(node):
            assert set(node.keys()) <= allowed | {"schema"}
            for c in node.get("children", []):
                _walk(c)
        _walk(d)


# ── Layout dict shape ────────────────────────────────────────────────────────

class TestLayoutShape:

    def test_layout_schema(self):
        L = compute_layout(_tree().to_dict())
        assert L["schema"] == LAYOUT_SCHEMA

    def test_canvas_keys(self):
        L = compute_layout(_tree().to_dict())
        assert set(L["canvas"]) == {"width", "height"}
        assert L["canvas"]["width"] > 0 and L["canvas"]["height"] > 0

    def test_node_keys(self):
        L = compute_layout(_tree().to_dict())
        for n in L["nodes"]:
            assert set(n.keys()) >= {"id", "label", "kind", "x", "y",
                                     "color", "is_leaf", "depth"}
            assert isinstance(n["color"], list) and len(n["color"]) == 3
            assert all(0 <= c <= 255 for c in n["color"])

    def test_edge_keys(self):
        L = compute_layout(_tree().to_dict())
        for e in L["edges"]:
            assert set(e.keys()) >= {"from", "to", "style", "color"}
            assert e["style"] in EDGE_STYLES

    def test_layout_is_json_serializable(self):
        L = compute_layout(_tree().to_dict())
        s = json.dumps(L)
        L2 = json.loads(s)
        assert L2["schema"] == LAYOUT_SCHEMA
        assert len(L2["nodes"]) == len(L["nodes"])

    @pytest.mark.parametrize("style", EDGE_STYLES)
    def test_edge_style_propagates(self, style):
        L = compute_layout(_tree().to_dict(), edge_style=style)
        assert L["edge_style"] == style
        for e in L["edges"]:
            assert e["style"] == style


# ── Reingold-Tilford no-overlap guarantee ────────────────────────────────────

class TestNoOverlap:

    def _leaves(self, layout):
        return [n for n in layout["nodes"] if n["is_leaf"]]

    def _by_depth(self, layout):
        out: dict[int, list] = {}
        for n in layout["nodes"]:
            out.setdefault(n["depth"], []).append(n)
        return out

    def test_siblings_non_overlapping_simple(self):
        L = compute_layout(_tree().to_dict())
        leaves = sorted(self._leaves(L), key=lambda n: n["x"])
        for a, b in zip(leaves, leaves[1:]):
            assert b["x"] - a["x"] > 0, "leaves overlap"

    def test_no_node_overlap_deep_tree(self):
        # mul(add(a, b), c) — three leaves at depth 1, two siblings deep.
        desc = "EML: ops.mul(ops.add(eml_vec('a'), eml_vec('b')), eml_vec('c'))"
        L = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict())
        leaves = sorted(self._leaves(L), key=lambda n: n["x"])
        for a, b in zip(leaves, leaves[1:]):
            assert b["x"] - a["x"] > 10.0, f"leaves too close: {a} {b}"

    def test_no_overlap_chain(self):
        # Nested unary chain shouldn't drift; children all align.
        desc = "EML: ops.exp(ops.exp(eml_vec('x')))"
        L = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict())
        # Single leaf → vertical line; no horizontal overlap to check.
        assert len(self._leaves(L)) == 1

    def test_each_node_unique_id(self):
        desc = "EML: ops.mul(ops.add(eml_vec('a'), eml_vec('b')), eml_vec('c'))"
        L = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict())
        ids = [n["id"] for n in L["nodes"]]
        assert len(ids) == len(set(ids))


# ── Direction projection ─────────────────────────────────────────────────────

class TestDirections:

    @pytest.mark.parametrize("direction", DIRECTIONS)
    def test_direction_round_trip(self, direction):
        L = compute_layout(_tree().to_dict(), direction=direction)
        assert L["direction"] == direction
        # All nodes inside canvas.
        w, h = L["canvas"]["width"], L["canvas"]["height"]
        for n in L["nodes"]:
            assert -1 <= n["x"] <= w + 1
            assert -1 <= n["y"] <= h + 1

    def test_down_inputs_at_top(self):
        desc = "EML: ops.mul(eml_vec('a'), eml_vec('b'))"
        L = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict(),
                           direction="down")
        leaves = [n for n in L["nodes"] if n["is_leaf"]]
        root = next(n for n in L["nodes"] if not n["is_leaf"])
        # Leaves should be ABOVE root (smaller y) in down-flow.
        for l in leaves:
            assert l["y"] < root["y"]

    def test_up_inputs_at_bottom(self):
        desc = "EML: ops.mul(eml_vec('a'), eml_vec('b'))"
        L = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict(),
                           direction="up")
        leaves = [n for n in L["nodes"] if n["is_leaf"]]
        root = next(n for n in L["nodes"] if not n["is_leaf"])
        for l in leaves:
            assert l["y"] > root["y"]


# ── Canvas auto-grow ─────────────────────────────────────────────────────────

class TestCanvasGrow:

    def test_canvas_grows_for_deep_tree(self):
        # Force a deep tree.
        desc = "EML: ops.mul(ops.mul(ops.mul(eml_vec('a'), eml_vec('b')), eml_vec('c')), eml_vec('d'))"
        L_shallow = compute_layout(parse_eml_tree("EML: eml_vec('a')").to_dict(),
                                   canvas=(720, 200))
        L_deep = compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict(),
                                canvas=(720, 200))
        assert L_deep["canvas"]["height"] >= L_shallow["canvas"]["height"]

    def test_canvas_does_not_shrink_below_input(self):
        L = compute_layout(_tree().to_dict(), canvas=(800, 600))
        assert L["canvas"]["width"] >= 800
        assert L["canvas"]["height"] >= 600


# ── Validation ───────────────────────────────────────────────────────────────

class TestValidation:

    def test_invalid_direction_raises(self):
        with pytest.raises(ValueError):
            compute_layout(_tree().to_dict(), direction="diagonal")

    def test_invalid_edge_style_raises(self):
        with pytest.raises(ValueError):
            compute_layout(_tree().to_dict(), edge_style="zigzag")
