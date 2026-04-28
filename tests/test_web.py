"""
Tests for eml_math.web — browser-side helper bundle.
"""
from pathlib import Path

import eml_math
from eml_math.web import FLOW_JS_PATH, get_flow_js


class TestPackagedJS:
    def test_path_exists(self):
        assert FLOW_JS_PATH.exists()
        assert FLOW_JS_PATH.is_file()
        assert FLOW_JS_PATH.suffix == ".js"

    def test_get_flow_js_returns_text(self):
        s = get_flow_js()
        assert isinstance(s, str)
        assert len(s) > 1000

    def test_exposes_renderFlowSvg(self):
        s = get_flow_js()
        assert "renderFlowSvg" in s
        assert "EmlFlow" in s

    def test_exposes_inflate(self):
        s = get_flow_js()
        # inflate() is essential for compact-tree round-trip in JS
        assert "inflate" in s

    def test_exposes_binarize(self):
        s = get_flow_js()
        assert "binarize" in s

    def test_kind_map_matches_python(self):
        s = get_flow_js()
        # Each KIND_CHAR entry should appear in the JS KIND_MAP
        from eml_math.tree import KIND_CHAR
        for kind, ch in KIND_CHAR.items():
            # the JS map uses 'kindname' as value — check at minimum the kind name appears
            assert kind in s, f"JS KIND_MAP missing {kind!r}"

    def test_top_level_export(self):
        # eml_math.get_flow_js exposed at package root
        assert eml_math.get_flow_js() == get_flow_js()
        assert eml_math.FLOW_JS_PATH == FLOW_JS_PATH
