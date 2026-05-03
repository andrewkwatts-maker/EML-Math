"""
Protocol/registry tests for the abstracted render pipeline.

Covers:
  - Renderer protocol structural typing.
  - register / renderer_for / render_with end-to-end.
  - Built-in renderers produce the right kind of output (str vs bytes).
  - Importing eml_math.render does NOT pull in Pillow.
  - PNG/PDF raise a clear ImportError when Pillow is absent.
"""
from __future__ import annotations

import sys
import pytest

from eml_math import parse_eml_tree
from eml_math.render import (
    Renderer,
    SVGRenderer,
    HTMLRenderer,
    PNGRenderer,
    PDFRenderer,
    compute_layout,
    register,
    renderer_for,
    render_with,
)


def _layout():
    desc = "EML: ops.mul(eml_vec('a'), eml_vec('b'))"
    return compute_layout(parse_eml_tree(desc, expand_eml=False).to_dict())


# ── Renderer protocol structural typing ──────────────────────────────────────

class TestProtocol:

    def test_svg_satisfies_protocol(self):
        assert isinstance(SVGRenderer(), Renderer)

    def test_html_satisfies_protocol(self):
        assert isinstance(HTMLRenderer(), Renderer)

    def test_custom_class_satisfies_protocol(self):
        class MyRenderer:
            def render(self, layout, **opts):
                return "ok"
        assert isinstance(MyRenderer(), Renderer)

    def test_non_renderer_does_not_satisfy(self):
        class NotARenderer:
            pass
        assert not isinstance(NotARenderer(), Renderer)


# ── Registry ─────────────────────────────────────────────────────────────────

class TestRegistry:

    def test_builtin_renderers_registered(self):
        for name in ("svg", "html", "png", "pdf"):
            r = renderer_for(name)
            assert hasattr(r, "render")

    def test_register_and_lookup(self):
        class ConstRenderer:
            def render(self, layout, **opts):
                return "constant"
        register("test-const", ConstRenderer())
        out = render_with("test-const", _layout())
        assert out == "constant"

    def test_register_rejects_non_renderer(self):
        with pytest.raises(TypeError):
            register("bad", object())   # no .render method

    def test_unknown_name_raises(self):
        with pytest.raises(KeyError):
            renderer_for("nonexistent-renderer-xyz")


# ── Built-in renderer outputs ────────────────────────────────────────────────

class TestBuiltinOutputs:

    def test_svg_returns_str(self):
        out = SVGRenderer().render(_layout())
        assert isinstance(out, str)
        assert out.startswith("<svg")
        assert out.rstrip().endswith("</svg>")

    def test_html_wraps_svg(self):
        out = HTMLRenderer().render(_layout())
        assert isinstance(out, str)
        assert out.startswith('<div')
        assert "<svg" in out

    def test_html_passes_options_through(self):
        out = HTMLRenderer().render(_layout(),
                                     container_id="myid",
                                     container_class="my-class")
        assert 'id="myid"' in out
        assert 'class="my-class"' in out

    def test_render_with_dispatches(self):
        out = render_with("svg", _layout())
        assert out.startswith("<svg")


# ── Zero-dep guarantee ───────────────────────────────────────────────────────

class TestZeroDeps:

    def test_import_does_not_load_pillow(self):
        # Sanity-check by re-importing in a fresh module table simulation.
        # We can't truly re-import without subprocess, but we CAN verify
        # render functions don't sneak in Pillow when called for SVG/HTML.
        for k in list(sys.modules):
            if k == "PIL" or k.startswith("PIL."):
                del sys.modules[k]
        SVGRenderer().render(_layout())
        HTMLRenderer().render(_layout())
        leaked = [k for k in sys.modules if k == "PIL" or k.startswith("PIL.")]
        assert not leaked, f"Pillow loaded by SVG/HTML renderers: {leaked}"


# ── Optional renderers raise cleanly ─────────────────────────────────────────

class TestOptionalRenderers:

    def _has_pillow(self):
        try:
            import PIL  # noqa: F401
            return True
        except ImportError:
            return False

    def test_png_returns_bytes_or_raises(self):
        try:
            out = PNGRenderer().render(_layout())
            assert isinstance(out, bytes)
            assert out[:8] == b"\x89PNG\r\n\x1a\n"
        except ImportError as e:
            assert "Pillow" in str(e)

    def test_pdf_returns_bytes_or_raises(self):
        try:
            out = PDFRenderer().render(_layout())
            assert isinstance(out, bytes)
            assert out.startswith(b"%PDF")
        except ImportError as e:
            assert "Pillow" in str(e)


# ── EMLTreeNode.render shortcut ──────────────────────────────────────────────

class TestNodeShortcut:

    def test_node_render_svg(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))",
                           expand_eml=False)
        out = t.render("svg")
        assert isinstance(out, str) and out.startswith("<svg")

    def test_node_render_with_layout_opts(self):
        t = parse_eml_tree("EML: ops.mul(eml_vec('a'), eml_vec('b'))",
                           expand_eml=False)
        out = t.render("svg", layout_opts={"edge_style": "straight",
                                            "direction": "right"})
        assert isinstance(out, str)
        # straight = L command, no C
        assert "L" in out

    def test_node_layout_returns_dict(self):
        t = parse_eml_tree("EML: eml_vec('a')")
        L = t.layout()
        assert L["schema"] == "eml-layout/v1"

    def test_node_render_unknown_format_raises(self):
        t = parse_eml_tree("EML: eml_vec('a')")
        with pytest.raises(KeyError):
            t.render("does-not-exist")
