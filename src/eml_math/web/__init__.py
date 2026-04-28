"""
eml_math.web — browser-side helpers shipped with the Python package.

Use :func:`get_flow_js` to read the bundled ``eml_flow.js`` source so you
can ``<script>``-include it in a generated HTML page.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["FLOW_JS_PATH", "get_flow_js"]

_HERE = Path(__file__).resolve().parent
FLOW_JS_PATH: Path = _HERE / "eml_flow.js"


def get_flow_js() -> str:
    """Return the contents of ``eml_flow.js`` as a string.

    The returned text is a UMD-style script — drop it into a ``<script>``
    tag (or write it to a file your page already loads) and call
    ``EmlFlow.renderFlowSvg(treeArr, opts)``.
    """
    return FLOW_JS_PATH.read_text(encoding="utf-8")
