"""
Palette helpers for the EML formula renderer.

Pure stdlib (hashlib + math). No third-party imports.
"""
from __future__ import annotations

import hashlib
from typing import Sequence, Tuple

__all__ = [
    "DEFAULT_PALETTE",
    "pastel_for_label",
    "blend",
    "rgb_hex",
]


# Distinct, vivid hues that survive averaging without becoming muddy.
DEFAULT_PALETTE: Sequence[Tuple[int, int, int]] = (
    (242, 165, 152),   # soft coral
    (152, 209, 219),   # soft sky
    (175, 188, 232),   # soft periwinkle
    (231, 184, 220),   # soft pink
    (181, 220, 174),   # soft sage
    (242, 207, 158),   # soft peach
    (200, 178, 230),   # soft lilac
    (216, 200, 168),   # soft sand
    (165, 207, 207),   # soft mint
    (236, 196, 178),   # soft apricot
)


def pastel_for_label(label: str) -> Tuple[int, int, int]:
    """Deterministic soft-pastel colour for *label*.

    Equal labels always map to the same colour, even when we run out of
    palette entries. The MD5 hash makes this stable across runs.
    """
    h = int(hashlib.md5(label.encode("utf-8")).hexdigest()[:8], 16)
    hue = (h % 360) / 360.0
    return _hsl_to_rgb(hue, 0.45, 0.78)


def _hsl_to_rgb(h: float, s: float, l: float) -> Tuple[int, int, int]:
    """h, s, l in [0, 1] → (r, g, b) in [0, 255]."""
    if s == 0:
        v = int(round(l * 255))
        return (v, v, v)

    def _hue(p: float, q: float, t: float) -> float:
        t %= 1.0
        if t < 1 / 6: return p + (q - p) * 6 * t
        if t < 1 / 2: return q
        if t < 2 / 3: return p + (q - p) * (2 / 3 - t) * 6
        return p

    q = l + s - l * s if l < 0.5 else l * (1 - s) + s
    p = 2 * l - q
    r = _hue(p, q, h + 1 / 3)
    g = _hue(p, q, h)
    b = _hue(p, q, h - 1 / 3)
    return (int(round(r * 255)), int(round(g * 255)), int(round(b * 255)))


def blend(*colors: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Component-wise mean of one or more RGB colours."""
    if not colors:
        return (128, 128, 128)
    n = len(colors)
    return (
        int(round(sum(c[0] for c in colors) / n)),
        int(round(sum(c[1] for c in colors) / n)),
        int(round(sum(c[2] for c in colors) / n)),
    )


def rgb_hex(rgb: Tuple[int, int, int]) -> str:
    """``(r, g, b)`` → ``"#rrggbb"``."""
    r, g, b = rgb
    return f"#{r:02x}{g:02x}{b:02x}"
