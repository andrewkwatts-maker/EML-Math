"""
Edge-path generators for the EML formula renderer.

Three styles, each a pure function ``(p1, p2, direction, **opts) -> str``
returning an SVG ``d`` attribute:

* :func:`straight` — single ``L`` line. Tightest, sharp at endpoints.
* :func:`curve`    — single cubic Bezier. Soft and natural (default).
* :func:`spline`   — Catmull-Rom through endpoints + waypoints. Routes
                     around obstacles in dense trees.

Pure stdlib (math). Renderers dispatch on ``edge["style"]`` in the layout
dict to pick the generator.
"""
from __future__ import annotations

from typing import Sequence, Tuple

__all__ = [
    "EDGE_STYLES",
    "straight",
    "curve",
    "spline",
    "path_for",
    "sample_path",
]


EDGE_STYLES = ("straight", "curve", "spline")

Pt = Tuple[float, float]


# ── Style: straight line ─────────────────────────────────────────────────────

def straight(p1: Pt, p2: Pt, direction: str = "down") -> str:
    """``M x1,y1 L x2,y2`` — tightest possible link."""
    x1, y1 = p1
    x2, y2 = p2
    return f"M{x1:.1f},{y1:.1f} L{x2:.1f},{y2:.1f}"


# ── Style: single cubic Bezier (the v1.0.0 default look) ─────────────────────

def curve(
    p1: Pt,
    p2: Pt,
    direction: str = "down",
    *,
    bias: float = 0.5,
) -> str:
    """Cubic Bezier from *p1* to *p2*.

    The two control points are placed along the **primary axis** at
    fractional distance *bias* (default 0.5 ⇒ symmetric S-curve).

    Higher bias values push controls further toward the opposite endpoint
    along the primary axis, forcing the curve to leave each endpoint more
    perpendicular to the cross axis. Useful for redirector links where
    cross-axis travel ≫ primary-axis travel.
    """
    bias = max(0.05, min(0.95, bias))
    x1, y1 = p1
    x2, y2 = p2
    if direction in ("down", "up"):
        cy1 = y1 + (y2 - y1) * bias
        cy2 = y2 - (y2 - y1) * bias
        return (f"M{x1:.1f},{y1:.1f} C{x1:.1f},{cy1:.1f} "
                f"{x2:.1f},{cy2:.1f} {x2:.1f},{y2:.1f}")
    cx1 = x1 + (x2 - x1) * bias
    cx2 = x2 - (x2 - x1) * bias
    return (f"M{x1:.1f},{y1:.1f} C{cx1:.1f},{y1:.1f} "
            f"{cx2:.1f},{y2:.1f} {x2:.1f},{y2:.1f}")


# ── Style: Catmull-Rom spline through waypoints ──────────────────────────────

def spline(
    p1: Pt,
    p2: Pt,
    direction: str = "down",
    *,
    waypoints: Sequence[Pt] = (),
    tension: float = 0.5,
) -> str:
    """Catmull-Rom spline through *p1* + *waypoints* + *p2*.

    With no waypoints, this is a single cubic Bezier whose control points
    sit at *tension* fraction along the primary axis — visually close to
    :func:`curve` with the same *bias*.

    With one or more waypoints, the curve passes **exactly** through every
    waypoint, smoothly. That makes spline edges the right choice when the
    layout pre-routes an edge around a sibling subtree.

    Implementation: Catmull-Rom is converted to a chain of cubic Beziers
    via the standard formula (each segment's control points are placed at
    ±1/(6·tension) of the chord between its neighbours, so adjacent
    segments share C1 continuity).
    """
    pts: list[Pt] = [p1, *waypoints, p2]
    if len(pts) < 2:
        return ""
    if len(pts) == 2:
        # Degenerate to a curve() call so a bare two-point spline still
        # looks like a curve, not a straight line.
        return curve(p1, p2, direction, bias=tension)

    # Pad with reflected endpoints so the first and last segments have
    # well-defined neighbours.
    p_pre = (2 * pts[0][0] - pts[1][0], 2 * pts[0][1] - pts[1][1])
    p_post = (2 * pts[-1][0] - pts[-2][0], 2 * pts[-1][1] - pts[-2][1])
    padded = [p_pre, *pts, p_post]

    parts: list[str] = [f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"]
    k = 1.0 / (6.0 * max(0.05, tension))
    for i in range(len(pts) - 1):
        p0 = padded[i]
        p1c = padded[i + 1]
        p2c = padded[i + 2]
        p3 = padded[i + 3]
        c1 = (p1c[0] + (p2c[0] - p0[0]) * k,
              p1c[1] + (p2c[1] - p0[1]) * k)
        c2 = (p2c[0] - (p3[0] - p1c[0]) * k,
              p2c[1] - (p3[1] - p1c[1]) * k)
        parts.append(
            f" C{c1[0]:.1f},{c1[1]:.1f} {c2[0]:.1f},{c2[1]:.1f} "
            f"{p2c[0]:.1f},{p2c[1]:.1f}"
        )
    return "".join(parts)


# ── Dispatcher used by renderers ─────────────────────────────────────────────

def path_for(
    style: str,
    p1: Pt,
    p2: Pt,
    direction: str = "down",
    **opts,
) -> str:
    """Generate an SVG ``d`` string for the given *style*.

    Unknown styles fall back to :func:`curve`. Unknown kwargs are silently
    dropped so renderers can pass a uniform options dict to every style.
    """
    if style == "straight":
        return straight(p1, p2, direction)
    if style == "spline":
        return spline(
            p1, p2, direction,
            waypoints=opts.get("waypoints", ()),
            tension=opts.get("tension", opts.get("bias", 0.5)),
        )
    return curve(p1, p2, direction, bias=opts.get("bias", 0.5))


# ── Sampling for raster renderers ────────────────────────────────────────────

def sample_path(
    style: str,
    p1: Pt,
    p2: Pt,
    direction: str = "down",
    *,
    samples: int = 32,
    **opts,
) -> list[Pt]:
    """Discretise a path into *samples*+1 evenly-spaced points.

    Used by raster renderers (Pillow has no SVG path support of its own).
    """
    if style == "straight":
        x1, y1 = p1
        x2, y2 = p2
        return [(x1 + (x2 - x1) * t / samples,
                 y1 + (y2 - y1) * t / samples)
                for t in range(samples + 1)]

    if style == "spline":
        # Sample each segment of the Catmull-Rom uniformly.
        wps = opts.get("waypoints", ())
        knots = [p1, *wps, p2]
        if len(knots) < 3:
            return _sample_cubic(p1, p2, direction,
                                 bias=opts.get("tension", 0.5),
                                 samples=samples)
        per = max(2, samples // (len(knots) - 1))
        out: list[Pt] = []
        for i in range(len(knots) - 1):
            seg = _sample_catmull_rom(knots, i, per,
                                       tension=opts.get("tension", 0.5))
            if i > 0:
                seg = seg[1:]   # avoid duplicating shared endpoint
            out.extend(seg)
        return out

    # default: curve
    return _sample_cubic(p1, p2, direction,
                         bias=opts.get("bias", 0.5),
                         samples=samples)


def _sample_cubic(p1: Pt, p2: Pt, direction: str, *,
                  bias: float, samples: int) -> list[Pt]:
    bias = max(0.05, min(0.95, bias))
    x1, y1 = p1
    x2, y2 = p2
    if direction in ("down", "up"):
        c1 = (x1, y1 + (y2 - y1) * bias)
        c2 = (x2, y2 - (y2 - y1) * bias)
    else:
        c1 = (x1 + (x2 - x1) * bias, y1)
        c2 = (x2 - (x2 - x1) * bias, y2)
    out: list[Pt] = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        x = (u**3 * x1 + 3*u*u*t * c1[0] + 3*u*t*t * c2[0] + t**3 * x2)
        y = (u**3 * y1 + 3*u*u*t * c1[1] + 3*u*t*t * c2[1] + t**3 * y2)
        out.append((x, y))
    return out


def _sample_catmull_rom(knots: list[Pt], i: int, samples: int, *,
                         tension: float) -> list[Pt]:
    """Sample segment i of a Catmull-Rom spline through *knots*."""
    n = len(knots)
    p0 = knots[i - 1] if i > 0 else (
        2 * knots[0][0] - knots[1][0], 2 * knots[0][1] - knots[1][1]
    )
    p1 = knots[i]
    p2 = knots[i + 1]
    p3 = knots[i + 2] if i + 2 < n else (
        2 * knots[-1][0] - knots[-2][0], 2 * knots[-1][1] - knots[-2][1]
    )
    out: list[Pt] = []
    for j in range(samples + 1):
        t = j / samples
        t2 = t * t
        t3 = t2 * t
        # Standard Catmull-Rom basis (parameter alpha=0.5 ⇒ centripetal-ish).
        a = -tension * t + 2 * tension * t2 - tension * t3
        b = 1 + (tension - 3) * t2 + (2 - tension) * t3
        c = tension * t + (3 - 2 * tension) * t2 + (tension - 2) * t3
        d = -tension * t2 + tension * t3
        x = a * p0[0] + b * p1[0] + c * p2[0] + d * p3[0]
        y = a * p0[1] + b * p1[1] + c * p2[1] + d * p3[1]
        out.append((x, y))
    return out
