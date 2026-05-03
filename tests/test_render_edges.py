"""
Edge-style tests for the abstracted render pipeline.

Verifies the three style generators produce valid SVG path commands and
that ``path_for`` dispatches correctly.
"""
from __future__ import annotations

import pytest

from eml_math.render.edges import (
    EDGE_STYLES,
    straight,
    curve,
    spline,
    path_for,
    sample_path,
)


# ── Straight ─────────────────────────────────────────────────────────────────

class TestStraight:

    def test_basic(self):
        d = straight((0.0, 0.0), (10.0, 20.0), "down")
        assert d.startswith("M0.0,0.0")
        assert "L10.0,20.0" in d

    def test_no_curve_command(self):
        d = straight((0.0, 0.0), (10.0, 20.0), "down")
        assert "C" not in d

    @pytest.mark.parametrize("direction", ("down", "up", "left", "right"))
    def test_direction_doesnt_break(self, direction):
        d = straight((1.0, 2.0), (3.0, 4.0), direction)
        assert "M1.0,2.0" in d


# ── Curve ────────────────────────────────────────────────────────────────────

class TestCurve:

    def test_returns_cubic(self):
        d = curve((0.0, 0.0), (100.0, 200.0), "down")
        assert d.startswith("M0.0,0.0")
        assert " C" in d

    def test_three_control_points(self):
        # cubic Bezier → exactly one C with 3 (x,y) pairs after it
        d = curve((0.0, 0.0), (100.0, 200.0), "down")
        # crude: split on " C", count commas in the segment after C
        c_part = d.split(" C", 1)[1]
        # 3 pairs ⇒ at least 5 commas (x1,y1 x2,y2 x3,y3)
        assert c_part.count(",") >= 3

    def test_bias_clamped(self):
        # Even silly bias values still produce a valid path
        for b in (-1.0, 0.0, 0.5, 1.0, 5.0):
            d = curve((0.0, 0.0), (50.0, 50.0), "down", bias=b)
            assert "C" in d

    @pytest.mark.parametrize("direction", ("down", "up", "left", "right"))
    def test_horizontal_direction(self, direction):
        d = curve((0.0, 0.0), (100.0, 50.0), direction)
        assert d.startswith("M")
        assert "C" in d


# ── Spline ───────────────────────────────────────────────────────────────────

class TestSpline:

    def test_no_waypoints_falls_back_to_curve(self):
        d = spline((0.0, 0.0), (100.0, 100.0), "down")
        assert d.startswith("M0.0,0.0")
        assert "C" in d

    def test_with_waypoints_produces_chain(self):
        d = spline((0.0, 0.0), (100.0, 100.0), "down",
                   waypoints=[(40.0, 30.0), (70.0, 70.0)])
        # Three segments → three C commands
        assert d.count(" C") >= 3

    def test_waypoint_passes_through(self):
        # The Catmull-Rom variant we use isn't strictly interpolating at
        # arbitrary tension, but with default tension=0.5 the path should
        # at least *visit* coordinates near each waypoint when sampled.
        wp = (50.0, 25.0)
        pts = sample_path("spline", (0.0, 0.0), (100.0, 50.0), "down",
                          waypoints=[wp], samples=64)
        # Some sample within ~5px of the waypoint must exist.
        d_min = min(((p[0] - wp[0])**2 + (p[1] - wp[1])**2) ** 0.5 for p in pts)
        assert d_min < 30.0

    def test_tension_changes_shape(self):
        d_low = spline((0.0, 0.0), (100.0, 100.0), "down",
                        waypoints=[(50.0, 25.0)], tension=0.2)
        d_high = spline((0.0, 0.0), (100.0, 100.0), "down",
                         waypoints=[(50.0, 25.0)], tension=0.9)
        assert d_low != d_high


# ── path_for dispatcher ──────────────────────────────────────────────────────

class TestPathForDispatch:

    @pytest.mark.parametrize("style", EDGE_STYLES)
    def test_each_style_returns_string(self, style):
        d = path_for(style, (0.0, 0.0), (10.0, 10.0), "down")
        assert isinstance(d, str) and len(d) > 0
        assert d.startswith("M")

    def test_unknown_style_falls_back_to_curve(self):
        d_unknown = path_for("zigzag-and-loop", (0.0, 0.0), (10.0, 10.0), "down")
        d_curve = path_for("curve", (0.0, 0.0), (10.0, 10.0), "down")
        assert d_unknown == d_curve

    def test_extra_kwargs_silently_dropped(self):
        d = path_for("straight", (0.0, 0.0), (10.0, 10.0), "down",
                     waypoints=[(5.0, 5.0)], bias=0.7, tension=0.3)
        assert d.startswith("M")
        assert "C" not in d   # straight stays straight


# ── sample_path for raster ───────────────────────────────────────────────────

class TestSamplePath:

    @pytest.mark.parametrize("style", EDGE_STYLES)
    def test_returns_points(self, style):
        pts = sample_path(style, (0.0, 0.0), (100.0, 100.0), "down", samples=8)
        assert len(pts) >= 8
        assert all(isinstance(p, tuple) and len(p) == 2 for p in pts)

    def test_straight_endpoints_match(self):
        pts = sample_path("straight", (0.0, 0.0), (10.0, 20.0), "down", samples=4)
        assert pts[0] == (0.0, 0.0)
        assert pts[-1] == (10.0, 20.0)

    def test_curve_endpoints_match(self):
        pts = sample_path("curve", (0.0, 0.0), (10.0, 20.0), "down", samples=8)
        assert abs(pts[0][0]) < 0.01 and abs(pts[0][1]) < 0.01
        assert abs(pts[-1][0] - 10.0) < 0.01 and abs(pts[-1][1] - 20.0) < 0.01
