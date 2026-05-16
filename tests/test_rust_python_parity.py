# Python fallback: src/eml_math/point.py
import math
import random

import pytest

from eml_math._dispatch import _HAS_RUST, _native
from eml_math.point import EMLPoint

pytestmark = pytest.mark.skipif(not _HAS_RUST, reason="Rust extension not built")


def test_tension_leaf_parity():
    rng = random.Random(42)
    for _ in range(100):
        x = rng.uniform(-5.0, 5.0)
        y = rng.uniform(0.01, 10.0)
        py_val = math.exp(x) - math.log(y)
        rs_val = _native.EMLPoint(x, y).tension()
        pt_val = EMLPoint(x, y).tension()
        assert abs(pt_val - py_val) < 1e-12, f"Python path deviated: x={x} y={y}"
        assert abs(pt_val - rs_val) < 1e-12, f"Rust/Python mismatch: x={x} y={y}: {pt_val} vs {rs_val}"


def test_tension_overflow_guard_parity():
    # x > OVERFLOW_THRESHOLD (709.78) — Rust and Python both apply ln(x) dampening
    x, y = 800.0, 2.0
    py_pt = EMLPoint(x, y)
    rs_val = _native.EMLPoint(x, y).tension()
    pt_val = py_pt.tension()
    assert abs(pt_val - rs_val) < 1e-9, f"Overflow guard mismatch: {pt_val} vs {rs_val}"


def test_tension_negative_y_guard_parity():
    x, y = 1.0, -3.0
    pt_val = EMLPoint(x, y).tension()
    rs_val = _native.EMLPoint(x, y).tension()
    assert abs(pt_val - rs_val) < 1e-12


def test_iterate_parity():
    pt = EMLPoint(1.0, 2.0)
    for _ in range(10):
        py_next = pt.iterate()
        rs_next = _native.EMLPoint(pt._x, pt._y).mirror_pulse()
        assert abs(py_next.x - rs_next.x) < 1e-12, f"iterate x mismatch: {py_next.x} vs {rs_next.x}"
        assert abs(py_next.y - rs_next.y) < 1e-12, f"iterate y mismatch: {py_next.y} vs {rs_next.y}"
        pt = py_next


def test_iterate_overflow_guard_parity():
    pt = EMLPoint(800.0, 1.5)
    py_next = pt.iterate()
    rs_next = _native.EMLPoint(800.0, 1.5).mirror_pulse()
    assert abs(py_next.x - rs_next.x) < 1e-9
    assert abs(py_next.y - rs_next.y) < 1e-9


def test_batch_tension_n_parity():
    xs = [float(i) * 0.5 for i in range(20)]
    ys = [float(i) * 0.5 + 0.1 for i in range(20)]
    rs_batch = _native.tension_n(xs, ys)
    for x, y, rs_val in zip(xs, ys, rs_batch):
        py_val = EMLPoint(x, y).tension()
        assert abs(py_val - rs_val) < 1e-12, f"tension_n mismatch x={x} y={y}: {py_val} vs {rs_val}"


def test_discrete_mode_stays_python():
    # D is set → must NOT use Rust path (quantization logic absent from Rust)
    pt = EMLPoint(1.0, 2.0, D=100.0)
    # Should not raise; Rust path is bypassed
    nxt = pt.iterate()
    assert nxt._D == 100.0
    assert isinstance(nxt._prev_x, float)


def test_nested_stays_python():
    # Nested EMLPoint (not a leaf) → must NOT use Rust path
    inner = EMLPoint(1.0, 1.0)
    outer = EMLPoint(inner, 2.0)
    assert not outer.is_leaf()
    val = outer.tension()
    assert math.isfinite(val)
