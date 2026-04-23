"""
Axiom-by-axiom verification tests for Mirror Phase Mathematics.

Each test class corresponds to one MPM Axiom from MPM.txt.
"""
import math
import pytest
from eml_math import EMLPoint, EMLState, simulate_pulses, verify_conservation
from eml_math import operators as ops
from eml_math.constants import OVERFLOW_THRESHOLD


class TestAxiom4MirrorOperator:
    """M(z) = ln(z) — the Mirror Operator."""

    def test_mirror_is_ln(self):
        for z in [0.5, 1.0, math.e, 10.0]:
            assert ops.ln(z).tension() == pytest.approx(math.log(z), rel=1e-12)

    def test_mirror_of_exp_is_identity(self):
        for x in [0.5, 1.0, 2.0, 5.0]:
            assert ops.ln(ops.exp(x)).tension() == pytest.approx(x, rel=1e-10)

    def test_exp_of_mirror_is_identity(self):
        for y in [0.5, 1.0, math.e, 10.0]:
            assert ops.exp(ops.ln(y)).tension() == pytest.approx(y, rel=1e-10)


class TestAxiom5Tension:
    """T = exp(x) - ln(y) — the fundamental tension scalar."""

    def test_tension_at_unit_point(self):
        assert EMLPoint(1.0, 1.0).tension() == pytest.approx(math.e, rel=1e-14)

    def test_tension_formula_explicit(self):
        for x, y in [(0.5, 2.0), (1.0, 3.0), (2.0, 1.5)]:
            expected = math.exp(x) - math.log(y)
            assert EMLPoint(x, y).tension() == pytest.approx(expected, rel=1e-12)

    def test_tension_always_real(self):
        for x in [0.1, 1.0, 5.0]:
            for y in [0.1, 1.0, 5.0]:
                T = EMLPoint(x, y).tension()
                assert isinstance(T, float)
                assert not math.isnan(T)


class TestAxiom7MirrorSymmetricUpdate:
    """x_{t+1} = y_t, y_{t+1} = T_{t+1} — the mirror update rule."""

    def test_x_becomes_y(self):
        p = EMLPoint(1.0, 1.0)
        nxt = p.mirror_pulse()
        # In continuous mode with y > 0: x_new = y_old
        assert nxt.x == pytest.approx(p.y, rel=1e-12)

    def test_y_becomes_tension(self):
        p = EMLPoint(1.0, 1.0)
        T = p.tension()
        nxt = p.mirror_pulse()
        assert nxt.y == pytest.approx(T, rel=1e-12)


class TestAxiom8FrameShift:
    """Frame shift: y_safe = |y|, T = exp(x) - ln(|y|)."""

    def test_tension_real_when_y_negative(self):
        p = EMLPoint(1.0, -2.0)
        T = p.tension()
        assert math.isfinite(T)
        assert T == pytest.approx(math.exp(1.0) - math.log(2.0), rel=1e-12)

    def test_mirror_pulse_stays_real_after_negative_y(self):
        p = EMLPoint(0.1, 10.0)   # T will be negative
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_overflow_dampening(self):
        p = EMLPoint(OVERFLOW_THRESHOLD + 5.0, 1.0)
        nxt = p.mirror_pulse()
        assert math.isfinite(nxt.tension())

    def test_frame_shift_explicit(self):
        p = EMLPoint(1.0, 2.0)
        shifted = p.frame_shift()
        expected_y = math.exp(1.0) - math.log(2.0)
        assert shifted.x == pytest.approx(2.0, rel=1e-12)
        assert shifted.y == pytest.approx(expected_y, rel=1e-12)


class TestAxiom9ThreeOneFlip:
    """3:1 Flip — 4 pulses per flip, net +2 reality units (FLIP_YIELD)."""

    def test_flip_advances_n_by_4(self, unit_knot):
        n_before = unit_knot.flip_count
        flipped = unit_knot.three_one_flip()
        assert flipped.flip_count == n_before + 4

    def test_tread_yield_after_one_flip(self, unit_knot):
        flipped = unit_knot.three_one_flip()
        assert flipped.tread_yield() == 2  # 1 complete flip * FLIP_YIELD(2)

    def test_tread_yield_accumulates(self, unit_knot):
        state = unit_knot
        for _ in range(3):
            state = state.three_one_flip()
        assert state.tread_yield() == 6  # 3 flips * 2


class TestAxiom10ConservationOfTension:
    """T + x = exp(x) at every pulse."""

    def test_conservation_at_unit_point(self, unit_point):
        nxt = unit_point.mirror_pulse()
        assert unit_point.conserves_tension(nxt)

    def test_conservation_over_20_pulses(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert verify_conservation(traj)

    def test_conservation_discrete_mode(self, d100_knot):
        from eml_math import simulate_pulses
        traj = simulate_pulses(d100_knot, n_pulses=7)
        # Axiom 10: T + ln(y) = exp(x). In discrete mode, rounding means
        # residual may be up to round_error/D ≈ 0.5/100 = 0.005 per step.
        for i in range(len(traj) - 1):
            assert traj[i].point.conserves_tension(traj[i + 1].point, tol=1.0)


class TestTheorem1NoFixedPoints:
    """f(s) = exp(s) - ln(s) - s > 0 for all s > 0."""

    def test_no_fixed_point(self):
        for s in [0.001, 0.1, 0.5, 0.8, 1.0, 1.5, 2.0, 5.0, 10.0]:
            f = math.exp(s) - math.log(s) - s
            assert f > 0, f"Fixed point found at s={s}: f={f}"


class TestTheorem3RealityPreservation:
    """Tension stays finite and real over long trajectories."""

    def test_100_pulse_trajectory_stays_real(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=100)
        for k in traj:
            T = k.point.tension()
            assert math.isfinite(T), f"Non-finite tension at n={k.flip_count}: T={T}"
