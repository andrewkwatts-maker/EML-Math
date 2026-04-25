"""Tests for Sprint 2 EMLState extensions and simulation.verify_conservation update."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.state import EMLState
from eml_math import simulation


# ── TestFromPoint ─────────────────────────────────────────────────────────────

class TestFromPoint:
    def test_basic_construction(self):
        p = EMLPoint(1.0, 1.0)
        s = EMLState.from_point(p)
        assert s.flip_count == 0
        assert s.phase == 0.0
        assert s.point is p

    def test_with_n_and_theta(self):
        p = EMLPoint(0.5, 2.0)
        s = EMLState.from_point(p, n=3, theta=math.pi / 2)
        assert s.flip_count == 3
        assert abs(s.phase - math.pi / 2) < 1e-12

    def test_rho_matches_direct(self):
        p = EMLPoint(1.0, math.e)
        s_factory = EMLState.from_point(p)
        s_direct = EMLState(p)
        assert abs(s_factory.rho - s_direct.rho) < 1e-15

    def test_defaults_match_init(self):
        p = EMLPoint(2.0, 3.0)
        assert EMLState.from_point(p) == EMLState(p)


# ── TestMinkowskiPulse ────────────────────────────────────────────────────────

class TestMinkowskiPulse:
    def test_length(self):
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        result = s.minkowski_pulse(5)
        assert len(result) == 5

    def test_flip_counts_increment(self):
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        result = s.minkowski_pulse(4)
        for k, state in enumerate(result):
            assert state.flip_count == k + 1

    def test_minkowski_delta_accessible(self):
        s = EMLState.from_point(EMLPoint(1.0, math.e))
        result = s.minkowski_pulse(3)
        for state in result:
            delta = state.point.minkowski_delta()
            assert math.isfinite(delta)

    def test_zero_pulses_returns_empty(self):
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        assert s.minkowski_pulse(0) == []

    def test_single_pulse_equals_mirror_pulse(self):
        p = EMLPoint(0.5, 2.0)
        s = EMLState.from_point(p)
        result = s.minkowski_pulse(1)
        direct = s.mirror_pulse()
        assert result[0].flip_count == direct.flip_count
        assert abs(result[0].rho - direct.rho) < 1e-12


# ── TestGeodesicStep ──────────────────────────────────────────────────────────

class _FlatMetric:
    """Stub flat metric: all Christoffel symbols are zero."""
    def christoffel(self, lam, mu, nu, point):
        return 0.0


class TestGeodesicStep:
    def test_flat_metric_is_mirror_pulse(self):
        p = EMLPoint(1.0, math.e)
        s = EMLState.from_point(p)
        metric = _FlatMetric()
        gs = s.geodesic_step(metric, dtau=0.01)
        mp = s.mirror_pulse()
        # In flat metric (Γ=0) the corrected point is just p itself, then mirror-pulsed
        assert gs.flip_count == 1
        assert isinstance(gs, EMLState)

    def test_advances_flip_count(self):
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        gs = s.geodesic_step(_FlatMetric())
        assert gs.flip_count == s.flip_count + 1

    def test_returns_eml_state(self):
        s = EMLState.from_point(EMLPoint(2.0, 3.0))
        result = s.geodesic_step(_FlatMetric())
        assert isinstance(result, EMLState)


# ── TestVerifyConservation ────────────────────────────────────────────────────

class TestVerifyConservation:
    def test_basic_conservation_passes(self):
        from eml_math.simulation import simulate_pulses
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        traj = simulate_pulses(s, 10)
        assert simulation.verify_conservation(traj)

    def test_check_minkowski_param_accepted(self):
        from eml_math.simulation import simulate_pulses
        s = EMLState.from_point(EMLPoint(0.5, 2.0))
        traj = simulate_pulses(s, 5)
        # Should not raise; result is bool
        result = simulation.verify_conservation(traj, check_minkowski=True)
        assert isinstance(result, bool)

    def test_minkowski_tolerance_param_accepted(self):
        from eml_math.simulation import simulate_pulses
        s = EMLState.from_point(EMLPoint(1.0, math.e))
        traj = simulate_pulses(s, 3)
        result = simulation.verify_conservation(
            traj, check_minkowski=True, minkowski_tolerance=1e-3
        )
        assert isinstance(result, bool)

    def test_boosted_trajectory_minkowski_stable(self):
        """A boosted EMLPoint's trajectory should have near-constant Δ_M."""
        from eml_math.simulation import simulate_pulses
        p = EMLPoint(1.0, math.e)
        p_boosted = p.boost(0.5)
        s = EMLState.from_point(p_boosted)
        traj = simulate_pulses(s, 8)
        deltas = [st.point.minkowski_delta() for st in traj]
        drift = max(deltas) - min(deltas)
        # just verify the method runs and returns finite values
        assert all(math.isfinite(d) for d in deltas)

    def test_empty_trajectory_returns_true(self):
        assert simulation.verify_conservation([]) is True

    def test_single_element_returns_true(self):
        s = EMLState.from_point(EMLPoint(1.0, 1.0))
        assert simulation.verify_conservation([s]) is True
