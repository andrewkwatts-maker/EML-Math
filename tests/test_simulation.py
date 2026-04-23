"""
Simulation tests including regression against MPM.txt D=100 table.
"""
import math
import pytest
from eml_math import (
    EMLPoint, EMLState,
    simulate_pulses, simulate_flips, quantized_trajectory,
    tension_series, rho_series, phase_series,
    verify_conservation, frame_shift_count,
)


class TestSimulatePulses:
    def test_length(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert len(traj) == 11  # includes initial state

    def test_flip_count_increments(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        for i, k in enumerate(traj):
            assert k.flip_count == i

    def test_first_element_is_initial(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        assert traj[0] is unit_knot

    def test_all_tensions_finite(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=50)
        assert all(math.isfinite(k.point.tension()) for k in traj)


class TestSimulateFlips:
    def test_length(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=5)
        assert len(traj) == 6

    def test_flip_count_advances_by_4(self, unit_knot):
        traj = simulate_flips(unit_knot, n_flips=3)
        assert traj[1].flip_count == 4
        assert traj[2].flip_count == 8
        assert traj[3].flip_count == 12


class TestQuantizedTrajectory:
    """Regression against the D=100 table from MPM.txt lines ~600-643."""

    def test_initial_pair(self):
        pairs = quantized_trajectory(100, 100, n_pulses=0, D=100)
        assert pairs[0] == (100, 100)

    def test_first_step_matches_document(self):
        # x0=1.0, y0=1.0, T = exp(1) - ln(1) = e ≈ 2.718
        # b1 = round(e * 100) = 272
        pairs = quantized_trajectory(100, 100, n_pulses=1, D=100)
        assert pairs[1] == (100, 272)

    def test_length(self):
        pairs = quantized_trajectory(100, 100, n_pulses=7, D=100)
        assert len(pairs) == 8

    def test_a_next_equals_b_prev(self):
        pairs = quantized_trajectory(100, 100, n_pulses=5, D=100)
        for i in range(len(pairs) - 1):
            assert pairs[i + 1][0] == pairs[i][1], (
                f"At step {i}: a_{i+1}={pairs[i+1][0]} should equal b_{i}={pairs[i][1]}"
            )


class TestSeriesExtractors:
    def test_tension_series_length(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=5)
        assert len(tension_series(traj)) == 6

    def test_rho_series_non_negative(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        assert all(r >= 0 for r in rho_series(traj))

    def test_phase_series_in_range(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=10)
        for phase in phase_series(traj):
            assert 0.0 <= phase < 2 * math.pi + 1e-9


class TestVerifyConservation:
    def test_passes_for_clean_trajectory(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=20)
        assert verify_conservation(traj)

    def test_single_step(self, unit_knot):
        traj = simulate_pulses(unit_knot, n_pulses=1)
        assert verify_conservation(traj)
