"""
Trajectory generation and analysis for TensionKnot sequences.

All functions work in continuous mode by default (D=None on the initial knot).
Discrete mode is inherited from the knot's EMLPoint.D setting.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.state import EMLState
from eml_math.point import EMLPoint


def simulate_pulses(
    knot: EMLState,
    n_pulses: int,
    precision: str = "float",
) -> list[TensionKnot]:
    """
    Run n_pulses Mirror-Pulses from an initial EMLState.

    Parameters
    ----------
    knot : EMLState
        Initial state. Use D=None (default) for continuous mode,
        or pass a knot with D set for discrete mode.
    n_pulses : int
        Number of Mirror-Pulse iterations to run.
    precision : str
        "float" (default) or "mpmath" for arbitrary-precision arithmetic.
        mpmath mode requires D to be set and mpmath to be installed.

    Returns
    -------
    list[TensionKnot]
        Length n_pulses + 1 — includes the initial state at index 0.
    """
    if precision == "mpmath":
        return _simulate_mpmath(knot, n_pulses)

    trajectory = [knot]
    current = knot
    for _ in range(n_pulses):
        current = current.mirror_pulse()
        trajectory.append(current)
    return trajectory


def simulate_flips(knot: EMLState, n_flips: int) -> list[TensionKnot]:
    """
    Run n_flips complete 3:1 Flip cycles.

    Each flip = 4 Mirror-Pulses. Returns n_flips + 1 states
    (one per completed flip cycle, plus the initial state).
    """
    trajectory = [knot]
    current = knot
    for _ in range(n_flips):
        current = current.three_one_flip()
        trajectory.append(current)
    return trajectory


def quantized_trajectory(
    a0: int,
    b0: int,
    n_pulses: int,
    D: float = 6.187e34,
) -> list[tuple[int, int]]:
    """
    Integer-pair trajectory in discrete mode (Axioms 6-8).

    Starts from integer pair (a0, b0), applies:
        a_{t+1} = b_t
        b_{t+1} = round(T_{t+1} * D)

    This is the pure discrete formulation from MPM.txt, independent of
    EMLState. Useful for reproducing the D=100 table from the document.

    Parameters
    ----------
    a0, b0 : int
        Initial integer pair.
    n_pulses : int
        Number of discrete steps.
    D : float
        Quantization scale.

    Returns
    -------
    list[tuple[int, int]]
        Length n_pulses + 1, each element is (a_t, b_t).
    """
    from eml_math.constants import OVERFLOW_THRESHOLD
    pairs = [(a0, b0)]
    a, b = a0, b0
    for _ in range(n_pulses):
        x = a / D
        y = b / D
        if x > OVERFLOW_THRESHOLD:
            x = math.log(x)  # Slipping Wheel guard
        y_safe = abs(y) if y <= 0 else y
        if y_safe == 0:
            y_safe = 1e-300
        T = math.exp(x) - math.log(y_safe)
        a_new = b
        b_new = round(T * D)
        a, b = a_new, b_new
        pairs.append((a, b))
    return pairs


def tension_series(trajectory: list[TensionKnot]) -> list[float]:
    """Extract the scalar tension T at each step of a trajectory."""
    return [k.point.tension() for k in trajectory]


def rho_series(trajectory: list[TensionKnot]) -> list[float]:
    """Extract the tension density ρ = |T| at each step."""
    return [k.rho for k in trajectory]


def phase_series(trajectory: list[TensionKnot]) -> list[float]:
    """Extract the phase θ at each step."""
    return [k.phase for k in trajectory]


def verify_conservation(
    trajectory: list[TensionKnot],
    tolerance: float = 1e-9,
    check_minkowski: bool = False,
    minkowski_tolerance: float = 1e-6,
    c: float = 1.0,
) -> bool:
    """
    Verify Axiom 10 (Conservation of Tension) holds at every step.

    Axiom 10: T_{t+1} + x_t = exp(x_t)
    Equivalently: conserves_tension() on each consecutive pair.

    Parameters
    ----------
    trajectory : list[EMLState]
        Sequence of states to check.
    tolerance : float
        Tolerance for Axiom-10 tension conservation.
    check_minkowski : bool
        When True, additionally verify that the Minkowski interval Δ_M is
        approximately constant across all states (geodesic invariance check).
    minkowski_tolerance : float
        Maximum permitted drift in Δ_M across the trajectory.
    c : float
        Speed-of-light scale passed to ``minkowski_delta()``.

    Returns
    -------
    bool
        True if all checks pass within their respective tolerances.
    """
    for i in range(len(trajectory) - 1):
        if not trajectory[i].point.conserves_tension(trajectory[i + 1].point, tol=tolerance):
            return False
    if check_minkowski and len(trajectory) > 1:
        deltas = [s.point.minkowski_delta(c=c) for s in trajectory]
        if max(deltas) - min(deltas) > minkowski_tolerance:
            return False
    return True


def frame_shift_count(trajectory: list[TensionKnot]) -> int:
    """
    Count how many Mirror-Pulses triggered a frame shift (y ≤ 0 condition).

    A frame shift is detected when x_{t+1} ≠ y_t (i.e., |y_t| was used instead).
    Only meaningful in discrete mode; in continuous mode all pulses may shift.
    """
    count = 0
    for i in range(len(trajectory) - 1):
        y_prev = trajectory[i].point.y
        x_next = trajectory[i + 1].point.x
        if abs(x_next - y_prev) > 1e-12:
            count += 1
    return count


def find_resonance_bands(
    trajectory: list[TensionKnot],
    tolerance: float = 1e-9,
) -> list[tuple[int, int]]:
    """
    Find step pairs (i, j) where the two knots resonate (Axiom 14).

    Resonance: tension ratios match — the MPM analog of equality.
    Only checks pairs within the trajectory (not exhaustive cross-product).

    Returns
    -------
    list[tuple[int, int]]
        Pairs of indices where resonance holds.
    """
    bands = []
    n = len(trajectory)
    for i in range(n):
        for j in range(i + 1, n):
            if trajectory[i].point.resonates_with(trajectory[j].point, tol=tolerance):
                bands.append((i, j))
    return bands


# ── mpmath precision mode (discrete/Planck-scale) ────────────────────────────

def _simulate_mpmath(knot: EMLState, n_pulses: int) -> list[TensionKnot]:
    """High-precision simulation using mpmath. Requires D to be set."""
    try:
        import mpmath as mp
    except ImportError as exc:
        raise ImportError(
            "mpmath is required for precision='mpmath'. "
            "Install with: pip install mpmath"
        ) from exc

    mp.mp.dps = 50  # 50 decimal places

    D = knot.point.D
    if D is None:
        raise ValueError(
            "precision='mpmath' requires a TensionKnot with D set. "
            "Use EMLPoint(x, y, D=your_D) to enable discrete mode."
        )

    trajectory = [knot]
    x = mp.mpf(knot.point.x)
    y = mp.mpf(knot.point.y)
    n = knot.flip_count
    theta = knot.phase

    for _ in range(n_pulses):
        y_safe = abs(y) if y <= 0 else y
        if y_safe == 0:
            y_safe = mp.mpf("1e-300")
        T = mp.exp(x) - mp.log(y_safe)
        x_new = y_safe
        y_new = T
        # Quantize
        b_new = int(mp.nint(y_new * D))
        y_new = mp.mpf(b_new) / D
        x = x_new
        y = y_new
        n += 1
        theta = (theta + math.pi / 2) % (2 * math.pi)
        point = EMLPoint(float(x), float(y), D=D)
        trajectory.append(EMLState(point, n=n, theta=theta))

    return trajectory
