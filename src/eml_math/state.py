"""
EMLState — the full EML iteration state Φ(n, ρ, θ).

Wraps a TensionPoint with step-count n and phase θ, implementing the
3:1 Flip cycle (Axiom 9) and the Rolling Wheel dynamics.
"""
from __future__ import annotations

import math
from typing import Optional, TYPE_CHECKING

from eml_math.point import EMLPoint
from eml_math.constants import FLIP_YIELD

if TYPE_CHECKING:
    pass

_TWO_PI = 2.0 * math.pi
_PHASE_STEP = _TWO_PI / 4.0  # quarter-phase advance per mirror pulse


class EMLState:
    """
    Full EML iteration state Φ(n, ρ, θ).

    Parameters
    ----------
    point : EMLPoint
        The paired coordinate state at this step-count.
    n : int
        Step-count — number of completed Mirror-Pulses.
    theta : float
        Phase θ in [0, 2π), advances by π/2 per pulse.

    Examples
    --------
    >>> p = EMLPoint(1.0, 1.0)
    >>> s = EMLState(p)
    >>> s.rho
    2.718281828459045
    >>> s2 = s.mirror_pulse()
    >>> s2.flip_count
    1
    """

    __slots__ = ("_point", "_n", "_theta")

    def __init__(
        self,
        point: EMLPoint,
        n: int = 0,
        theta: float = 0.0,
    ) -> None:
        self._point = point
        self._n = int(n)
        self._theta = theta % _TWO_PI

    # ── properties ────────────────────────────────────────────────────────────

    @property
    def point(self) -> TensionPoint:
        return self._point

    @property
    def rho(self) -> float:
        """Tension density ρ = |eml(x, y)| = |exp(x) − ln(y)|."""
        return abs(self._point.tension())

    @property
    def flip_count(self) -> int:
        """Number of completed Mirror-Pulses since initialisation."""
        return self._n

    @property
    def phase(self) -> float:
        """Phase θ in [0, 2π)."""
        return self._theta

    # ── iteration ─────────────────────────────────────────────────────────────

    def mirror_pulse(self) -> "EMLState":
        """One Mirror-Pulse: advance the TensionPoint and increment n."""
        new_point = self._point.mirror_pulse()
        return EMLState(
            new_point,
            n=self._n + 1,
            theta=self._theta + _PHASE_STEP,
        )

    def three_one_flip(self) -> "EMLState":
        """
        One complete 3:1 Flip cycle (Axiom 9).

        Three growth pulses followed by one mirrored reflection,
        yielding FLIP_YIELD net reality units per cycle.
        """
        state = self
        for _ in range(4):  # 3 growth + 1 reflection = 4 pulses total
            state = state.mirror_pulse()
        return state

    def tread_yield(self) -> int:
        """Net reality units accumulated since initialisation."""
        complete_flips = self._n // 4
        return complete_flips * FLIP_YIELD

    # ── state inspection ──────────────────────────────────────────────────────

    def is_prime_tension(self) -> bool:
        """True if this state's tension is an indivisible (prime) tension."""
        from eml_math.extensions.primes import is_prime_tension
        return is_prime_tension(self)

    def share_axle_with(self, other: "EMLState") -> "SharedAxle":
        """Postulate Q5: create a SharedAxle entangled pair with another state."""
        from eml_math.qm.entanglement import SharedAxle
        return SharedAxle(self, other)

    # aliases
    def pulse(self) -> "EMLState":
        return self.mirror_pulse()

    def flip(self) -> "EMLState":
        return self.three_one_flip()

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"EMLState(n={self._n}, rho={self.rho:.6g}, "
            f"theta={self._theta:.4f}, point={self._point!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLState):
            return NotImplemented
        return self._point.resonates_with(other._point) and self._n == other._n
