"""Prime Tension detection — Axiom 15."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eml_math.state import EMLState

#: Largest ``n`` the sympy-free trial-division fallback will attempt.
#: sqrt(1e18)/2 ≈ 5e8 iterations worst case — seconds, not centuries.
_TRIAL_DIVISION_MAX: int = 10 ** 18


def is_prime_tension(knot: "EMLState") -> bool:
    """
    Axiom 15: returns True if the knot's tension density is indivisible (prime).

    For physical-scale D (≈ 6.187e34), uses integer primality of round(ρ·D).
    For toy-scale D or continuous mode, uses integer primality of round(ρ).

    Requires sympy for large integers (pip install eml[ext]).
    """
    rho = knot.rho
    D = knot.point.D
    prod = rho * D if D is not None else rho
    # round() raises OverflowError on inf and ValueError on NaN; a primality
    # predicate should answer False for a non-finite tension, not explode.
    if not math.isfinite(prod):
        return False
    n = round(prod)
    if n < 2:
        return False
    try:
        from sympy import isprime
        return bool(isprime(n))
    except ImportError:
        # Naive trial division — only viable for small n. At physical-scale
        # D (~6.187e34) the loop would need ~1.2e17 iterations, so it is
        # bounded and refuses rather than hanging the caller.
        if n > _TRIAL_DIVISION_MAX:
            raise RuntimeError(
                f"is_prime_tension: n={n} exceeds the trial-division limit "
                f"({_TRIAL_DIVISION_MAX}). Install sympy (pip install eml[ext]) "
                "for large-integer primality."
            )
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        i = 3
        while i * i <= n:
            if n % i == 0:
                return False
            i += 2
        return True
