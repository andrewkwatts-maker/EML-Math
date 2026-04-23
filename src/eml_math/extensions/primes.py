"""Prime Tension detection — Axiom 15."""
from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from eml_math.state import EMLState


def is_prime_tension(knot: "EMLState") -> bool:
    """
    Axiom 15: returns True if the knot's tension density is indivisible (prime).

    For physical-scale D (≈ 6.187e34), uses integer primality of round(ρ·D).
    For toy-scale D or continuous mode, uses integer primality of round(ρ).

    Requires sympy for large integers (pip install eml[ext]).
    """
    rho = knot.rho
    D = knot.point.D
    n = round(rho * D) if D is not None else round(rho)
    if n < 2:
        return False
    try:
        from sympy import isprime
        return bool(isprime(n))
    except ImportError:
        # Naive trial division for small n
        if n < 2:
            return False
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
