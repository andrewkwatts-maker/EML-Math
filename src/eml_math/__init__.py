"""
eml_math — EML Mathematics

The EML Sheffer operator  eml(x, y) = exp(x) − ln(y)  is the universal
primitive for all elementary mathematics (arXiv:2603.21852v2).

Core types
----------
EMLPoint  — the EML computation node: EMLPoint(x, y).eml() = exp(x) − ln(y)
EMLPair   — two-real replacement for complex numbers
EMLState  — full iteration state Φ(n, ρ, θ) for EML dynamics

Quick start
-----------
>>> from eml_math import EMLPoint, EMLPair, EMLState, simulate_pulses
>>> import math
>>>
>>> EMLPoint(1, 1).eml()               # e  =  eml(1, 1)
2.718281828459045
>>> EMLPoint(2, 1).eml()               # exp(2)
7.38905609893065
>>>
>>> # Phase rotation (quantum evolution)
>>> p = EMLPair.from_values(1.0, 0.0)
>>> p.rotate_phase(math.pi / 2).imag_tension   # sin(π/2) = 1
1.0
>>>
>>> # EML iteration trajectory
>>> s = EMLState(EMLPoint(1.0, 1.0))
>>> traj = simulate_pulses(s, n_pulses=8)
>>> [f"{st.rho:.4f}" for st in traj]
[...]
"""

from eml_math.constants import (
    PLANCK_D,
    PLANCK_LENGTH,
    PLANCK_ENERGY,
    FLIP_YIELD,
    FLIP_RATIO,
    OVERFLOW_THRESHOLD,
    DEFAULT_D,
)

from eml_math.point import EMLPoint, _VarNode
from eml_math.state import EMLState, EMLKnot
from eml_math.pair import EMLPair
from eml_math.simulation import (
    simulate_pulses,
    simulate_flips,
    quantized_trajectory,
    tension_series,
    rho_series,
    phase_series,
    verify_conservation,
    frame_shift_count,
    find_resonance_bands,
)

iterate = simulate_pulses

__version__ = "0.2.0"
__author__ = "Andrew K Watts"

__all__ = [
    # Constants
    "PLANCK_D",
    "PLANCK_LENGTH",
    "PLANCK_ENERGY",
    "FLIP_YIELD",
    "FLIP_RATIO",
    "OVERFLOW_THRESHOLD",
    "DEFAULT_D",
    # Core types
    "EMLPoint",
    "EMLPair",
    "EMLState",
    "EMLKnot",
    "iterate",
    # Simulation
    "simulate_pulses",
    "simulate_flips",
    "quantized_trajectory",
    "tension_series",
    "rho_series",
    "phase_series",
    "verify_conservation",
    "frame_shift_count",
    "find_resonance_bands",
]
