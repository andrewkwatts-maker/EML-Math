"""
eml_math — EML Mathematics  (v1.0.0)

The EML Sheffer operator  eml(x, y) = exp(x) − ln(y)  is the universal
primitive for all elementary mathematics (arXiv:2603.21852v2).

Core types
----------
EMLPoint  — the EML computation node: EMLPoint(x, y).eml() = exp(x) − ln(y)
EMLPair   — two-real replacement for complex numbers
EMLState  — full iteration state Φ(n, ρ, θ) for EML dynamics

Geometry & physics layer (v1.0.0)
----------------------------------
MetricTensor        — general-relativistic spacetime metrics (flat, Schwarzschild,
                      FLRW, AdS₅×S⁵, Calabi–Yau, G₂-holonomy, …)
FourMomentum        — relativistic four-momentum with Lorentz boost
MinkowskiFourVector — (3+1)D Minkowski four-vector with boost
EMLMultivector      — Clifford algebra Cl(p,q) with geometric product
Octonion            — 8-component non-associative normed division algebra
EMLNDVector         — N-dimensional EML lattice vector; E₈ and Leech lattice helpers

Formula output formats
----------------------
``decompress(result, fmt=...)`` supports six rendering targets:

- ``"math"``    — clean standard notation  ``exp(x) - ln(x)``
- ``"latex"``   — LaTeX commands           ``\\exp(x) - \\ln(x)``  (for ``$...$``)
- ``"mathjax"`` — inline MathJax           ``\\( \\exp(x) - \\ln(x) \\)``
- ``"mathml"``  — MathML markup            ``<math>...</math>``
- ``"python"``  — runnable Python          ``import math; f = lambda x: ...``
- ``"eml"``     — raw EML formula string

C / C++ / Rust API
------------------
The compiled Python wheel does not include the C shared library.
To build ``eml_math.dll`` / ``libeml_math.so`` from source:

    git clone https://github.com/andrewkwatts-maker/EML-Math
    cd EML-Math
    cargo build --release -p eml_c_api

The generated header ``c_api/eml_math.h`` documents all exported functions.
New in v1.0.0: full arithmetic (eml_exp, eml_ln, eml_add, eml_mul, eml_div,
eml_sqrt, eml_pow, eml_neg, eml_inv, eml_sin, eml_cos, eml_tan, eml_sinh,
eml_cosh, eml_tanh, …) plus batch variants (eml_exp_batch, eml_mul_batch, …).

Rust batch API (Python)
-----------------------
``eml_math.eml_core`` exposes Rayon-parallel batch operators:
``exp_n``, ``ln_n``, ``add_n``, ``sub_n``, ``mul_n``, ``div_n``,
``sqrt_n``, ``sin_n``, ``cos_n``, ``tension_n``, ``pow_n``.

Quick start
-----------
>>> from eml_math import EMLPoint, EMLPair, EMLState, simulate_pulses
>>> import math
>>>
>>> EMLPoint(1, 1).eml()               # e  =  eml(1, 1)
2.718281828459045
>>>
>>> # Minkowski invariant under Lorentz boost
>>> p = EMLPoint(1.0, math.e)
>>> p2 = p.boost(0.693)
>>> abs(p.minkowski_delta() - p2.minkowski_delta()) < 1e-10
True
>>>
>>> # Schwarzschild metric geodesic step
>>> from eml_math.metric import MetricTensor
>>> m = MetricTensor.schwarzschild(rs=2.0)
>>> from eml_math import EMLState
>>> s = EMLState.from_point(EMLPoint(3.0, 1.0))
>>> s2 = s.geodesic_step(m, dtau=0.005)
>>> isinstance(s2, EMLState)
True
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
from eml_math.state import EMLState
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

# Formula discovery / equation compression
from eml_math.discover import (
    Searcher,
    SearchResult,
    compress,
    recognize,
    compress_str,
    compress_latex,
    decompress,
    get,
)

# v1.0.0 geometry and physics layer
from eml_math.momentum import FourMomentum
from eml_math.discrete import planck_delta, lattice_distance, is_lattice_neighbor
from eml_math.metric import MetricTensor
from eml_math.ndim import EMLNDVector, e8_lattice_points, leech_lattice_points
from eml_math.octonion import Octonion, basis_octonion
from eml_math.fourvector import MinkowskiFourVector
from eml_math.geometric_algebra import EMLMultivector

iterate = simulate_pulses

__version__ = "1.0.0"
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
    # Formula discovery / equation compression
    "Searcher",
    "SearchResult",
    "compress",
    "recognize",
    "compress_str",
    "compress_latex",
    "decompress",
    "get",
    # Geometry and physics (v1.0.0)
    "FourMomentum",
    "planck_delta",
    "lattice_distance",
    "is_lattice_neighbor",
    "MetricTensor",
    "EMLNDVector",
    "e8_lattice_points",
    "leech_lattice_points",
    "Octonion",
    "basis_octonion",
    "MinkowskiFourVector",
    "EMLMultivector",
]
