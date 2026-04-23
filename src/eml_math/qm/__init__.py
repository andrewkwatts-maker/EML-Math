"""
MPM Quantum Mechanics — TensionPair replaces Python complex throughout.

All quantum operations map to TensionPair rotations:
    i · ψ                →  psi.rotate_phase(π/2)
    e^{-iEt/ħ} |E(0)⟩   →  psi.rotate_phase(-E*t/ħ)
    |ψ|²                 →  pair.modulus**2
    ψ*                   →  pair.conjugate()

No `import cmath` anywhere in this package.
"""
from eml_math.qm.states import (
    QuantumState,
    Qubit,
    bell_phi_plus,
    bell_phi_minus,
    bell_psi_plus,
)
from eml_math.qm.evolution import (
    evolve_eigenstate,
    evolve_superposition,
    harmonic_oscillator_energy,
    harmonic_oscillator_frequency,
    two_level_rabi,
    schrodinger_step_diagonal,
    norm_squared,
    expect_position,
)

__all__ = [
    # States
    "QuantumState",
    "Qubit",
    "bell_phi_plus",
    "bell_phi_minus",
    "bell_psi_plus",
    # Evolution
    "evolve_eigenstate",
    "evolve_superposition",
    "harmonic_oscillator_energy",
    "harmonic_oscillator_frequency",
    "two_level_rabi",
    "schrodinger_step_diagonal",
    "norm_squared",
    "expect_position",
]
