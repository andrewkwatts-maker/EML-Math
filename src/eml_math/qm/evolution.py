"""
Quantum evolution functions — all via EMLPair, no complex type.

The Schrödinger equation iħ ∂ψ/∂t = Hψ has three levels of treatment:

1. Energy eigenstates (exact):
   |E(t)⟩ = e^{-iEt/ħ}|E(0)⟩ = |E(0)⟩.rotate_phase(-Et/ħ)
   One function call. No ODE. No Euler formula.

2. Diagonal Hamiltonian (exact):
   ψ_n(t+dt) = e^{-iV_n·dt/ħ}ψ_n(t) = ψ_n.rotate_phase(-V_n·dt/ħ)
   Loop over sites, one rotate_phase per site.

3. Full kinetic + potential (Trotter split-operator):
   exp(-iHdt/ħ) ≈ exp(-iKdt/2ħ)·exp(-iVdt/ħ)·exp(-iKdt/2ħ)
   Kinetic part requires momentum-space evolution (FFT needed, optional).

In all cases the imaginary unit i appears only as a rotation angle ±π/2
inside rotate_phase(). Python's complex type is never used.

Harmonic oscillator
-------------------
Energy levels: E_n = ħω(n + 1/2)
Ground state frequency: ω₀ = ω/2
Level spacing: ΔE = ħω

These are purely real numbers. The wave functions involve complex phases
(e^{-iE_nt/ħ}), but those phases are just TensionPair rotations.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.pair import EMLPair
from eml_math.qm.states import QuantumState, Qubit


def evolve_eigenstate(
    state: QuantumState,
    energy: float,
    t: float,
    hbar: float = 1.0,
) -> QuantumState:
    """
    Exact time evolution of an energy eigenstate.

    |E(t)⟩ = e^{−iEt/ħ}|E(0)⟩

    Standard QM requires evaluating a complex exponential via Euler's formula:
        e^{−iφ} = cos(φ) − i·sin(φ)
    followed by complex multiplication.

    MPM replaces both steps with one call:
        state.rotate(−E·t/ħ)

    The result is identical — the difference is conceptual clarity.
    """
    return state.evolve(energy, t, hbar)


def evolve_superposition(
    states: list[QuantumState],
    energies: list[float],
    coefficients: list[TensionPair],
    t: float,
    hbar: float = 1.0,
) -> list[tuple[EMLPair, QuantumState]]:
    """
    Time-evolve a superposition |ψ⟩ = Σ c_n|E_n⟩.

    Each component acquires its own phase: c_n → c_n·e^{−iE_nt/ħ}.
    Returns list of (evolved_coefficient, eigenstate) pairs.

    All phase factors c_n·e^{−iEt/ħ} computed via coefficient.rotate_phase(−Et/ħ).
    """
    result = []
    for state, E, c in zip(states, energies, coefficients):
        evolved_c = c.rotate_phase(-E * t / hbar)
        result.append((evolved_c, state))
    return result


def harmonic_oscillator_energy(n: int, omega: float = 1.0, hbar: float = 1.0) -> float:
    """
    Harmonic oscillator energy level: E_n = ħω(n + 1/2).

    This is a purely real number — the wave function phases are complex,
    but the energy eigenvalues are always real (Hermitian operator theorem).
    """
    return hbar * omega * (n + 0.5)


def harmonic_oscillator_frequency(n: int, omega: float = 1.0) -> float:
    """Oscillation frequency of eigenstate n: f_n = ω(n + 1/2) / (2π)."""
    return omega * (n + 0.5) / (2.0 * math.pi)


def two_level_rabi(
    qubit: Qubit,
    omega_rabi: float,
    t: float,
    detuning: float = 0.0,
) -> Qubit:
    """
    Rabi oscillation between |0⟩ and |1⟩.

    Driven two-level system with Rabi frequency Ω and detuning δ:
        Ω_eff = √(Ω² + δ²)
        P(|1⟩, t) = (Ω/Ω_eff)²·sin²(Ω_eff·t/2)

    Implemented via R_x rotation (off-resonance via R_y tilt for detuning).
    At resonance (δ=0): pure R_x rotation at rate Ω.

    Returns the evolved qubit state. No complex arithmetic.
    """
    omega_eff = math.sqrt(omega_rabi ** 2 + detuning ** 2)
    if omega_eff < 1e-300:
        return qubit
    # Rotation axis tilted by detuning
    theta = omega_eff * t
    if abs(detuning) < 1e-12:
        return qubit.apply_rx(theta)
    # General rotation: R_n(θ) where n = (Ω/Ω_eff, 0, δ/Ω_eff)
    # = cos(θ/2)·I − i·sin(θ/2)·(n_x·σ_x + n_z·σ_z)
    nx = omega_rabi / omega_eff
    nz = detuning / omega_eff
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    ar, ai = qubit.alpha.real_tension, qubit.alpha.imag_tension
    br, bi = qubit.beta.real_tension, qubit.beta.imag_tension
    # R = [[c-i·s·nz, -i·s·nx],[-i·s·nx, c+i·s·nz]]
    # new_alpha = (c-i·s·nz)·alpha + (-i·s·nx)·beta
    # new_beta  = (-i·s·nx)·alpha  + (c+i·s·nz)·beta
    new_ar = c * ar + s * nz * ai + s * nx * bi
    new_ai = c * ai - s * nz * ar - s * nx * br
    new_br = s * nx * ai + c * br - s * nz * bi
    new_bi = -s * nx * ar + c * bi + s * nz * br
    return Qubit(
        EMLPair.from_values(new_ar, new_ai),
        EMLPair.from_values(new_br, new_bi),
    )


def schrodinger_step_diagonal(
    psi: list[TensionPair],
    H_diag: list[float],
    dt: float,
    hbar: float = 1.0,
) -> list[TensionPair]:
    """
    One step of Schrödinger evolution for a diagonal Hamiltonian.

    ψ_n(t+dt) = e^{−iH_n·dt/ħ}·ψ_n(t) = ψ_n.rotate_phase(−H_n·dt/ħ)

    For energy eigenstates this is exact. For the potential half of the
    split-operator method this is the potential phase step.

    The complex exponential e^{−iHt/ħ} is a rotation — not a Taylor series,
    not Euler's formula, just rotate_phase(). QM is simpler than it looks.
    """
    angle_per_site = [-H * dt / hbar for H in H_diag]
    return [psi[n].rotate_phase(angle_per_site[n]) for n in range(len(psi))]


def norm_squared(psi: list[TensionPair], spacing: float = 1.0) -> float:
    """∑|ψ_n|²·spacing — total probability (should be ≈ 1 for normalised ψ)."""
    return sum(p.modulus ** 2 for p in psi) * spacing


def expect_position(psi: list[TensionPair], spacing: float = 1.0) -> float:
    """⟨x⟩ = Σ n·|ψ_n|²·spacing — expectation value of position."""
    total = 0.0
    for i, p in enumerate(psi):
        total += i * spacing * p.modulus ** 2
    return total * spacing
