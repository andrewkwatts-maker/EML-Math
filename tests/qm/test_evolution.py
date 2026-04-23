"""
Tests for quantum evolution functions (qm/evolution.py).

Key demonstrations:
1. Energy eigenstate evolution = single rotate_phase() — no ODE, no complex exponential.
2. Harmonic oscillator energy levels are purely real (ħω(n+1/2)).
3. Rabi oscillations computed without complex arithmetic.
4. Diagonal Hamiltonian evolution via site-by-site rotate_phase().
"""
import math
import pytest
from eml_math.qm.states import QuantumState, Qubit
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
from eml_math.pair import EMLPair


# ── Eigenstate evolution ──────────────────────────────────────────────────────

class TestEvolvEigenstate:

    def test_probability_conserved(self):
        """
        |E(t)⟩ = e^{-iEt/ħ}|E(0)⟩ — probability unchanged.

        MPM: state.rotate(-E*t/ħ). The complex exponential is a pure rotation.
        |ψ|² is invariant under rotation.
        """
        state = QuantumState.from_polar(1.0, 0.3)
        prob0 = state.probability
        for t in [0.0, 0.1, 0.5, 1.0, 2.0 * math.pi]:
            evolved = evolve_eigenstate(state, energy=2.5, t=t)
            assert evolved.probability == pytest.approx(prob0, rel=1e-10)

    def test_zero_time_is_identity(self):
        state = QuantumState.from_values(0.6, 0.8)
        evolved = evolve_eigenstate(state, energy=3.0, t=0.0)
        assert evolved.amplitude.real_tension == pytest.approx(0.6, rel=1e-12)
        assert evolved.amplitude.imag_tension == pytest.approx(0.8, rel=1e-12)

    def test_full_period_returns_to_start(self):
        """After t = 2πħ/E the state completes a full phase cycle."""
        state = QuantumState.from_values(1.0, 0.0)
        E, hbar = 2.0, 1.0
        T = 2 * math.pi * hbar / E
        evolved = evolve_eigenstate(state, energy=E, t=T, hbar=hbar)
        assert evolved.amplitude.real_tension == pytest.approx(1.0, rel=1e-10)
        assert abs(evolved.amplitude.imag_tension) < 1e-10

    def test_phase_increases_linearly(self):
        """Phase angle = -E*t/ħ grows linearly with t (verified via amplitude components)."""
        state = QuantumState.from_polar(1.0, 0.0)
        E, hbar = 1.5, 1.0
        for t in [0.1, 0.2, 0.4]:
            evolved = evolve_eigenstate(state, energy=E, t=t, hbar=hbar)
            phi = -E * t / hbar
            # Amplitude: (cos(phi), sin(phi)) since initial state is (1, 0)
            assert evolved.amplitude.real_tension == pytest.approx(math.cos(phi), abs=1e-10)
            assert evolved.amplitude.imag_tension == pytest.approx(math.sin(phi), abs=1e-10)

    def test_no_ode_solver_needed(self):
        """evolve_eigenstate is a direct call — no iteration, no Euler method."""
        # Just check it works instantly for large t without stepping
        state = QuantumState.from_polar(1.0, 0.0)
        evolved = evolve_eigenstate(state, energy=1.0, t=1e6)
        assert math.isfinite(evolved.probability)


# ── Superposition evolution ───────────────────────────────────────────────────

class TestEvolveSuperposition:

    def test_each_component_evolves_independently(self):
        s = 1.0 / math.sqrt(2.0)
        states = [QuantumState.from_polar(1.0, 0.0), QuantumState.from_polar(1.0, 0.0)]
        energies = [1.0, 2.0]
        coeffs = [EMLPair.from_values(s, 0.0), EMLPair.from_values(s, 0.0)]
        result = evolve_superposition(states, energies, coeffs, t=0.5)
        assert len(result) == 2
        for c, st in result:
            assert isinstance(c, EMLPair)
            assert isinstance(st, QuantumState)

    def test_coefficients_rotate_with_energy(self):
        state = QuantumState.from_polar(1.0, 0.0)
        E = 2.0
        c0 = EMLPair.from_values(1.0, 0.0)
        result = evolve_superposition([state], [E], [c0], t=1.0)
        c_evolved, _ = result[0]
        # c → c·e^{-iEt/ħ} = rotate_phase(-E)
        expected = c0.rotate_phase(-E * 1.0)
        assert c_evolved.real_tension == pytest.approx(expected.real_tension, rel=1e-10)
        assert c_evolved.imag_tension == pytest.approx(expected.imag_tension, rel=1e-10)


# ── Harmonic oscillator ───────────────────────────────────────────────────────

class TestHarmonicOscillator:

    def test_ground_state_energy(self):
        """E_0 = ħω/2 — zero-point energy."""
        E0 = harmonic_oscillator_energy(n=0, omega=1.0, hbar=1.0)
        assert E0 == pytest.approx(0.5, rel=1e-12)

    def test_first_excited_energy(self):
        """E_1 = 3ħω/2."""
        E1 = harmonic_oscillator_energy(n=1, omega=1.0, hbar=1.0)
        assert E1 == pytest.approx(1.5, rel=1e-12)

    def test_energy_levels_equally_spaced(self):
        """ΔE = ħω between adjacent levels."""
        omega, hbar = 2.0, 1.0
        for n in range(5):
            En = harmonic_oscillator_energy(n, omega, hbar)
            En1 = harmonic_oscillator_energy(n + 1, omega, hbar)
            assert En1 - En == pytest.approx(hbar * omega, rel=1e-12)

    def test_energy_is_real(self):
        """Energy eigenvalues are purely real (Hermitian operator theorem)."""
        for n in range(10):
            E = harmonic_oscillator_energy(n, omega=1.5, hbar=1.0)
            assert isinstance(E, float)
            assert math.isfinite(E)
            assert E > 0

    def test_ground_state_frequency(self):
        """f_0 = ω/(4π)."""
        omega = 2.0
        f0 = harmonic_oscillator_frequency(n=0, omega=omega)
        assert f0 == pytest.approx(omega / (4 * math.pi), rel=1e-12)

    def test_frequency_scales_with_n(self):
        omega = 1.0
        f0 = harmonic_oscillator_frequency(n=0, omega=omega)
        f2 = harmonic_oscillator_frequency(n=2, omega=omega)
        assert f2 / f0 == pytest.approx(5.0, rel=1e-12)  # (2+0.5)/(0+0.5) = 5


# ── Rabi oscillations ─────────────────────────────────────────────────────────

class TestTwoLevelRabi:

    def test_resonance_prob_zero_at_t0(self):
        """Starting in |0⟩, P(|1⟩, 0) = 0."""
        q = Qubit.zero()
        qr = two_level_rabi(q, omega_rabi=1.0, t=0.0)
        assert qr.prob_one == pytest.approx(0.0, abs=1e-10)

    def test_resonance_pi_pulse_flips(self):
        """At t = π/Ω (resonance): P(|1⟩) = 1."""
        q = Qubit.zero()
        Omega = 1.0
        qr = two_level_rabi(q, omega_rabi=Omega, t=math.pi / Omega)
        assert qr.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_resonance_two_pi_returns(self):
        """After 2π/Ω: state returns to |0⟩."""
        q = Qubit.zero()
        Omega = 1.0
        qr = two_level_rabi(q, omega_rabi=Omega, t=2 * math.pi / Omega)
        assert qr.prob_zero == pytest.approx(1.0, rel=1e-10)

    def test_rabi_probability_formula(self):
        """P(|1⟩) = sin²(Ω·t/2) at resonance."""
        q = Qubit.zero()
        Omega = 2.0
        for t in [0.1, 0.5, 1.0, 1.5]:
            qr = two_level_rabi(q, omega_rabi=Omega, t=t)
            expected = math.sin(Omega * t / 2) ** 2
            assert qr.prob_one == pytest.approx(expected, rel=1e-8)

    def test_rabi_preserves_norm(self):
        q = Qubit.from_bloch(math.pi / 4, math.pi / 3)
        for t in [0.1, 0.5, 1.2]:
            qr = two_level_rabi(q, omega_rabi=1.5, t=t, detuning=0.5)
            assert qr.prob_zero + qr.prob_one == pytest.approx(1.0, rel=1e-8)

    def test_zero_omega_no_evolution(self):
        q = Qubit.zero()
        qr = two_level_rabi(q, omega_rabi=0.0, t=1.0, detuning=0.0)
        assert qr.prob_zero == pytest.approx(1.0, rel=1e-12)

    def test_detuned_effective_frequency(self):
        """With detuning, effective Ω_eff = √(Ω² + δ²)."""
        q = Qubit.zero()
        Omega, delta = 3.0, 4.0
        Omega_eff = math.sqrt(Omega ** 2 + delta ** 2)  # = 5.0
        # P(|1⟩) = (Ω/Ω_eff)² · sin²(Ω_eff·t/2)
        t = 0.4
        qr = two_level_rabi(q, omega_rabi=Omega, t=t, detuning=delta)
        expected = (Omega / Omega_eff) ** 2 * math.sin(Omega_eff * t / 2) ** 2
        assert qr.prob_one == pytest.approx(expected, rel=1e-6)

    def test_output_is_qubit(self):
        q = Qubit.zero()
        qr = two_level_rabi(q, omega_rabi=1.0, t=0.5)
        assert isinstance(qr, Qubit)


# ── Diagonal Hamiltonian evolution ────────────────────────────────────────────

class TestSchrodingerStepDiagonal:

    def test_modulus_preserved(self):
        """ψ_n → e^{-iH_n dt/ħ}·ψ_n: modulus unchanged."""
        N = 8
        psi = [EMLPair.from_values(math.cos(i), math.sin(i)) for i in range(N)]
        H_diag = [float(i) for i in range(N)]
        mods_before = [p.modulus for p in psi]
        psi_after = schrodinger_step_diagonal(psi, H_diag, dt=0.1)
        mods_after = [p.modulus for p in psi_after]
        for m0, m1 in zip(mods_before, mods_after):
            assert m1 == pytest.approx(m0, rel=1e-10)

    def test_zero_hamiltonian_is_identity(self):
        N = 4
        psi = [EMLPair.from_values(float(i + 1), 0.0) for i in range(N)]
        H_diag = [0.0] * N
        psi_after = schrodinger_step_diagonal(psi, H_diag, dt=1.0)
        for p0, p1 in zip(psi, psi_after):
            assert p1.real_tension == pytest.approx(p0.real_tension, rel=1e-12)
            assert p1.imag_tension == pytest.approx(p0.imag_tension, rel=1e-12)

    def test_phase_rotation_angle(self):
        """Site with H=E, dt=dt: phase rotates by -E*dt/ħ."""
        psi = [EMLPair.from_values(1.0, 0.0)]
        E, dt, hbar = 2.0, 0.5, 1.0
        psi_after = schrodinger_step_diagonal(psi, [E], dt=dt, hbar=hbar)
        expected = EMLPair.from_values(1.0, 0.0).rotate_phase(-E * dt / hbar)
        assert psi_after[0].real_tension == pytest.approx(expected.real_tension, rel=1e-10)
        assert psi_after[0].imag_tension == pytest.approx(expected.imag_tension, rel=1e-10)

    def test_output_length(self):
        N = 5
        psi = [EMLPair.from_values(1.0, 0.0)] * N
        H = [1.0] * N
        result = schrodinger_step_diagonal(psi, H, dt=0.1)
        assert len(result) == N


# ── Observables ───────────────────────────────────────────────────────────────

class TestObservables:

    def test_norm_squared_normalised(self):
        """∑|ψ_n|²·spacing ≈ 1 for normalised state."""
        N = 8
        s = 1.0 / math.sqrt(N)
        psi = [EMLPair.from_values(s, 0.0)] * N
        assert norm_squared(psi, spacing=1.0) == pytest.approx(1.0, rel=1e-12)

    def test_expect_position_center(self):
        """Uniform distribution → ⟨x⟩ at middle of lattice."""
        N = 10
        spacing = 1.0
        s = 1.0 / math.sqrt(N)
        psi = [EMLPair.from_values(s, 0.0)] * N
        x_exp = expect_position(psi, spacing=spacing)
        # ∑ n·(1/N)·spacing² = spacing²·N(N-1)/2 / N = spacing²·(N-1)/2
        expected = spacing ** 2 * (N - 1) / 2
        assert x_exp == pytest.approx(expected, rel=1e-10)

    def test_expect_position_localized(self):
        """State localised at site k → ⟨x⟩ ≈ k."""
        N = 8
        k = 5
        psi = [EMLPair.from_values(0.0, 0.0)] * N
        psi[k] = EMLPair.from_values(1.0, 0.0)
        x_exp = expect_position(psi, spacing=1.0)
        assert x_exp == pytest.approx(float(k), rel=1e-10)
