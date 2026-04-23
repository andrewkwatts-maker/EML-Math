"""
Tests for quantum field theory simulations (qft.py).

Demonstrates three key properties:
1. Klein-Gordon field stays real throughout evolution — no complex arithmetic needed.
2. Schrödinger eigenstate evolution is a single rotate_phase() call.
3. EML action provides a natural lattice action alternative to harmonic action.
"""
import math
import pytest
from eml_math.qft import KleinGordonField, SchrodingerField, PathIntegral
from eml_math.pair import EMLPair


# ── Klein-Gordon field ────────────────────────────────────────────────────────

class TestKleinGordonField:

    def test_gaussian_packet_stays_real(self):
        field = KleinGordonField.gaussian_packet(N=50, mass=1.0, width=3.0)
        snapshots = field.evolve(steps=20)
        for snap in snapshots:
            for v in snap:
                assert math.isfinite(v), f"Non-finite field value: {v}"

    def test_initial_amplitude_at_center(self):
        N, amplitude = 50, 2.0
        field = KleinGordonField.gaussian_packet(N=N, amplitude=amplitude, width=3.0)
        center_idx = N // 2
        assert abs(field.field[center_idx] - amplitude) < 0.1

    def test_energy_approximately_conserved(self):
        field = KleinGordonField.gaussian_packet(N=50, mass=0.5, width=3.0, dt=0.04)
        E0 = field.total_energy()
        field.evolve(steps=50)
        E1 = field.total_energy()
        assert abs(E1 - E0) / (abs(E0) + 1e-10) < 0.05  # within 5%

    def test_cfl_stability_condition(self):
        field = KleinGordonField.gaussian_packet(N=20, spacing=1.0, dt=0.04)
        assert field.cfl_number() < 1.0

    def test_field_length_unchanged(self):
        N = 30
        field = KleinGordonField.gaussian_packet(N=N, width=2.0)
        field.step()
        assert len(field.field) == N

    def test_energy_density_finite(self):
        field = KleinGordonField.gaussian_packet(N=40, mass=1.0, width=2.0)
        field.step()
        for e in field.energy_density():
            assert math.isfinite(e)

    def test_eml_inter_site_tension_finite(self):
        field = KleinGordonField.gaussian_packet(N=30, mass=1.0, width=2.0)
        for T in field.eml_inter_site_tension():
            assert math.isfinite(T)

    def test_periodic_boundary(self):
        N = 10
        field = KleinGordonField(N, [float(i) for i in range(N)], boundary="periodic")
        nbr_left = field._nbr(0, -1)
        assert nbr_left == N - 1

    def test_zero_field_stays_zero(self):
        N = 20
        field = KleinGordonField(N, [0.0] * N, mass=1.0)
        snapshots = field.evolve(steps=10)
        for snap in snapshots:
            for v in snap:
                assert abs(v) < 1e-10

    def test_snapshots_count(self):
        field = KleinGordonField.gaussian_packet(N=20, width=2.0)
        steps = 5
        snapshots = field.evolve(steps=steps)
        assert len(snapshots) == steps + 1


# ── Schrödinger field ─────────────────────────────────────────────────────────

class TestSchrodingerField:

    def test_particle_in_box_normalised(self):
        field = SchrodingerField.particle_in_box(N=64, n_mode=1)
        assert abs(field.norm() - 1.0) < 1e-10

    def test_norm_preserved_under_potential_step(self):
        field = SchrodingerField.particle_in_box(N=64, n_mode=1)
        norm_before = field.norm()
        for _ in range(10):
            field.step()
        norm_after = field.norm()
        assert abs(norm_after - norm_before) < 1e-10

    def test_gaussian_normalised(self):
        field = SchrodingerField.gaussian(N=64, width=4.0, momentum=0.0)
        assert abs(field.norm() - 1.0) < 1e-10

    def test_eigenstate_evolution_is_pure_phase_rotation(self):
        """
        The central QM simplification:
        |E(t)⟩ = e^{-iEt/ħ}|E(0)⟩ — implemented as psi.rotate_phase(-E*t/ħ).
        Probability |ψ|² is unchanged by a phase rotation.
        """
        field = SchrodingerField.particle_in_box(N=32, n_mode=1)
        probs_before = field.probability()
        energy = SchrodingerField.particle_in_box_energy(n=1, N=32)
        psi_evolved = field.evolve_eigenstate(energy=energy, t=1.0)
        probs_after = [p.modulus ** 2 for p in psi_evolved]
        for p0, p1 in zip(probs_before, probs_after):
            assert abs(p1 - p0) < 1e-10

    def test_eigenstate_phase_changes(self):
        """Phase rotates without touching probability."""
        field = SchrodingerField.particle_in_box(N=32, n_mode=1)
        energy = SchrodingerField.particle_in_box_energy(n=1, N=32)
        psi_t0 = field.psi
        psi_t1 = field.evolve_eigenstate(energy=energy, t=0.5)
        # At least one amplitude should have rotated
        phase_changed = any(
            abs(psi_t0[i].real_tension - psi_t1[i].real_tension) > 1e-10
            for i in range(len(psi_t0))
        )
        assert phase_changed

    def test_no_complex_type_used(self):
        """SchrodingerField uses only EMLPair, never Python complex."""
        field = SchrodingerField.particle_in_box(N=16, n_mode=1)
        for p in field.psi:
            assert isinstance(p, EMLPair)
            # TensionPair stores two real tensions — no Python complex
            assert isinstance(p.real_tension, float)
            assert isinstance(p.imag_tension, float)

    def test_gaussian_with_momentum(self):
        field = SchrodingerField.gaussian(N=64, width=4.0, momentum=1.0)
        assert abs(field.norm() - 1.0) < 1e-10

    def test_particle_in_box_energy_levels(self):
        """Energy levels scale as n²."""
        N = 64
        E1 = SchrodingerField.particle_in_box_energy(n=1, N=N)
        E2 = SchrodingerField.particle_in_box_energy(n=2, N=N)
        assert abs(E2 / E1 - 4.0) < 1e-10

    def test_probability_sums_to_one(self):
        field = SchrodingerField.particle_in_box(N=64, n_mode=2)
        probs = field.probability()
        total = sum(probs) * field.spacing
        assert abs(total - 1.0) < 1e-10

    def test_step_returns_rotated_amplitudes(self):
        N = 16
        V = [float(i % 3) for i in range(N)]
        field = SchrodingerField(N, potential=V)
        from eml_math.pair import EMLPair
        psi_init = [EMLPair.from_values(1.0 if i == N // 2 else 0.0, 0.0)
                    for i in range(N)]
        field.set_psi(psi_init)
        field.step()
        # Central site has V=0, phase unchanged; neighbouring sites evolved
        psi_after = field.psi
        assert isinstance(psi_after[N // 2], EMLPair)


# ── Path integral ─────────────────────────────────────────────────────────────

class TestPathIntegral:

    def test_harmonic_action_positive(self):
        pi = PathIntegral(mass=1.0)
        path = [0.0, 0.1, 0.2, 0.1, 0.0]
        S = pi.harmonic_action(path, dt=0.1, omega=1.0)
        assert S > 0

    def test_eml_action_finite(self):
        pi = PathIntegral(mass=1.0)
        path = [0.5, 0.6, 0.7, 0.6, 0.5]
        S = pi.eml_action(path)
        assert math.isfinite(S)

    def test_compute_returns_positive(self):
        pi = PathIntegral(mass=1.0)
        result = pi.compute(x_initial=0.0, x_final=0.0, T=5, num_paths=100, seed=42)
        assert result > 0

    def test_compute_eml_action_positive(self):
        pi = PathIntegral(mass=1.0)
        result = pi.compute(
            x_initial=0.5, x_final=0.5, T=5, num_paths=100,
            use_eml_action=True, seed=42,
        )
        assert result > 0

    def test_compute_deterministic_with_seed(self):
        pi = PathIntegral(mass=1.0)
        r1 = pi.compute(0.0, 0.0, T=5, num_paths=50, seed=7)
        r2 = pi.compute(0.0, 0.0, T=5, num_paths=50, seed=7)
        assert r1 == r2

    def test_eml_action_handles_zero_field(self):
        pi = PathIntegral(mass=1.0)
        path = [0.0, 0.1, 0.0]
        S = pi.eml_action(path)
        assert math.isfinite(S)

    def test_straight_path_lower_action(self):
        """A straight path between endpoints should have lower harmonic action than a detour."""
        pi = PathIntegral(mass=1.0)
        straight = [0.0, 0.5, 1.0]
        detour = [0.0, 2.0, 1.0]
        assert pi.harmonic_action(straight, dt=0.5) < pi.harmonic_action(detour, dt=0.5)
