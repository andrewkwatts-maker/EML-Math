"""
Tests for quantum state representations (qm/states.py).

Central demonstration: all complex QM operations map to TensionPair rotations.
Python's complex type is never used. The imaginary unit i appears only as
a rotation angle ±π/2 inside rotate_phase().
"""
import math
import pytest
from eml_math.qm.states import QuantumState, Qubit, bell_phi_plus, bell_phi_minus, bell_psi_plus
from eml_math.pair import EMLPair


# ── QuantumState ──────────────────────────────────────────────────────────────

class TestQuantumState:

    def test_from_values(self):
        qs = QuantumState.from_values(1.0, 0.0)
        assert qs.probability == pytest.approx(1.0, rel=1e-12)

    def test_from_polar(self):
        qs = QuantumState.from_polar(1.0, math.pi / 4)
        assert qs.probability == pytest.approx(1.0, rel=1e-10)

    def test_probability_born_rule(self):
        """P = |ψ|² = Re² + Im²."""
        qs = QuantumState.from_values(3.0, 4.0)
        assert qs.probability == pytest.approx(25.0, rel=1e-12)

    def test_phase(self):
        qs = QuantumState.from_polar(1.0, math.pi / 3)
        assert qs.phase == pytest.approx(math.pi / 3, rel=1e-10)

    def test_normalize(self):
        qs = QuantumState.from_values(3.0, 4.0).normalize()
        assert qs.probability == pytest.approx(1.0, rel=1e-10)

    def test_rotate_preserves_probability(self):
        """e^{iθ}·ψ changes phase but not |ψ|²."""
        qs = QuantumState.from_values(0.6, 0.8)
        prob_before = qs.probability
        for angle in [0.3, math.pi / 4, math.pi, 2 * math.pi]:
            qs_rot = qs.rotate(angle)
            assert qs_rot.probability == pytest.approx(prob_before, rel=1e-10)

    def test_evolve_preserves_probability(self):
        """Schrödinger eigenstate evolution: probability constant."""
        qs = QuantumState.from_polar(1.0, 0.0)
        for t in [0.1, 0.5, 1.0, 2.0]:
            qs_t = qs.evolve(energy=2.0, t=t)
            assert qs_t.probability == pytest.approx(1.0, rel=1e-10)

    def test_inner_product_self(self):
        """⟨ψ|ψ⟩ = |ψ|² (real, positive)."""
        qs = QuantumState.from_values(0.6, 0.8)
        ip = qs.inner_product(qs)
        # Inner product of normalised state with itself = 1 + 0i
        assert ip.real_tension == pytest.approx(1.0, rel=1e-10)
        assert abs(ip.imag_tension) < 1e-10

    def test_inner_product_orthogonal(self):
        """⟨0|1⟩ = 0."""
        qs0 = QuantumState.from_values(1.0, 0.0)
        qs1 = QuantumState.from_values(0.0, 1.0)
        ip = qs0.inner_product(qs1)
        assert abs(ip.real_tension) < 1e-12
        # imag part = 0*(0) + 1*(1) = no — let's be precise:
        # ⟨qs0|qs1⟩ = conj(1+0i)·(0+1i) = (1)(0+i) = 0+i  → real=0, imag=1
        assert ip.imag_tension == pytest.approx(1.0, rel=1e-10)


# ── Qubit: gate correctness ───────────────────────────────────────────────────

class TestQubit:

    def test_zero_state(self):
        q = Qubit.zero()
        assert q.prob_zero == pytest.approx(1.0, rel=1e-12)
        assert q.prob_one == pytest.approx(0.0, abs=1e-12)

    def test_one_state(self):
        q = Qubit.one()
        assert q.prob_zero == pytest.approx(0.0, abs=1e-12)
        assert q.prob_one == pytest.approx(1.0, rel=1e-12)

    def test_plus_state_normalised(self):
        q = Qubit.plus()
        assert q.prob_zero + q.prob_one == pytest.approx(1.0, rel=1e-12)
        assert q.prob_zero == pytest.approx(0.5, rel=1e-12)

    def test_pauli_x_swaps(self):
        q = Qubit.zero()
        q1 = q.apply_pauli_x()
        assert q1.prob_zero == pytest.approx(0.0, abs=1e-12)
        assert q1.prob_one == pytest.approx(1.0, rel=1e-12)

    def test_pauli_x_twice_is_identity(self):
        q = Qubit.plus()
        q2 = q.apply_pauli_x().apply_pauli_x()
        assert q2.prob_zero == pytest.approx(q.prob_zero, rel=1e-12)
        assert q2.prob_one == pytest.approx(q.prob_one, rel=1e-12)

    def test_pauli_z_on_zero_unchanged(self):
        q = Qubit.zero()
        qz = q.apply_pauli_z()
        assert qz.prob_zero == pytest.approx(1.0, rel=1e-12)
        assert qz.prob_one == pytest.approx(0.0, abs=1e-12)

    def test_pauli_z_negates_beta(self):
        q = Qubit.plus()
        qz = q.apply_pauli_z()
        # |+⟩ = (|0⟩+|1⟩)/√2; after σ_z: (|0⟩-|1⟩)/√2 = |−⟩
        assert qz.prob_zero == pytest.approx(0.5, rel=1e-12)
        assert qz.prob_one == pytest.approx(0.5, rel=1e-12)
        # β real tension negated
        assert qz.beta.real_tension == pytest.approx(-q.beta.real_tension, rel=1e-12)

    def test_pauli_y_needs_no_complex_type(self):
        """
        The central demonstration: σ_y requires multiplying by ±i.

        Standard QM:  new_α = -i·β  requires Python complex or cmath.
        MPM:          new_α = β.rotate_phase(-π/2)  — purely real rotation.

        Verify result matches the rotation formula algebraically.
        σ_y = [[0,-i],[i,0]]:
            new_α = -i·β → rotate_phase(-π/2) → (b_r·0 - b_i·(-1), b_r·(-1) + b_i·0) ...
            In TensionPair: rotate_phase(-π/2) maps (r, im) → (im, -r)
        """
        q = Qubit.from_bloch(theta=math.pi / 3, phi=math.pi / 4)
        br, bi = q.beta.real_tension, q.beta.imag_tension
        ar, ai = q.alpha.real_tension, q.alpha.imag_tension

        qy = q.apply_pauli_y()

        # new_alpha = -i·beta: (br, bi) → rotate(-π/2) → (bi, -br)
        assert qy.alpha.real_tension == pytest.approx(bi, rel=1e-10)
        assert qy.alpha.imag_tension == pytest.approx(-br, rel=1e-10)
        # new_beta = +i·alpha: (ar, ai) → rotate(+π/2) → (-ai, ar)
        assert qy.beta.real_tension == pytest.approx(-ai, rel=1e-10)
        assert qy.beta.imag_tension == pytest.approx(ar, rel=1e-10)

    def test_pauli_y_preserves_norm(self):
        q = Qubit.from_bloch(theta=math.pi / 3, phi=math.pi / 4)
        qy = q.apply_pauli_y()
        assert qy.prob_zero + qy.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_pauli_y_twice_is_minus_identity(self):
        """σ_y² = I (probabilities unchanged, amplitudes may flip sign)."""
        q = Qubit.plus()
        q2 = q.apply_pauli_y().apply_pauli_y()
        assert q2.prob_zero == pytest.approx(q.prob_zero, rel=1e-10)
        assert q2.prob_one == pytest.approx(q.prob_one, rel=1e-10)

    def test_hadamard_zero_to_plus(self):
        q = Qubit.zero().apply_hadamard()
        assert q.prob_zero == pytest.approx(0.5, rel=1e-10)
        assert q.prob_one == pytest.approx(0.5, rel=1e-10)

    def test_hadamard_one_to_minus(self):
        q = Qubit.one().apply_hadamard()
        assert q.prob_zero == pytest.approx(0.5, rel=1e-10)
        assert q.prob_one == pytest.approx(0.5, rel=1e-10)
        assert q.beta.real_tension == pytest.approx(-q.alpha.real_tension, rel=1e-10)

    def test_hadamard_twice_is_identity(self):
        q = Qubit.from_bloch(theta=math.pi / 3, phi=math.pi / 5)
        q2 = q.apply_hadamard().apply_hadamard()
        assert q2.alpha.real_tension == pytest.approx(q.alpha.real_tension, rel=1e-10)
        assert q2.alpha.imag_tension == pytest.approx(q.alpha.imag_tension, rel=1e-10)
        assert q2.beta.real_tension == pytest.approx(q.beta.real_tension, rel=1e-10)
        assert q2.beta.imag_tension == pytest.approx(q.beta.imag_tension, rel=1e-10)

    def test_phase_gate_zero_unchanged(self):
        q = Qubit.zero().apply_phase_gate(math.pi / 3)
        assert q.prob_zero == pytest.approx(1.0, rel=1e-12)
        assert q.prob_one == pytest.approx(0.0, abs=1e-12)

    def test_phase_gate_rotates_beta(self):
        q = Qubit.plus()
        phi = math.pi / 3
        qp = q.apply_phase_gate(phi)
        # β → e^{iφ}β: |β|² unchanged
        assert qp.prob_one == pytest.approx(q.prob_one, rel=1e-10)

    def test_rx_pi_is_pauli_x(self):
        """R_x(π) = -i·σ_x: swaps |0⟩↔|1⟩ up to global phase."""
        q = Qubit.zero()
        qrx = q.apply_rx(math.pi)
        assert qrx.prob_zero == pytest.approx(0.0, abs=1e-10)
        assert qrx.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_rx_preserves_norm(self):
        q = Qubit.from_bloch(theta=1.0, phi=0.5)
        qrx = q.apply_rx(0.7)
        assert qrx.prob_zero + qrx.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_ry_pi_is_pauli_y_rotation(self):
        """R_y(π)|0⟩ = |1⟩."""
        q = Qubit.zero()
        qry = q.apply_ry(math.pi)
        assert qry.prob_zero == pytest.approx(0.0, abs=1e-10)
        assert qry.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_ry_preserves_norm(self):
        q = Qubit.from_bloch(theta=0.8, phi=1.2)
        qry = q.apply_ry(0.9)
        assert qry.prob_zero + qry.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_from_bloch_normalised(self):
        for theta in [0.0, math.pi / 4, math.pi / 2, math.pi]:
            for phi in [0.0, math.pi / 3, math.pi]:
                q = Qubit.from_bloch(theta, phi)
                assert q.prob_zero + q.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_from_bloch_north_pole_is_zero(self):
        q = Qubit.from_bloch(theta=0.0, phi=0.0)
        assert q.prob_zero == pytest.approx(1.0, rel=1e-12)

    def test_from_bloch_south_pole_is_one(self):
        q = Qubit.from_bloch(theta=math.pi, phi=0.0)
        assert q.prob_one == pytest.approx(1.0, rel=1e-10)

    def test_is_normalized(self):
        assert Qubit.zero().is_normalized()
        assert Qubit.plus().is_normalized()
        assert Qubit.from_bloch(0.7, 1.3).is_normalized()

    def test_normalize_already_normalised(self):
        q = Qubit.plus()
        qn = q.normalize()
        assert qn.prob_zero + qn.prob_one == pytest.approx(1.0, rel=1e-12)

    def test_amplitudes_are_tensionpairs(self):
        """All amplitudes are TensionPair — no Python complex anywhere."""
        q = Qubit.from_bloch(0.5, 0.8)
        assert isinstance(q.alpha, EMLPair)
        assert isinstance(q.beta, EMLPair)
        # Real and imaginary parts are plain floats
        assert isinstance(q.alpha.real_tension, float)
        assert isinstance(q.alpha.imag_tension, float)


# ── Bell states ───────────────────────────────────────────────────────────────

class TestBellStates:

    def test_phi_plus_normalised(self):
        a00, a11 = bell_phi_plus()
        total = a00.modulus ** 2 + a11.modulus ** 2
        assert total == pytest.approx(1.0, rel=1e-12)

    def test_phi_minus_normalised(self):
        a00, a11 = bell_phi_minus()
        total = a00.modulus ** 2 + a11.modulus ** 2
        assert total == pytest.approx(1.0, rel=1e-12)

    def test_psi_plus_normalised(self):
        a01, a10 = bell_psi_plus()
        total = a01.modulus ** 2 + a10.modulus ** 2
        assert total == pytest.approx(1.0, rel=1e-12)

    def test_phi_plus_minus_differ(self):
        a00_p, a11_p = bell_phi_plus()
        a00_m, a11_m = bell_phi_minus()
        assert a11_p.real_tension != pytest.approx(a11_m.real_tension, abs=1e-5)

    def test_bell_states_are_tensionpairs(self):
        for factory in [bell_phi_plus, bell_phi_minus, bell_psi_plus]:
            pair = factory()
            assert all(isinstance(p, EMLPair) for p in pair)
