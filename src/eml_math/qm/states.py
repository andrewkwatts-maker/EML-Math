"""
Quantum state representations using TensionPair — no complex type required.

QuantumState
    Single amplitude ψ ∈ ℂ as EMLPair(Re[ψ], Im[ψ]).

Qubit
    Two-level system |ψ⟩ = α|0⟩ + β|1⟩.
    α and β are TensionPairs. All gates (Pauli, Hadamard, Phase) use
    rotate_phase() — the imaginary unit i never appears as a Python value.

The key insight
---------------
Standard QM uses complex numbers for amplitudes. Every complex operation:

    Traditional                     MPM (EMLPair)
    ──────────────────────────────────────────────────────────
    i · ψ                           psi.rotate_phase(π/2)
    e^{iθ} · ψ                      psi.rotate_phase(θ)
    |ψ|²  (norm)                    psi.modulus ** 2
    ψ*    (conjugate)               psi.conjugate()
    σ_y|↑⟩ = i|↓⟩                  beta.rotate_phase(π/2)

All imaginary-unit operations are rotations. TensionPair makes this
explicit and geometric. No `import cmath` anywhere in this module.
"""
from __future__ import annotations

import math
from typing import Optional

from eml_math.pair import EMLPair


class QuantumState:
    """
    A single quantum amplitude ψ encoded as a EMLPair.

    EMLPair(Re[ψ], Im[ψ]) holds the real and imaginary parts as two
    real tensions. Every standard complex operation maps to a TensionPair
    operation — no Python complex type needed.
    """

    __slots__ = ("_pair",)

    def __init__(self, pair: EMLPair) -> None:
        self._pair = pair

    @classmethod
    def from_values(cls, real: float, imag: float) -> "QuantumState":
        return cls(EMLPair.from_values(real, imag))

    @classmethod
    def from_polar(cls, r: float, theta: float) -> "QuantumState":
        """ψ = r·e^{iθ} = r·(cos θ + i·sin θ), encoded as EMLPair."""
        return cls(EMLPair.from_polar(r, theta))

    @property
    def amplitude(self) -> TensionPair:
        return self._pair

    @property
    def probability(self) -> float:
        """Born rule: P = |ψ|² = real² + imag²."""
        return self._pair.modulus ** 2

    @property
    def phase(self) -> float:
        """Phase angle θ = arctan(Im/Re)."""
        return self._pair.argument

    def normalize(self) -> "QuantumState":
        m = self._pair.modulus
        if m < 1e-300:
            return QuantumState(EMLPair.from_values(1.0, 0.0))
        return QuantumState(EMLPair.from_values(
            self._pair.real_tension / m,
            self._pair.imag_tension / m,
        ))

    def rotate(self, angle: float) -> "QuantumState":
        """
        Multiply by e^{iθ}: the quantum phase shift.

        e^{iθ}·ψ = ψ.rotate_phase(θ)

        In standard QM this uses complex multiplication via Euler's formula.
        Here it is a direct geometric rotation — no Euler formula invocation.
        """
        return QuantumState(self._pair.rotate_phase(angle))

    def evolve(self, energy: float, t: float, hbar: float = 1.0) -> "QuantumState":
        """
        Exact Schrödinger time evolution for an energy eigenstate.

        |E(t)⟩ = e^{−iEt/ħ}|E(0)⟩ = psi.rotate_phase(−E·t/ħ)

        The full Schrödinger equation collapses to a single rotation call.
        """
        return self.rotate(-energy * t / hbar)

    def inner_product(self, other: "QuantumState") -> TensionPair:
        """
        ⟨self|other⟩ = self*.other = (a−ib)(c+id) = (ac+bd) + i(ad−bc).

        Using TensionPair conjugate and multiplication — no complex type.
        """
        conj = self._pair.conjugate()
        return conj * other._pair

    def __repr__(self) -> str:
        return f"QuantumState({self._pair})"


class Qubit:
    """
    Two-level quantum system: |ψ⟩ = α|0⟩ + β|1⟩.

    α, β are TensionPairs (complex amplitudes without Python complex).
    Normalisation: |α|² + |β|² = 1.

    All single-qubit gates are implemented using EMLPair.rotate_phase().
    The imaginary unit i appears only as a rotation angle ±π/2 — never as
    a Python value. This is the central simplification MPM offers for QM.

    Gate summary (standard vs MPM)
    ───────────────────────────────────────────────────────────
    σ_x            flip α↔β                   (no phase)
    σ_z            β → −β                     (negate β)
    σ_y            new_α = −i·β = β.rotate_phase(−π/2)
                   new_β = +i·α = α.rotate_phase(+π/2)
    H              α′ = (α+β)/√2, β′ = (α−β)/√2
    R_z(φ)         β → e^{iφ}β = β.rotate_phase(φ)
    ───────────────────────────────────────────────────────────
    """

    __slots__ = ("_alpha", "_beta")

    def __init__(self, alpha: EMLPair, beta: EMLPair) -> None:
        self._alpha = alpha
        self._beta = beta

    @property
    def alpha(self) -> TensionPair:
        """Amplitude for |0⟩."""
        return self._alpha

    @property
    def beta(self) -> TensionPair:
        """Amplitude for |1⟩."""
        return self._beta

    @property
    def prob_zero(self) -> float:
        """P(|0⟩) = |α|²."""
        return self._alpha.modulus ** 2

    @property
    def prob_one(self) -> float:
        """P(|1⟩) = |β|²."""
        return self._beta.modulus ** 2

    def is_normalized(self, tol: float = 1e-9) -> bool:
        return abs(self.prob_zero + self.prob_one - 1.0) < tol

    def normalize(self) -> "Qubit":
        norm = math.sqrt(self.prob_zero + self.prob_one)
        if norm < 1e-300:
            return Qubit.zero()
        s = 1.0 / norm
        return Qubit(
            EMLPair.from_values(self._alpha.real_tension * s, self._alpha.imag_tension * s),
            EMLPair.from_values(self._beta.real_tension * s, self._beta.imag_tension * s),
        )

    # ── standard gate set ─────────────────────────────────────────────────────

    def apply_pauli_x(self) -> "Qubit":
        """σ_x: quantum NOT gate. |0⟩↔|1⟩, swaps α and β."""
        return Qubit(self._beta, self._alpha)

    def apply_pauli_z(self) -> "Qubit":
        """σ_z: |0⟩→|0⟩, |1⟩→−|1⟩. Negate the |1⟩ amplitude."""
        neg_beta = EMLPair.from_values(-self._beta.real_tension, -self._beta.imag_tension)
        return Qubit(self._alpha, neg_beta)

    def apply_pauli_y(self) -> "Qubit":
        """
        σ_y = [[0, −i], [i, 0]].

        Standard QM: requires complex multiplication by ±i.
        MPM: rotate_phase(±π/2) — the imaginary unit is a rotation angle.

            σ_y|ψ⟩: new_α = −i·β = β.rotate_phase(−π/2)
                    new_β = +i·α = α.rotate_phase(+π/2)

        No complex type used. The 'i' in σ_y is handled geometrically.
        """
        new_alpha = self._beta.rotate_phase(-math.pi / 2)   # −i·β
        new_beta = self._alpha.rotate_phase(math.pi / 2)    # +i·α
        return Qubit(new_alpha, new_beta)

    def apply_hadamard(self) -> "Qubit":
        """H = (σ_x + σ_z)/√2. Maps |0⟩→|+⟩, |1⟩→|−⟩."""
        s = 1.0 / math.sqrt(2.0)
        ar, ai = self._alpha.real_tension, self._alpha.imag_tension
        br, bi = self._beta.real_tension, self._beta.imag_tension
        return Qubit(
            EMLPair.from_values(s * (ar + br), s * (ai + bi)),
            EMLPair.from_values(s * (ar - br), s * (ai - bi)),
        )

    def apply_phase_gate(self, phi: float) -> "Qubit":
        """R_z(φ): |0⟩→|0⟩, |1⟩→e^{iφ}|1⟩ = β.rotate_phase(φ)."""
        return Qubit(self._alpha, self._beta.rotate_phase(phi))

    def apply_rx(self, theta: float) -> "Qubit":
        """R_x(θ) = exp(−iθσ_x/2) = I·cos(θ/2) − i·σ_x·sin(θ/2)."""
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        ar, ai = self._alpha.real_tension, self._alpha.imag_tension
        br, bi = self._beta.real_tension, self._beta.imag_tension
        # R_x|ψ⟩: α′ = c·α − i·s·β = c·α + s·β.rotate_phase(−π/2)
        # β′ = −i·s·α + c·β
        new_alpha = EMLPair.from_values(c * ar + s * bi, c * ai - s * br)
        new_beta = EMLPair.from_values(c * br + s * ai, c * bi - s * ar)
        return Qubit(new_alpha, new_beta)

    def apply_ry(self, theta: float) -> "Qubit":
        """R_y(θ) = exp(−iθσ_y/2) = I·cos(θ/2) − i·σ_y·sin(θ/2)."""
        c = math.cos(theta / 2.0)
        s = math.sin(theta / 2.0)
        ar, ai = self._alpha.real_tension, self._alpha.imag_tension
        br, bi = self._beta.real_tension, self._beta.imag_tension
        new_alpha = EMLPair.from_values(c * ar - s * br, c * ai - s * bi)
        new_beta = EMLPair.from_values(s * ar + c * br, s * ai + c * bi)
        return Qubit(new_alpha, new_beta)

    # ── factories ─────────────────────────────────────────────────────────────

    @classmethod
    def zero(cls) -> "Qubit":
        """|0⟩ = |↑⟩: α = 1, β = 0."""
        return cls(EMLPair.from_values(1.0, 0.0), EMLPair.from_values(0.0, 0.0))

    @classmethod
    def one(cls) -> "Qubit":
        """|1⟩ = |↓⟩: α = 0, β = 1."""
        return cls(EMLPair.from_values(0.0, 0.0), EMLPair.from_values(1.0, 0.0))

    @classmethod
    def plus(cls) -> "Qubit":
        """|+⟩ = (|0⟩ + |1⟩)/√2."""
        s = 1.0 / math.sqrt(2.0)
        return cls(EMLPair.from_values(s, 0.0), EMLPair.from_values(s, 0.0))

    @classmethod
    def minus(cls) -> "Qubit":
        """|−⟩ = (|0⟩ − |1⟩)/√2."""
        s = 1.0 / math.sqrt(2.0)
        return cls(EMLPair.from_values(s, 0.0), EMLPair.from_values(-s, 0.0))

    @classmethod
    def from_bloch(cls, theta: float, phi: float) -> "Qubit":
        """
        Bloch sphere parameterisation:
            |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩.

        e^{iφ} = from_polar(1, φ) — EMLPair, not cmath.exp.
        """
        a = math.cos(theta / 2.0)
        b = math.sin(theta / 2.0)
        return cls(
            EMLPair.from_values(a, 0.0),
            EMLPair.from_polar(b, phi),
        )

    def __repr__(self) -> str:
        return f"Qubit(α={self._alpha}, β={self._beta})"


def bell_phi_plus() -> tuple[EMLPair, TensionPair]:
    """
    Bell state |Φ+⟩ = (|00⟩ + |11⟩)/√2.

    Returns (alpha_00, alpha_11) as TensionPairs.
    A two-qubit entangled state where measuring one qubit collapses the other.
    """
    s = 1.0 / math.sqrt(2.0)
    return EMLPair.from_values(s, 0.0), EMLPair.from_values(s, 0.0)


def bell_phi_minus() -> tuple[EMLPair, TensionPair]:
    """Bell state |Φ−⟩ = (|00⟩ − |11⟩)/√2."""
    s = 1.0 / math.sqrt(2.0)
    return EMLPair.from_values(s, 0.0), EMLPair.from_values(-s, 0.0)


def bell_psi_plus() -> tuple[EMLPair, TensionPair]:
    """Bell state |Ψ+⟩ = (|01⟩ + |10⟩)/√2."""
    s = 1.0 / math.sqrt(2.0)
    return EMLPair.from_values(s, 0.0), EMLPair.from_values(s, 0.0)
