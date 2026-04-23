"""
Quantum field theory simulation using EML arithmetic.

Klein-Gordon: (∂²/∂t² - ∂²/∂x² + m²)φ = 0  — real scalar field
Schrödinger:  iħ ∂ψ/∂t = Hψ                 — via EMLPair, no complex type

The key demonstration: quantum mechanics requires no complex arithmetic.
TensionPair encodes amplitudes ψ = (Re[ψ], Im[ψ]) as two real tensions.
Multiplication by i = rotate_phase(π/2). Time evolution = rotate_phase(-Et/ħ).

EML connection
--------------
The inter-site coupling T_n = eml(|φ_n|, |φ_{n+1}|) defines the EML lattice
action. The KG propagator and path integral weights are exp(-S_EML[path]).
"""
from __future__ import annotations

import math
import random
from typing import Optional

from eml_math.point import EMLPoint
from eml_math.state import EMLState
from eml_math.pair import EMLPair
from eml_math.constants import OVERFLOW_THRESHOLD


class LatticeField:
    """
    Base 1D lattice field on N sites.

    Subclasses provide .step() and field-specific initialisers.
    """

    def __init__(
        self,
        N: int,
        field_values: list[float],
        spacing: float = 1.0,
        dt: float = 0.1,
        boundary: str = "periodic",
        D: Optional[float] = None,
    ) -> None:
        if len(field_values) != N:
            raise ValueError(f"field_values length {len(field_values)} != N={N}")
        self._N = N
        self._phi = list(field_values)
        self._spacing = spacing
        self._dt = dt
        self._boundary = boundary
        self._D = D

    @property
    def N(self) -> int:
        return self._N

    @property
    def field(self) -> list[float]:
        return list(self._phi)

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def spacing(self) -> float:
        return self._spacing

    def _nbr(self, i: int, offset: int) -> int:
        """Neighbour index with boundary condition."""
        if self._boundary == "periodic":
            return (i + offset) % self._N
        j = i + offset
        return j if 0 <= j < self._N else -1

    def _get(self, i: int) -> float:
        return 0.0 if i == -1 else self._phi[i]

    def laplacian(self, i: int) -> float:
        """Discrete Laplacian ∇²φ at site i."""
        il = self._nbr(i, -1)
        ir = self._nbr(i, +1)
        return (self._get(ir) - 2.0 * self._phi[i] + self._get(il)) / (self._spacing ** 2)

    def evolve(self, steps: int) -> list[list[float]]:
        """Run for `steps` time steps; return field snapshots (length steps+1)."""
        snapshots = [self.field]
        for _ in range(steps):
            self.step()
            snapshots.append(self.field)
        return snapshots

    def step(self) -> None:
        raise NotImplementedError


class KleinGordonField(LatticeField):
    """
    Real scalar Klein-Gordon field: (∂²φ/∂t² − ∂²φ/∂x² + m²φ) = 0.

    Leapfrog (Störmer-Verlet) discretisation:
        φ(t+dt) = 2φ(t) − φ(t−dt) + dt²·[∇²φ − m²φ]

    Stable when CFL number dt/spacing < 1.
    All values strictly real — no complex arithmetic.

    EML connection
    --------------
    The inter-site EML tension T_n = eml(|φ_n|, |φ_{n+1}|) = exp(|φ_n|) − ln(|φ_{n+1}|)
    is the elementary coupling between adjacent sites in the EML lattice action.
    """

    def __init__(
        self,
        N: int,
        field_values: list[float],
        field_velocities: Optional[list[float]] = None,
        mass: float = 1.0,
        spacing: float = 1.0,
        dt: float = 0.05,
        boundary: str = "periodic",
        D: Optional[float] = None,
    ) -> None:
        super().__init__(N, field_values, spacing, dt, boundary, D)
        self._mass = mass
        vel = field_velocities if field_velocities is not None else [0.0] * N
        # φ_prev via backward step from initial velocity
        self._phi_prev = [self._phi[i] - dt * vel[i] for i in range(N)]

    @classmethod
    def gaussian_packet(
        cls,
        N: int,
        mass: float = 1.0,
        center: Optional[float] = None,
        width: float = 2.0,
        amplitude: float = 1.0,
        spacing: float = 1.0,
        dt: float = 0.05,
        boundary: str = "periodic",
    ) -> "KleinGordonField":
        """Gaussian initial data φ(x) = A·exp(−(x−x₀)²/(2σ²))."""
        if center is None:
            center = N * spacing / 2.0
        phi = [
            amplitude * math.exp(-((i * spacing - center) ** 2) / (2.0 * width ** 2))
            for i in range(N)
        ]
        return cls(N, phi, mass=mass, spacing=spacing, dt=dt, boundary=boundary)

    def step(self) -> None:
        """One leapfrog step."""
        dt2 = self._dt ** 2
        phi_new = [
            2.0 * self._phi[i]
            - self._phi_prev[i]
            + dt2 * (self.laplacian(i) - self._mass ** 2 * self._phi[i])
            for i in range(self._N)
        ]
        self._phi_prev = self._phi
        self._phi = phi_new

    def energy_density(self) -> list[float]:
        """
        Local energy ε_n = ½(∂φ/∂t)² + ½(∂φ/∂x)² + ½m²φ².

        Time derivative estimated from leapfrog half-step.
        """
        result = []
        for i in range(self._N):
            dphi_dt = (self._phi[i] - self._phi_prev[i]) / self._dt
            ir = self._nbr(i, 1)
            grad = (self._get(ir) - self._phi[i]) / self._spacing
            result.append(0.5 * (dphi_dt ** 2 + grad ** 2 + self._mass ** 2 * self._phi[i] ** 2))
        return result

    def total_energy(self) -> float:
        return sum(self.energy_density()) * self._spacing

    def eml_inter_site_tension(self) -> list[float]:
        """
        EML coupling T_n = eml(|φ_n|, |φ_{n+1}|) between adjacent sites.

        This is the EML Sheffer operator applied to the field amplitudes,
        giving the natural lattice action for the EML field theory.
        """
        result = []
        for i in range(self._N):
            ir = self._nbr(i, 1)
            a = abs(self._phi[i]) or 1e-300
            b = abs(self._get(ir)) or 1e-300
            xv = min(a, OVERFLOW_THRESHOLD)
            result.append(math.exp(xv) - math.log(b))
        return result

    def cfl_number(self) -> float:
        """CFL number = dt/spacing. Stable when < 1."""
        return self._dt / self._spacing


class SchrodingerField:
    """
    1D Schrödinger equation: iħ ∂ψ/∂t = Hψ.

    Wave function stored as a list of TensionPair — one amplitude per site.
    All complex arithmetic replaced by TensionPair operations:

        Multiply by i            →  pair.rotate_phase(π/2)
        Complex conjugate        →  pair.conjugate()
        Phase evolution e^{-iφ}  →  pair.rotate_phase(−φ)
        Probability |ψ|²         →  pair.modulus**2
        Norm ∑|ψ_n|²             →  sum of modulus² (× spacing)

    No Python complex type is used anywhere in this module.

    Demonstration
    -------------
    For an energy eigenstate |E⟩, the exact time evolution is:

        |E(t)⟩ = e^{−iEt/ħ}|E(0)⟩

    In standard QM this requires complex exponentiation via Euler's formula.
    In MPM this is just:

        psi_t = psi_0.rotate_phase(−E * t / hbar)

    A pure geometric rotation — self-evidently real, no imaginary unit needed.
    """

    def __init__(
        self,
        N: int,
        hbar: float = 1.0,
        mass: float = 1.0,
        spacing: float = 1.0,
        dt: float = 0.01,
        potential: Optional[list[float]] = None,
    ) -> None:
        self._N = N
        self._hbar = hbar
        self._mass = mass
        self._spacing = spacing
        self._dt = dt
        self._V = potential if potential is not None else [0.0] * N
        self._psi: list[TensionPair] = [EMLPair.from_values(0.0, 0.0) for _ in range(N)]

    @property
    def spacing(self) -> float:
        return self._spacing

    @property
    def psi(self) -> list[TensionPair]:
        return list(self._psi)

    def set_psi(self, psi: list[TensionPair]) -> None:
        if len(psi) != self._N:
            raise ValueError(f"psi length {len(psi)} != N={self._N}")
        self._psi = list(psi)

    def probability(self) -> list[float]:
        """Born rule: P_n = |ψ_n|² at each site."""
        return [p.modulus ** 2 for p in self._psi]

    def norm(self) -> float:
        """∑|ψ_n|²·spacing — should remain ≈ 1 for normalised states."""
        return sum(self.probability()) * self._spacing

    def normalize(self) -> None:
        n = self.norm()
        if n < 1e-300:
            return
        s = 1.0 / math.sqrt(n)
        self._psi = [EMLPair.from_values(p.real_tension * s, p.imag_tension * s)
                     for p in self._psi]

    def step(self) -> None:
        """
        One potential-phase step: ψ_n → e^{−iV_n·dt/ħ} · ψ_n.

        In MPM: ψ_n.rotate_phase(−V_n·dt/ħ).

        This is the potential half of the Trotter split-operator method.
        The complex exponential e^{−iVt/ħ} is a pure phase rotation —
        TensionPair makes this geometrically transparent.
        """
        self._psi = [
            self._psi[i].rotate_phase(-self._V[i] * self._dt / self._hbar)
            for i in range(self._N)
        ]

    def evolve_eigenstate(self, energy: float, t: float) -> list[TensionPair]:
        """
        Exact time evolution for a single energy eigenstate.

        |E(t)⟩ = e^{−iEt/ħ}|E(0)⟩ = psi.rotate_phase(−E·t/ħ)

        No ODE solver, no complex exponential, no Euler formula.
        The Schrödinger equation for eigenstates is a pure rotation.
        """
        angle = -energy * t / self._hbar
        return [p.rotate_phase(angle) for p in self._psi]

    @classmethod
    def particle_in_box(
        cls,
        N: int,
        n_mode: int = 1,
        hbar: float = 1.0,
        mass: float = 1.0,
        spacing: float = 1.0,
        dt: float = 0.01,
    ) -> "SchrodingerField":
        """
        n-th energy eigenstate of the infinite square well.

        ψ_k = √(2/L)·sin(nπk/N),  E_n = (nπħ)²/(2mL²).
        """
        field = cls(N, hbar=hbar, mass=mass, spacing=spacing, dt=dt)
        L = N * spacing
        A = math.sqrt(2.0 / L)
        psi = [EMLPair.from_values(A * math.sin(n_mode * math.pi * i / N), 0.0)
               for i in range(N)]
        field.set_psi(psi)
        return field

    @classmethod
    def gaussian(
        cls,
        N: int,
        center: Optional[float] = None,
        width: float = 2.0,
        momentum: float = 0.0,
        hbar: float = 1.0,
        mass: float = 1.0,
        spacing: float = 1.0,
        dt: float = 0.01,
    ) -> "SchrodingerField":
        """
        Gaussian wave packet with momentum p₀.

        ψ_k = A·exp(−(x−x₀)²/(4σ²))·e^{ip₀x/ħ}

        The plane-wave phase e^{ip₀x/ħ} encoded via EMLPair.from_polar:
            (cos(p₀x/ħ), sin(p₀x/ħ)) — no complex exponential required.
        """
        if center is None:
            center = N * spacing / 2.0
        field = cls(N, hbar=hbar, mass=mass, spacing=spacing, dt=dt)
        psi = []
        for i in range(N):
            x = i * spacing
            env = math.exp(-((x - center) ** 2) / (4.0 * width ** 2))
            phase = momentum * x / hbar
            psi.append(EMLPair.from_polar(env, phase))
        field.set_psi(psi)
        field.normalize()
        return field

    @staticmethod
    def particle_in_box_energy(n: int, N: int, hbar: float = 1.0, mass: float = 1.0,
                               spacing: float = 1.0) -> float:
        """E_n = (nπħ)² / (2m L²) where L = N·spacing."""
        L = N * spacing
        return (n * math.pi * hbar) ** 2 / (2.0 * mass * L ** 2)


class PathIntegral:
    """
    Discrete Euclidean path integral.

    Standard action (for comparison):
        S_harm[path] = Σ [m(φ_{t+1}−φ_t)²/(2dt) + m·ω²·dt·φ_t²/2]

    EML action (MPM's native lattice action):
        S_EML[path] = Σ eml(|φ_t|, |φ_{t+1}|) = Σ exp(|φ_t|) − ln(|φ_{t+1}|)

    Both compute Z = Σ_paths exp(−S[path]) via Monte Carlo.
    The EML action is the one that emerges naturally from the Sheffer structure.
    """

    def __init__(self, mass: float = 1.0) -> None:
        self._mass = mass

    def eml_action(self, path: list[float]) -> float:
        """S_EML = Σ eml(|φ_t|, |φ_{t+1}|)."""
        S = 0.0
        for t in range(len(path) - 1):
            a = abs(path[t]) or 1e-300
            b = abs(path[t + 1]) or 1e-300
            S += math.exp(min(a, OVERFLOW_THRESHOLD)) - math.log(b)
        return S

    def harmonic_action(self, path: list[float], dt: float = 0.1, omega: float = 1.0) -> float:
        """Standard Euclidean harmonic oscillator action."""
        S = 0.0
        for t in range(len(path) - 1):
            S += self._mass * (path[t + 1] - path[t]) ** 2 / (2.0 * dt)
            S += self._mass * omega ** 2 * dt * path[t] ** 2 / 2.0
        return S

    def compute(
        self,
        x_initial: float,
        x_final: float,
        T: int = 10,
        num_paths: int = 300,
        dt: float = 0.1,
        use_eml_action: bool = False,
        seed: Optional[int] = None,
    ) -> float:
        """
        Monte Carlo estimate of Z(x_i → x_f).

        Returns mean of exp(−S[path]) over sampled paths.
        """
        rng = random.Random(seed)
        total = 0.0
        count = 0
        action_fn = self.eml_action if use_eml_action else (
            lambda p: self.harmonic_action(p, dt=dt)
        )
        for _ in range(num_paths):
            path = [x_initial]
            for t in range(1, T):
                frac = t / T
                mid = (1.0 - frac) * x_initial + frac * x_final
                path.append(mid + rng.gauss(0.0, math.sqrt(dt)))
            path.append(x_final)
            S = action_fn(path)
            if S < 500:
                total += math.exp(-S)
                count += 1
        return total / count if count > 0 else 0.0
