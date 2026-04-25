"""
EMLMultivector — Clifford algebra multivector over EMLPoint components.

The geometric product is the fundamental product of Clifford algebra; the
inner (dot) and outer (wedge) products are derived from it. For grade-1
vectors a, b the decomposition a·b + a∧b = ab holds, but this identity
does not generalise to arbitrary multivectors — for general multivectors the
geometric product is irreducible.

Supports arbitrary metric signatures (e.g., (1,-1) for Minkowski 2D,
(1,-1,-1,-1) for spacetime, (1,)*7 for G2, (1,)*8 for E8).
"""
from __future__ import annotations

import math
from typing import List, Tuple, Optional, Callable

from eml_math.point import EMLPoint

try:
    from eml_math import eml_core as _core
    _RUST_GA = True
except ImportError:
    _RUST_GA = False


class EMLMultivector:
    """
    Multivector in a Clifford algebra Cl(p, q) with metric signature.

    The algebra has dimension 2^n where n = len(signature). Each basis blade
    is encoded as a bitmask; component[k] is the coefficient of blade k.

    Parameters
    ----------
    components : list[EMLPoint]
        2^n EMLPoints. The x-coordinate of each is the scalar coefficient.
        n must satisfy 2^n == len(components).
    signature : tuple[int, ...]
        Metric signature (+1 or -1) for each basis vector. Length n.
        Default (1, -1) for 2D Minkowski.
    """

    __slots__ = ("_comps", "_sig", "_n")

    def __init__(
        self,
        components: List[EMLPoint],
        signature: Tuple[int, ...] = (1, -1),
    ) -> None:
        n = len(signature)
        expected = 1 << n  # 2^n
        if len(components) != expected:
            raise ValueError(
                f"signature of length {n} requires {expected} components, "
                f"got {len(components)}"
            )
        self._comps = list(components)
        self._sig = tuple(signature)
        self._n = n

    # ── component access ──────────────────────────────────────────────────────

    def _scalars(self) -> List[float]:
        return [p.x for p in self._comps]

    def scalar_part(self) -> float:
        return self._comps[0].x

    def grade(self, k: int) -> "EMLMultivector":
        """Project onto grade-k subspace."""
        scalars = [0.0] * len(self._comps)
        for mask in range(len(self._comps)):
            if bin(mask).count("1") == k:
                scalars[mask] = self._comps[mask].x
        return EMLMultivector(
            [EMLPoint(s, 1.0) for s in scalars],
            signature=self._sig,
        )

    # ── geometric product ─────────────────────────────────────────────────────

    def _blade_product(self, a_mask: int, b_mask: int) -> Tuple[float, int]:
        """
        Compute e_A * e_B for basis blades A, B (encoded as bitmasks).

        Returns (sign, result_mask).
        """
        sign = 1.0
        result = a_mask ^ b_mask  # XOR: symmetric difference

        # Count sign changes from anti-commutativity
        tmp = a_mask
        b = b_mask
        while b:
            lsb = b & (-b)
            bit_pos = lsb.bit_length() - 1
            # Count set bits in tmp above this position
            above = bin(tmp >> (bit_pos + 1)).count("1")
            sign *= (-1) ** above
            # Apply metric: if the bit was in both, e_i² = sig[i]
            if a_mask & lsb:
                i = bit_pos
                sign *= self._sig[i]
            b &= b - 1

        return sign, result

    def __mul__(self, other: "EMLMultivector") -> "EMLMultivector":
        """Full geometric product."""
        if self._sig != other._sig:
            raise ValueError("Cannot multiply multivectors with different signatures")
        a = self._scalars()
        b = other._scalars()
        if _RUST_GA and len(self._sig) <= 8:
            sig_i8 = [int(s) for s in self._sig]
            result = list(_core.geometric_product_n([a], [b], sig_i8)[0])
        else:
            dim = len(self._comps)
            result = [0.0] * dim
            for i in range(dim):
                if a[i] == 0.0:
                    continue
                for j in range(dim):
                    if b[j] == 0.0:
                        continue
                    sign, k = self._blade_product(i, j)
                    result[k] += sign * a[i] * b[j]
        return EMLMultivector(
            [EMLPoint(v, 1.0) for v in result],
            signature=self._sig,
        )

    def __add__(self, other: "EMLMultivector") -> "EMLMultivector":
        if self._sig != other._sig:
            raise ValueError("Signature mismatch in addition")
        return EMLMultivector(
            [EMLPoint(a.x + b.x, 1.0) for a, b in zip(self._comps, other._comps)],
            signature=self._sig,
        )

    def __sub__(self, other: "EMLMultivector") -> "EMLMultivector":
        if self._sig != other._sig:
            raise ValueError("Signature mismatch in subtraction")
        return EMLMultivector(
            [EMLPoint(a.x - b.x, 1.0) for a, b in zip(self._comps, other._comps)],
            signature=self._sig,
        )

    def reverse(self) -> "EMLMultivector":
        """Grade reversal ~A: reverses order of each blade's basis vectors."""
        scalars = self._scalars()
        result = list(scalars)
        for mask in range(len(self._comps)):
            grade_k = bin(mask).count("1")
            if grade_k * (grade_k - 1) // 2 % 2 == 1:
                result[mask] = -scalars[mask]
        return EMLMultivector(
            [EMLPoint(v, 1.0) for v in result],
            signature=self._sig,
        )

    def quadratic(self) -> float:
        """
        Quadratic form v·v = scalar part of (v * reverse(v)).

        For a grade-1 vector v = Σ vᵢ eᵢ:
            v·v = Σ sig[i] * vᵢ²

        This unifies euclidean_delta (sig=(1,1,...)) and
        minkowski_delta (sig=(1,-1,-1,...)).
        """
        # Grade-1 projection
        n = len(self._comps)
        result = 0.0
        for mask in range(n):
            if bin(mask).count("1") == 1:
                i = mask.bit_length() - 1
                vi = self._comps[mask].x
                result += self._sig[i] * vi * vi
        return result

    # ── rotors ────────────────────────────────────────────────────────────────

    def rotor(self, angle: float, plane: Tuple[int, int]) -> "EMLMultivector":
        """
        Compute the rotor R = exp(-B*θ/2) for rotation in the e_{i}∧e_{j} plane.

        R = cos(θ/2) - sin(θ/2) * e_i∧e_j

        Note: this formula is correct for Euclidean signature planes where
        (e_i∧e_j)² = -1. For timelike planes in Minkowski signature (where
        (e_0∧e_k)² = +1), the rotation becomes a hyperbolic rotation (Lorentz
        boost) and the rotor takes the form:
            R = cosh(θ/2) - sinh(θ/2) * e_0∧e_k
        The current implementation always uses sin/cos and is only correct
        for spacelike rotation planes in Euclidean or Minkowski signature.

        Parameters
        ----------
        angle : float
            Rotation angle (or rapidity for timelike planes — see note above).
        plane : (int, int)
            Indices (i, j) of the rotation plane (0-indexed).
        """
        i, j = plane
        if i >= self._n or j >= self._n:
            raise ValueError(f"plane indices {plane} out of range for n={self._n}")
        half = angle / 2.0
        cos_h = math.cos(half)
        sin_h = math.sin(half)

        dim = 1 << self._n
        scalars = [0.0] * dim
        scalars[0] = cos_h  # scalar part
        blade = (1 << i) | (1 << j)  # e_i ∧ e_j bitmask
        scalars[blade] = -sin_h
        return EMLMultivector(
            [EMLPoint(s, 1.0) for s in scalars],
            signature=self._sig,
        )

    def rotate(self, R: "EMLMultivector") -> "EMLMultivector":
        """Sandwich product: R * v * ~R."""
        return R * self * R.reverse()

    def exp(self) -> "EMLMultivector":
        """Element-wise exp of scalar coefficients."""
        return EMLMultivector(
            [EMLPoint(math.exp(min(p.x, 709.0)), 1.0) for p in self._comps],
            signature=self._sig,
        )

    # ── factories ─────────────────────────────────────────────────────────────

    @staticmethod
    def spacetime(comps: List[EMLPoint]) -> "EMLMultivector":
        """Spacetime algebra Cl(1,3): signature (1,-1,-1,-1)."""
        if len(comps) != 16:
            raise ValueError(f"spacetime requires 16 components, got {len(comps)}")
        return EMLMultivector(comps, signature=(1, -1, -1, -1))

    @staticmethod
    def g2(comps: List[EMLPoint]) -> "EMLMultivector":
        """G₂ algebra: signature (1,)*7 → 128 components."""
        if len(comps) != 128:
            raise ValueError(f"g2 requires 128 components, got {len(comps)}")
        return EMLMultivector(comps, signature=(1,) * 7)

    @staticmethod
    def flrw(comps: List[EMLPoint], scale_factor: float = 1.0) -> "EMLMultivector":
        """FLRW algebra: signature (-1, 1, 1, 1) → 16 components."""
        if len(comps) != 16:
            raise ValueError(f"flrw requires 16 components, got {len(comps)}")
        return EMLMultivector(comps, signature=(-1, 1, 1, 1))

    @staticmethod
    def e8(comps: List[EMLPoint]) -> "EMLMultivector":
        """E₈ algebra: signature (1,)*8 → 256 components."""
        if len(comps) != 256:
            raise ValueError(f"e8 requires 256 components, got {len(comps)}")
        return EMLMultivector(comps, signature=(1,) * 8)

    @staticmethod
    def leech(comps: List[EMLPoint]) -> "EMLMultivector":
        """Leech algebra: signature (1,)*24 → 2^24 components (use with care)."""
        if len(comps) != (1 << 24):
            raise ValueError(f"leech requires 2^24 components")
        return EMLMultivector(comps, signature=(1,) * 24)

    # ── dunder ────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        s = self.scalar_part()
        return f"EMLMultivector(sig={self._sig}, scalar={s:.6g}, dim={len(self._comps)})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, EMLMultivector):
            return NotImplemented
        if self._sig != other._sig:
            return False
        return all(
            abs(a.x - b.x) < 1e-9
            for a, b in zip(self._comps, other._comps)
        )
