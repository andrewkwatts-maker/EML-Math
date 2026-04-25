"""Tests for Sprint 6 EMLMultivector (Clifford algebra)."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.geometric_algebra import EMLMultivector


def _zero_mv(sig=(1, -1)):
    """Zero multivector with given signature."""
    dim = 1 << len(sig)
    return EMLMultivector([EMLPoint(0.0, 1.0)] * dim, signature=sig)


def _scalar_mv(val, sig=(1, -1)):
    """Scalar multivector."""
    dim = 1 << len(sig)
    comps = [EMLPoint(0.0, 1.0)] * dim
    comps[0] = EMLPoint(val, 1.0)
    return EMLMultivector(comps, signature=sig)


def _grade1_mv(coeffs, sig=(1, -1)):
    """Grade-1 vector from list of float coefficients."""
    n = len(sig)
    dim = 1 << n
    comps = [EMLPoint(0.0, 1.0)] * dim
    for i, v in enumerate(coeffs):
        comps[1 << i] = EMLPoint(v, 1.0)
    return EMLMultivector(comps, signature=sig)


class TestConstruction:
    def test_basic_2d(self):
        mv = EMLMultivector([EMLPoint(float(i), 1.0) for i in range(4)])
        assert mv.scalar_part() == 0.0

    def test_wrong_component_count_raises(self):
        with pytest.raises(ValueError):
            EMLMultivector([EMLPoint(1.0, 1.0)] * 3, signature=(1, -1))

    def test_repr(self):
        mv = _zero_mv()
        assert "EMLMultivector" in repr(mv)


class TestQuadratic:
    def test_euclidean_2d(self):
        # sig=(1,1): v = (a, b) → v·v = a² + b²
        v = _grade1_mv([3.0, 4.0], sig=(1, 1))
        assert abs(v.quadratic() - 25.0) < 1e-9

    def test_minkowski_2d(self):
        # sig=(1,-1): v = (t, x) → v·v = t² - x²
        v = _grade1_mv([5.0, 3.0], sig=(1, -1))
        assert abs(v.quadratic() - (25.0 - 9.0)) < 1e-9

    def test_zero_lightlike(self):
        # sig=(1,-1): (1, 1) → 1 - 1 = 0
        v = _grade1_mv([1.0, 1.0], sig=(1, -1))
        assert abs(v.quadratic()) < 1e-9


class TestGeometricProduct:
    def test_scalar_times_vector(self):
        # 2 * e1 = 2e1
        s = _scalar_mv(2.0)
        v = _grade1_mv([1.0, 0.0])
        result = s * v
        assert abs(result._comps[1].x - 2.0) < 1e-9

    def test_e1_sq_in_minkowski(self):
        # sig=(1,-1): e1*e1 = sig[0] = +1
        v = _grade1_mv([1.0, 0.0])
        result = v * v
        assert abs(result.scalar_part() - 1.0) < 1e-9

    def test_e2_sq_in_minkowski(self):
        # sig=(1,-1): e2*e2 = sig[1] = -1
        v = _grade1_mv([0.0, 1.0])
        result = v * v
        assert abs(result.scalar_part() - (-1.0)) < 1e-9

    def test_different_sig_raises(self):
        v1 = _grade1_mv([1.0, 0.0], sig=(1, -1))
        v2 = _grade1_mv([1.0, 0.0], sig=(1, 1))
        with pytest.raises(ValueError):
            _ = v1 * v2


class TestRotor:
    def test_rotor_zero_angle_is_identity(self):
        # R(0) = 1 (scalar part = 1, all other parts = 0)
        mv = _zero_mv(sig=(1, 1))
        R = mv.rotor(0.0, plane=(0, 1))
        assert abs(R.scalar_part() - 1.0) < 1e-9

    def test_rotate_preserves_quadratic(self):
        # v = (3, 4), sig=(1,1): rotate by π/4, quadratic should be preserved
        sig = (1, 1)
        v = _grade1_mv([3.0, 4.0], sig=sig)
        q0 = v.quadratic()
        mv = _zero_mv(sig=sig)
        R = mv.rotor(math.pi / 4, plane=(0, 1))
        v_rot = v.rotate(R)
        assert abs(v_rot.quadratic() - q0) < 1e-8

    def test_rotor_out_of_range_raises(self):
        mv = _zero_mv(sig=(1, -1))
        with pytest.raises(ValueError):
            mv.rotor(1.0, plane=(0, 5))


class TestReverse:
    def test_scalar_reverse_unchanged(self):
        s = _scalar_mv(3.0)
        assert abs(s.reverse().scalar_part() - 3.0) < 1e-9

    def test_grade2_reverse_flips_sign(self):
        sig = (1, -1)
        # Build a grade-2 blade (bivector)
        dim = 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[3] = EMLPoint(1.0, 1.0)  # e1∧e2 (mask = 0b11 = 3)
        mv = EMLMultivector(comps, signature=sig)
        rev = mv.reverse()
        assert abs(rev._comps[3].x - (-1.0)) < 1e-9


class TestFactories:
    def test_spacetime_factory(self):
        comps = [EMLPoint(0.0, 1.0)] * 16
        mv = EMLMultivector.spacetime(comps)
        assert mv._sig == (1, -1, -1, -1)

    def test_spacetime_wrong_count_raises(self):
        with pytest.raises(ValueError):
            EMLMultivector.spacetime([EMLPoint(0.0, 1.0)] * 4)

    def test_flrw_factory(self):
        comps = [EMLPoint(0.0, 1.0)] * 16
        mv = EMLMultivector.flrw(comps)
        assert mv._sig == (-1, 1, 1, 1)

    def test_grade_projection(self):
        sig = (1, -1)
        comps = [EMLPoint(float(i), 1.0) for i in range(4)]
        mv = EMLMultivector(comps, signature=sig)
        grade0 = mv.grade(0)
        assert abs(grade0.scalar_part() - 0.0) < 1e-9  # comps[0] = 0.0
        grade1 = mv.grade(1)
        # grade-1 components are at masks 1 (0b01) and 2 (0b10)
        assert abs(grade1._comps[1].x - 1.0) < 1e-9
        assert abs(grade1._comps[2].x - 2.0) < 1e-9


# ── new expanded tests ────────────────────────────────────────────────────────

class TestBasisAnticommutativity:
    def test_e0_e1_anticommute_minkowski(self):
        # sig=(1,-1): e_0*e_1 + e_1*e_0 == 0
        sig = (1, -1)
        e0 = _grade1_mv([1.0, 0.0], sig=sig)
        e1 = _grade1_mv([0.0, 1.0], sig=sig)
        ab = e0 * e1
        ba = e1 * e0
        total = ab + ba
        # All components of the sum must be zero
        for comp in total._comps:
            assert abs(comp.x) < 1e-9

    def test_anticommutativity_euclidean(self):
        sig = (1, 1)
        e0 = _grade1_mv([1.0, 0.0], sig=sig)
        e1 = _grade1_mv([0.0, 1.0], sig=sig)
        ab = e0 * e1
        ba = e1 * e0
        total = ab + ba
        for comp in total._comps:
            assert abs(comp.x) < 1e-9


class TestBasisSquares:
    @pytest.mark.parametrize("sig,expected", [
        ((1, -1), [1.0, -1.0]),
        ((1, 1), [1.0, 1.0]),
        ((-1, -1), [-1.0, -1.0]),
        ((-1, 1), [-1.0, 1.0]),
    ])
    def test_basis_squares(self, sig, expected):
        for i, exp_val in enumerate(expected):
            coeffs = [0.0] * len(sig)
            coeffs[i] = 1.0
            ei = _grade1_mv(coeffs, sig=sig)
            result = ei * ei
            # e_i * e_i = sig[i] * scalar(1)
            assert abs(result.scalar_part() - exp_val) < 1e-9, \
                f"sig={sig}, e{i}^2: got {result.scalar_part()}, expected {exp_val}"

    def test_basis_squares_4d(self):
        # 4D Euclidean: all squares = +1
        sig = (1, 1, 1, 1)
        for i in range(4):
            coeffs = [0.0] * 4
            coeffs[i] = 1.0
            dim = 1 << 4
            comps = [EMLPoint(0.0, 1.0)] * dim
            comps[1 << i] = EMLPoint(1.0, 1.0)
            ei = EMLMultivector(comps, signature=sig)
            result = ei * ei
            assert abs(result.scalar_part() - 1.0) < 1e-9


class TestRotorPreservesNorm:
    @pytest.mark.parametrize("angle", [0.0, math.pi / 6, math.pi / 4, math.pi / 3, math.pi / 2])
    def test_2d_euclidean_rotor_preserves_quadratic(self, angle):
        sig = (1, 1)
        v = _grade1_mv([3.0, 4.0], sig=sig)
        q0 = v.quadratic()
        R = _zero_mv(sig=sig).rotor(angle, plane=(0, 1))
        v_rot = v.rotate(R)
        assert abs(v_rot.quadratic() - q0) < 1e-8

    @pytest.mark.parametrize("angle", [0.0, math.pi / 4, math.pi / 2, math.pi])
    def test_4d_euclidean_rotor_preserves_quadratic(self, angle):
        sig = (1, 1, 1, 1)
        dim = 1 << 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        # Grade-1 vector with all 4 components set to 1
        for i in range(4):
            comps[1 << i] = EMLPoint(1.0, 1.0)
        v = EMLMultivector(comps, signature=sig)
        q0 = v.quadratic()
        # Rotor in plane (0,1)
        zero_mv = EMLMultivector([EMLPoint(0.0, 1.0)] * dim, signature=sig)
        R = zero_mv.rotor(angle, plane=(0, 1))
        v_rot = v.rotate(R)
        assert abs(v_rot.quadratic() - q0) < 1e-8


class TestGradeProjection:
    def test_grade0_extracts_scalar(self):
        sig = (1, -1)
        comps = [EMLPoint(float(i), 1.0) for i in range(4)]
        mv = EMLMultivector(comps, signature=sig)
        g0 = mv.grade(0)
        # Only component at mask=0 survives
        assert abs(g0.scalar_part() - comps[0].x) < 1e-9
        assert abs(g0._comps[1].x) < 1e-9
        assert abs(g0._comps[2].x) < 1e-9
        assert abs(g0._comps[3].x) < 1e-9

    def test_grade1_extracts_vectors(self):
        sig = (1, -1)
        comps = [EMLPoint(float(i + 1), 1.0) for i in range(4)]
        mv = EMLMultivector(comps, signature=sig)
        g1 = mv.grade(1)
        # Masks 1 (0b01) and 2 (0b10) are grade-1
        assert abs(g1._comps[0].x) < 1e-9  # scalar zeroed
        assert abs(g1._comps[1].x - comps[1].x) < 1e-9
        assert abs(g1._comps[2].x - comps[2].x) < 1e-9
        assert abs(g1._comps[3].x) < 1e-9  # bivector zeroed

    def test_grade2_extracts_bivectors(self):
        sig = (1, -1)
        comps = [EMLPoint(float(i + 1), 1.0) for i in range(4)]
        mv = EMLMultivector(comps, signature=sig)
        g2 = mv.grade(2)
        # Mask 3 (0b11) is the only grade-2 component in 2D
        assert abs(g2._comps[0].x) < 1e-9
        assert abs(g2._comps[1].x) < 1e-9
        assert abs(g2._comps[2].x) < 1e-9
        assert abs(g2._comps[3].x - comps[3].x) < 1e-9

    def test_grade_sum_reconstructs_original(self):
        sig = (1, -1)
        comps = [EMLPoint(float(i + 1), 1.0) for i in range(4)]
        mv = EMLMultivector(comps, signature=sig)
        reconstructed = mv.grade(0) + mv.grade(1) + mv.grade(2)
        for k in range(4):
            assert abs(reconstructed._comps[k].x - mv._comps[k].x) < 1e-9


class TestLargeSignature:
    def test_4d_euclidean_geometric_product_associative(self):
        sig = (1, 1, 1, 1)
        dim = 1 << 4

        def _basis_blade_4d(bit):
            comps = [EMLPoint(0.0, 1.0)] * dim
            comps[bit] = EMLPoint(1.0, 1.0)
            return EMLMultivector(comps, signature=sig)

        # e1, e2, e4 (bitmasks 1, 2, 4)
        e1 = _basis_blade_4d(1)
        e2 = _basis_blade_4d(2)
        e4 = _basis_blade_4d(4)

        # (e1 * e2) * e4 == e1 * (e2 * e4)
        lhs = (e1 * e2) * e4
        rhs = e1 * (e2 * e4)
        assert lhs == rhs

    def test_4d_euclidean_scalar_sq_one(self):
        sig = (1, 1, 1, 1)
        dim = 1 << 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[0] = EMLPoint(1.0, 1.0)
        scalar_one = EMLMultivector(comps, signature=sig)
        result = scalar_one * scalar_one
        assert abs(result.scalar_part() - 1.0) < 1e-9
