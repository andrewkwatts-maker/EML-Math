"""Tests for Sprint 5 Octonion class."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.octonion import Octonion, basis_octonion, is_g2_automorphism, MULT_TABLE


class TestOctonionBasic:
    def test_construction_8_components(self):
        comps = [EMLPoint(float(i), 1.0) for i in range(8)]
        o = Octonion(comps)
        assert o.component(0) == 0.0
        assert o.component(7) == 7.0

    def test_wrong_length_raises(self):
        with pytest.raises(ValueError):
            Octonion([EMLPoint(1.0, 1.0)] * 7)

    def test_basis_octonion_real_unit(self):
        e0 = basis_octonion(0)
        assert e0.component(0) == 1.0
        for i in range(1, 8):
            assert e0.component(i) == 0.0

    def test_basis_octonion_imag_unit(self):
        e3 = basis_octonion(3)
        assert e3.component(3) == 1.0
        for i in range(8):
            if i != 3:
                assert e3.component(i) == 0.0

    def test_basis_index_out_of_range(self):
        with pytest.raises(ValueError):
            basis_octonion(8)


class TestOctonionMultiplication:
    def test_real_unit_identity(self):
        e0 = basis_octonion(0)
        e1 = basis_octonion(1)
        result = e0 * e1
        assert abs(result.component(1) - 1.0) < 1e-9

    def test_e1_times_e2_is_e4(self):
        # From Fano line (1,2,4): e1*e2 = e4
        e1 = basis_octonion(1)
        e2 = basis_octonion(2)
        result = e1 * e2
        assert abs(result.component(4) - 1.0) < 1e-9

    def test_e2_times_e1_is_minus_e4(self):
        # Anti-commutativity: e2*e1 = -e4
        e1 = basis_octonion(1)
        e2 = basis_octonion(2)
        result = e2 * e1
        assert abs(result.component(4) - (-1.0)) < 1e-9

    def test_e1_sq_is_minus_real(self):
        # e1 * e1 = -e0
        e1 = basis_octonion(1)
        result = e1 * e1
        assert abs(result.component(0) - (-1.0)) < 1e-9

    def test_norm_multiplicative(self):
        o1 = Octonion([EMLPoint(float(i % 3), 1.0) for i in range(8)])
        o2 = Octonion([EMLPoint(float((i + 1) % 4), 1.0) for i in range(8)])
        assert abs(o1.norm() * o2.norm() - (o1 * o2).norm()) < 1e-9

    def test_mult_table_shape(self):
        assert len(MULT_TABLE) == 8
        assert all(len(row) == 8 for row in MULT_TABLE)


class TestOctonionNorm:
    def test_real_unit_norm_one(self):
        assert abs(basis_octonion(0).norm() - 1.0) < 1e-9

    def test_imag_unit_norm_one(self):
        for i in range(1, 8):
            assert abs(basis_octonion(i).norm() - 1.0) < 1e-9

    def test_zero_norm(self):
        zero = Octonion([EMLPoint(0.0, 1.0)] * 8)
        assert abs(zero.norm()) < 1e-9


class TestOctonionConjugate:
    def test_conjugate_real_part_unchanged(self):
        e0 = basis_octonion(0)
        assert abs(e0.conjugate().component(0) - 1.0) < 1e-9

    def test_conjugate_flips_imag(self):
        e1 = basis_octonion(1)
        assert abs(e1.conjugate().component(1) - (-1.0)) < 1e-9

    def test_double_conjugate_identity(self):
        o = Octonion([EMLPoint(float(i), 1.0) for i in range(8)])
        o_cc = o.conjugate().conjugate()
        for i in range(8):
            assert abs(o.component(i) - o_cc.component(i)) < 1e-9


class TestG2Automorphism:
    def test_identity_is_automorphism(self):
        o1 = basis_octonion(1)
        o2 = basis_octonion(2)
        assert is_g2_automorphism(o1, o2, lambda x: x) is True

    def test_negate_is_not_automorphism(self):
        o1 = basis_octonion(1)
        o2 = basis_octonion(2)
        # Negation: norm preserved but product sign flipped
        def negate(o: Octonion) -> Octonion:
            return Octonion([EMLPoint(-o.component(i), 1.0) for i in range(8)])
        # negate(o1 * o2) = -e4; negate(o1)*negate(o2) = (-e1)*(-e2) = e4
        result = is_g2_automorphism(o1, o2, negate)
        assert result is False


# ── new expanded tests ────────────────────────────────────────────────────────

def _make_octonion(scalars):
    """Build an Octonion from a list of 8 floats."""
    return Octonion([EMLPoint(float(s), 1.0) for s in scalars])


class TestAlternativeLaw:
    @pytest.mark.parametrize("i", range(1, 8))
    @pytest.mark.parametrize("j", range(1, 8))
    def test_left_alternative(self, i, j):
        # (a*a)*b == a*(a*b)
        a = basis_octonion(i)
        b = basis_octonion(j)
        lhs = (a * a) * b
        rhs = a * (a * b)
        for k in range(8):
            assert abs(lhs.component(k) - rhs.component(k)) < 1e-9, \
                f"Left alternative law failed for e{i}, e{j}: comp {k}"

    @pytest.mark.parametrize("i", range(1, 8))
    @pytest.mark.parametrize("j", range(1, 8))
    def test_right_alternative(self, i, j):
        # (a*b)*b == a*(b*b)
        a = basis_octonion(i)
        b = basis_octonion(j)
        lhs = (a * b) * b
        rhs = a * (b * b)
        for k in range(8):
            assert abs(lhs.component(k) - rhs.component(k)) < 1e-9, \
                f"Right alternative law failed for e{i}, e{j}: comp {k}"


class TestMoufangIdentity:
    # Fixed octonion triples — deterministic
    _TRIPLES = [
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0]),
        ([1, 1, 0, 0, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0], [0, 0, 1, 1, 0, 0, 0, 0]),
        ([0.5, 0.5, 0, 0, 0, 0, 0, 0], [0, 0.5, 0.5, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0]),
    ]

    @pytest.mark.parametrize("xs,ys,zs", _TRIPLES)
    def test_moufang(self, xs, ys, zs):
        # z*(x*(z*y)) == ((z*x)*z)*y
        x = _make_octonion(xs)
        y = _make_octonion(ys)
        z = _make_octonion(zs)
        lhs = z * (x * (z * y))
        rhs = ((z * x) * z) * y
        for k in range(8):
            assert abs(lhs.component(k) - rhs.component(k)) < 1e-9, \
                f"Moufang identity failed at component {k}"


class TestNormMultiplicativity:
    # 10 fixed pairs of octonions
    _PAIRS = [
        ([1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0]),
        ([0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]),
        ([1, 1, 0, 0, 0, 0, 0, 0], [1, -1, 0, 0, 0, 0, 0, 0]),
        ([2, 3, 0, 0, 0, 0, 0, 0], [1, 0, 2, 0, 0, 0, 0, 0]),
        ([0.5] * 8, [0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]),
        ([1, 0, 0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0, 1, 0]),
        ([3, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 3, 0, 0, 0]),
        ([1, 2, 3, 4, 0, 0, 0, 0], [0, 0, 0, 0, 1, 2, 3, 4]),
        ([1, -1, 1, -1, 0, 0, 0, 0], [0, 0, 1, -1, 1, -1, 0, 0]),
        ([0, 0, 0, 0, 1, 1, 1, 1], [1, 1, 1, 1, 0, 0, 0, 0]),
    ]

    @pytest.mark.parametrize("as_,bs", _PAIRS)
    def test_norm_multiplicative(self, as_, bs):
        a = _make_octonion(as_)
        b = _make_octonion(bs)
        assert abs((a * b).norm() - a.norm() * b.norm()) < 1e-9


class TestConjugateInverse:
    @pytest.mark.parametrize("i", range(1, 8))
    def test_basis_times_conjugate_is_real(self, i):
        o = basis_octonion(i)
        prod = o * o.conjugate()
        for k in range(1, 8):
            assert abs(prod.component(k)) < 1e-9, \
                f"e{i} * conj(e{i}) has non-zero imag component {k}: {prod.component(k)}"

    @pytest.mark.parametrize("i", range(1, 8))
    def test_real_part_equals_norm_sq(self, i):
        o = basis_octonion(i)
        prod = o * o.conjugate()
        assert abs(prod.component(0) - o.norm_sq()) < 1e-9

    def test_general_octonion_conjugate_product_real(self):
        o = _make_octonion([1, 2, 3, 4, 0, 0, 0, 0])
        prod = o * o.conjugate()
        for k in range(1, 8):
            assert abs(prod.component(k)) < 1e-9

    def test_general_octonion_real_part_equals_norm_sq(self):
        o = _make_octonion([1, 2, 3, 4, 0, 0, 0, 0])
        prod = o * o.conjugate()
        assert abs(prod.component(0) - o.norm_sq()) < 1e-9


class TestAllBasisProducts:
    """Verify every e_i * e_j against the MULT_TABLE."""

    @pytest.mark.parametrize("i", range(8))
    @pytest.mark.parametrize("j", range(8))
    def test_product_matches_mult_table(self, i, j):
        ei = basis_octonion(i)
        ej = basis_octonion(j)
        result = ei * ej
        expected_sign, expected_idx = MULT_TABLE[i][j]
        # The result should have expected_sign at component expected_idx,
        # and zero everywhere else.
        for k in range(8):
            if k == expected_idx:
                assert abs(result.component(k) - float(expected_sign)) < 1e-9, \
                    f"e{i}*e{j}: comp[{k}] = {result.component(k)}, expected {expected_sign}"
            else:
                assert abs(result.component(k)) < 1e-9, \
                    f"e{i}*e{j}: comp[{k}] = {result.component(k)}, expected 0"
