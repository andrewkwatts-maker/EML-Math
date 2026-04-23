"""Tests for TensionPair — real replacement for complex numbers."""
import math
import pytest
from eml_math import EMLPair


class TestConstruction:
    def test_unit_i(self):
        i = EMLPair.unit_i()
        assert i.real_tension == pytest.approx(0.0, abs=1e-10)
        assert i.imag_tension == pytest.approx(1.0, rel=1e-10)

    def test_from_values(self):
        z = EMLPair.from_values(3.0, 4.0)
        assert z.real_tension == pytest.approx(3.0, rel=1e-10)
        assert z.imag_tension == pytest.approx(4.0, rel=1e-10)

    def test_one(self):
        one = EMLPair.one()
        assert one.real_tension == pytest.approx(1.0, rel=1e-10)
        assert one.imag_tension == pytest.approx(0.0, abs=1e-10)


class TestModulusArgument:
    def test_modulus_345(self):
        z = EMLPair.from_values(3.0, 4.0)
        assert z.modulus == pytest.approx(5.0, rel=1e-8)

    def test_modulus_unit_i(self):
        assert EMLPair.unit_i().modulus == pytest.approx(1.0, rel=1e-8)

    def test_argument_unit_i(self):
        i = EMLPair.unit_i()
        # arctan(1/0) → π/2 (handled via arctan of large number)
        assert abs(i.argument) > 1.0  # approaches π/2


class TestArithmetic:
    def test_add(self):
        z1 = EMLPair.from_values(1.0, 2.0)
        z2 = EMLPair.from_values(3.0, 4.0)
        z3 = z1 + z2
        assert z3.real_tension == pytest.approx(4.0, rel=1e-10)
        assert z3.imag_tension == pytest.approx(6.0, rel=1e-10)

    def test_mul(self):
        # (1+2i)(3+4i) = (3-8) + (4+6)i = -5 + 10i
        z1 = EMLPair.from_values(1.0, 2.0)
        z2 = EMLPair.from_values(3.0, 4.0)
        z3 = z1 * z2
        assert z3.real_tension == pytest.approx(-5.0, abs=1e-9)
        assert z3.imag_tension == pytest.approx(10.0, rel=1e-9)

    def test_i_squared_is_neg_one(self):
        i = EMLPair.unit_i()
        i2 = i * i    # i² = -1 + 0i
        assert i2.real_tension == pytest.approx(-1.0, abs=1e-9)
        assert i2.imag_tension == pytest.approx(0.0, abs=1e-9)

    def test_conjugate(self):
        z = EMLPair.from_values(3.0, 4.0)
        zc = z.conjugate()
        assert zc.real_tension == pytest.approx(3.0, rel=1e-10)
        assert zc.imag_tension == pytest.approx(-4.0, abs=1e-9)


class TestRotatePhase:
    def test_rotate_half_pi_gives_i_times(self):
        # Rotating (1, 0) by π/2 should give (0, 1) = i
        one = EMLPair.from_values(1.0, 0.0)
        rotated = one.rotate_phase(math.pi / 2)
        assert rotated.real_tension == pytest.approx(0.0, abs=1e-8)
        assert rotated.imag_tension == pytest.approx(1.0, rel=1e-8)

    def test_rotate_pi_gives_negation(self):
        # Rotating (1, 0) by π should give (-1, 0)
        one = EMLPair.from_values(1.0, 0.0)
        rotated = one.rotate_phase(math.pi)
        assert rotated.real_tension == pytest.approx(-1.0, abs=1e-8)
        assert rotated.imag_tension == pytest.approx(0.0, abs=1e-8)


class TestFromPolar:
    def test_unit_magnitude(self):
        z = EMLPair.from_polar(1.0, 0.0)
        assert z.real_tension == pytest.approx(1.0, rel=1e-10)
        assert z.imag_tension == pytest.approx(0.0, abs=1e-10)

    def test_45_degrees(self):
        z = EMLPair.from_polar(math.sqrt(2), math.pi / 4)
        assert z.real_tension == pytest.approx(1.0, rel=1e-8)
        assert z.imag_tension == pytest.approx(1.0, rel=1e-8)
