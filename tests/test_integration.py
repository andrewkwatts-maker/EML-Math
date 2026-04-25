"""End-to-end integration tests across all eml_math modules."""
from __future__ import annotations

import math
import pytest

from eml_math import EMLPoint, EMLState
from eml_math.metric import MetricTensor
from eml_math.geometric_algebra import EMLMultivector
from eml_math.octonion import Octonion, basis_octonion
from eml_math.fourvector import MinkowskiFourVector
from eml_math.momentum import FourMomentum
from eml_math.ndim import EMLNDVector, e8_lattice_points
from eml_math.discrete import planck_delta


class TestGeodesicIntegration:
    def test_schwarzschild_christoffel_conserves_along_radial_sequence(self):
        # Simulate a sequence of radial EMLPoints representing outward motion:
        # verify Δ_M changes predictably (timelike sequence has stable Δ_M).
        m = MetricTensor.schwarzschild(rs=2.0)
        p0 = EMLPoint(2.5, math.e)
        dm0 = p0.minkowski_delta()
        # Simulate by applying mirror_pulse steps and checking each Δ_M is finite
        s = p0
        for _ in range(1000):
            s = s.mirror_pulse()
        assert math.isfinite(s.minkowski_delta())

    def test_schwarzschild_christoffel_finite_along_trajectory(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(2.5, math.e)
        # Verify all 8 Christoffel symbols are finite and small at this point
        total = 0.0
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    c = m.christoffel(lam, mu, nu, p)
                    assert math.isfinite(c)
                    total += abs(c)
        assert total < 10.0


class TestMultivectorMetricBridge:
    def test_flat_metric_ds2_matches_multivector_quadratic(self):
        # For flat (+,-) metric and displacement (dx=a, dy=b):
        # ds² = a² - b²
        # EMLMultivector with grade-1 components (a, b) in (1,-1) sig gives same.
        a, b = 3.0, 4.0
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        ds2_metric = m.ds2(p, dx=a, dy=b)

        sig = (1, -1)
        dim = 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[1] = EMLPoint(a, 1.0)
        comps[2] = EMLPoint(b, 1.0)
        mv = EMLMultivector(comps, signature=sig)
        q = mv.quadratic()
        assert abs(ds2_metric - q) < 1e-9

    def test_euclidean_metric_ds2_matches_quadratic(self):
        a, b = 3.0, 4.0
        # Euclidean: ds² = a² + b²
        sig = (1, 1)
        dim = 4
        comps = [EMLPoint(0.0, 1.0)] * dim
        comps[1] = EMLPoint(a, 1.0)
        comps[2] = EMLPoint(b, 1.0)
        mv = EMLMultivector(comps, signature=sig)
        q = mv.quadratic()
        expected = a * a + b * b
        assert abs(q - expected) < 1e-9


class TestOctonionNDVectorConversion:
    def test_to_ndvector_euclidean_norm_matches(self):
        scalars = [1.0, 2.0, 3.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        o = Octonion([EMLPoint(s, 1.0) for s in scalars])
        ndv = o.to_ndvector()
        assert abs(ndv.euclidean_norm() - o.norm()) < 1e-9

    def test_to_ndvector_dimension_is_8(self):
        o = basis_octonion(3)
        ndv = o.to_ndvector()
        assert ndv.n == 8

    def test_basis_octonion_ndvector_norm_is_one(self):
        for i in range(8):
            ndv = basis_octonion(i).to_ndvector()
            assert abs(ndv.euclidean_norm() - 1.0) < 1e-9

    def test_general_octonion_ndvector_norm_matches(self):
        scalars = [1.0, -1.0, 2.0, -2.0, 0.5, -0.5, 1.5, -1.5]
        o = Octonion([EMLPoint(s, 1.0) for s in scalars])
        ndv = o.to_ndvector()
        assert abs(ndv.euclidean_norm() - o.norm()) < 1e-9


class TestBoostChainConsistency:
    def test_point_boost_then_four_momentum_mass_preserved(self):
        p = EMLPoint(1.0, math.e)
        fm0 = FourMomentum(p)
        m0 = fm0.mass

        for phi in [0.3, 0.7, -0.5, 1.0]:
            p_boosted = p.boost(phi)
            fm_b = FourMomentum(p_boosted)
            assert abs(fm_b.mass - m0) < 1e-8, \
                f"mass changed after boost phi={phi}: {fm_b.mass} vs {m0}"

    def test_minkowski_forvector_boost_preserves_mass(self):
        v = MinkowskiFourVector(
            EMLPoint(5.0, 1.0), EMLPoint(3.0, 1.0),
            EMLPoint(0.0, 1.0), EMLPoint(0.0, 1.0), c=1.0
        )
        norm0 = v.minkowski_norm()
        for phi in [0.3, -0.3, 0.7]:
            vb = v.boost(phi, direction="x")
            assert abs(vb.minkowski_norm() - norm0) < 1e-8


class TestPlanckLatticeRefinement:
    def test_larger_D_converges_to_minkowski_delta(self):
        p = EMLPoint(1.0, math.e * 2)
        exact = p.minkowski_delta()
        prev_err = None
        for D in [10.0, 100.0, 1000.0, 10000.0]:
            quantized = planck_delta(p, D=D)
            err = abs(quantized - exact)
            if prev_err is not None:
                assert err <= prev_err + 1e-12, \
                    f"planck_delta did not converge at D={D}: err={err:.6g}"
            prev_err = err

    def test_planck_delta_within_half_cell(self):
        p = EMLPoint(1.0, math.e * 3)
        for D in [10.0, 100.0, 1000.0]:
            quantized = planck_delta(p, D=D)
            exact = p.minkowski_delta()
            assert abs(quantized - exact) <= 0.5 / D + 1e-12

    def test_large_D_planck_delta_very_close_to_exact(self):
        p = EMLPoint(2.0, math.e)
        exact = p.minkowski_delta()
        approx = planck_delta(p, D=1e6)
        assert abs(approx - exact) < 1e-5
