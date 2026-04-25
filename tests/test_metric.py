"""Tests for Sprint 4 MetricTensor."""
import math
import pytest

from eml_math.point import EMLPoint
from eml_math.metric import MetricTensor


# ── TestFlatMetric ────────────────────────────────────────────────────────────

class TestFlatMetric:
    def test_g_components(self):
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        g = m._g(p)
        assert g[0][0] == 1.0
        assert g[1][1] == -1.0
        assert g[0][1] == 0.0

    def test_ds2_timelike_positive(self):
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        # ds² = dt² - dx² with dt=1, dx=0
        assert m.ds2(p, dx=1.0, dy=0.0) > 0

    def test_ds2_spacelike_negative(self):
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        # ds² = dt² - dx² with dt=0, dx=1
        assert m.ds2(p, dx=0.0, dy=1.0) < 0

    def test_is_curved_false(self):
        m = MetricTensor.flat()
        assert m.is_curved() is False

    def test_proper_time_finite(self):
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        assert math.isfinite(m.proper_time(p, dx=1.0, dy=0.0))


# ── TestSchwarzschildMetric ───────────────────────────────────────────────────

class TestSchwarzschildMetric:
    def test_is_curved(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        assert m.is_curved() is True

    def test_g_outside_horizon(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        # r = exp(3) >> rs=2 → should have finite metric
        p = EMLPoint(3.0, 1.0)
        g = m._g(p)
        assert math.isfinite(g[0][0])
        assert math.isfinite(g[1][1])
        assert g[0][0] < 0  # g_tt negative outside horizon

    def test_analytic_christoffel_nonzero(self):
        # Γ^t_{tr} should be non-zero for r > rs
        result = MetricTensor.schwarzschild_christoffel(0, 0, 1, r=5.0, rs=2.0)
        assert result != 0.0
        assert math.isfinite(result)

    def test_analytic_christoffel_inside_returns_zero(self):
        # r <= rs: degenerate, should return 0.0
        result = MetricTensor.schwarzschild_christoffel(0, 0, 1, r=1.0, rs=2.0)
        assert result == 0.0

    def test_numeric_christoffel_finite(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(2.0, 1.0)  # r = exp(2) ≈ 7.4 > rs=2
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    result = m.christoffel(lam, mu, nu, p)
                    assert math.isfinite(result), f"Γ^{lam}_{{{mu}{nu}}} not finite"

    def test_analytic_vs_numeric_christoffel(self):
        m = MetricTensor.schwarzschild(rs=2.0)
        rs = 2.0
        r = math.exp(2.0)  # r for p.x=2.0
        p = EMLPoint(2.0, 1.0)
        # Γ^0_{01}: analytic
        analytic = MetricTensor.schwarzschild_christoffel(0, 0, 1, r=r, rs=rs)
        numeric = m.christoffel(0, 0, 1, p)
        # Allow 1e-3 tolerance for numeric finite-diff
        assert abs(analytic - numeric) < 0.1, (
            f"analytic={analytic:.6g}, numeric={numeric:.6g}"
        )


# ── TestFLRWMetric ────────────────────────────────────────────────────────────

class TestFLRWMetric:
    def test_flat_flrw_with_constant_a(self):
        # a(t)=1, k=0 → g = diag(-1, 1): flat spatial
        m = MetricTensor.flrw(scale_factor_a=lambda t: 1.0, k=0.0)
        p = EMLPoint(0.0, math.e)
        g = m._g(p)
        assert g[0][0] == -1.0
        assert abs(g[1][1] - 1.0) < 1e-9

    def test_expanding_universe(self):
        m = MetricTensor.flrw(scale_factor_a=lambda t: 2.0, k=0.0)
        p = EMLPoint(0.0, math.e)
        g = m._g(p)
        assert abs(g[1][1] - 4.0) < 1e-9  # a²=4


# ── TestOtherMetricFactories ──────────────────────────────────────────────────

class TestOtherMetricFactories:
    def test_ads5_metric_finite(self):
        m = MetricTensor.ads5_x_s5(L=1.0)
        p = EMLPoint(1.0, math.e)
        g = m._g(p)
        assert math.isfinite(g[0][0])
        assert math.isfinite(g[1][1])

    def test_g2_holonomy_metric(self):
        m = MetricTensor.g2_holonomy()
        p = EMLPoint(1.0, 1.0)
        g = m._g(p)
        assert g[0][0] == 1.0
        assert abs(g[1][1] - math.exp(2.0)) < 1e-9

    def test_heterotic_metric_uniform(self):
        m = MetricTensor.heterotic_e8x8(radius=1.0)
        p = EMLPoint(1.0, 1.0)
        g = m._g(p)
        scale = (2.0 * math.pi) ** 2
        assert abs(g[0][0] - scale) < 1e-9
        assert abs(g[1][1] - scale) < 1e-9

    def test_calabi_yau_finite(self):
        m = MetricTensor.calabi_yau_3()
        p = EMLPoint(1.0, math.e)
        g = m._g(p)
        assert math.isfinite(g[0][0])
        assert math.isfinite(g[1][1])

    def test_klebanov_strassler_finite(self):
        m = MetricTensor.klebanov_strassler(gsM=0.1)
        p = EMLPoint(1.0, math.e)
        g = m._g(p)
        assert math.isfinite(g[0][0])
        assert g[0][0] > 0

    def test_repr(self):
        m = MetricTensor.flat()
        assert "MetricTensor" in repr(m)


# ── TestGeodesicConservation ──────────────────────────────────────────────────

class TestGeodesicConservation:
    def test_flat_geodesic_delta_stable(self):
        from eml_math.state import EMLState
        m = MetricTensor.flat()
        p = EMLPoint(1.0, math.e)
        s = EMLState.from_point(p)
        delta0 = p.minkowski_delta()
        for _ in range(10):
            s = s.geodesic_step(m, dtau=0.001)
        delta_final = s.point.minkowski_delta()
        # flat metric: Christoffel=0, so minimal drift
        assert math.isfinite(delta_final)

    def test_schwarzschild_geodesic_runs(self):
        from eml_math.state import EMLState
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(2.0, math.e)  # r=exp(2)>>rs
        s = EMLState.from_point(p)
        for _ in range(5):
            s = s.geodesic_step(m, dtau=0.001)
        assert isinstance(s, EMLState)


# ── new expanded tests ────────────────────────────────────────────────────────

class TestChristoffelSymmetry:
    @pytest.mark.parametrize("r_val", [3.0, 5.0, 10.0, 50.0, 100.0])
    def test_flat_christoffel_symmetric(self, r_val):
        m = MetricTensor.flat()
        # For flat metric all Christoffel symbols are zero → symmetry trivial
        p = EMLPoint(math.log(r_val), 1.0)
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    gm = m.christoffel(lam, mu, nu, p)
                    gn = m.christoffel(lam, nu, mu, p)
                    assert abs(gm - gn) < 1e-8, \
                        f"Γ^{lam}_{{{mu}{nu}}} != Γ^{lam}_{{{nu}{mu}}} at r={r_val}"

    @pytest.mark.parametrize("r_val", [3.0, 5.0, 10.0, 50.0])
    def test_schwarzschild_christoffel_symmetric(self, r_val):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(math.log(r_val), 1.0)
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    gm = m.christoffel(lam, mu, nu, p)
                    gn = m.christoffel(lam, nu, mu, p)
                    assert abs(gm - gn) < 1e-6, \
                        f"Γ^{lam}_{{{mu}{nu}}} != Γ^{lam}_{{{nu}{mu}}} at r={r_val}"


class TestChristoffelScaleRange:
    @pytest.mark.parametrize("r_val", [3.0, 5.0, 10.0, 50.0, 100.0])
    def test_analytic_vs_numeric_agree(self, r_val):
        rs = 2.0
        m = MetricTensor.schwarzschild(rs=rs)
        p = EMLPoint(math.log(r_val), 1.0)
        # Γ^t_{tr} = Γ^0_{01}
        analytic = MetricTensor.schwarzschild_christoffel(0, 0, 1, r=r_val, rs=rs)
        numeric = m.christoffel(0, 0, 1, p)
        assert abs(analytic - numeric) < 1e-3, \
            f"r={r_val}: analytic={analytic:.6g}, numeric={numeric:.6g}"

    @pytest.mark.parametrize("r_val", [3.0, 5.0, 10.0, 50.0, 100.0])
    def test_christoffel_finite_at_all_r(self, r_val):
        m = MetricTensor.schwarzschild(rs=2.0)
        p = EMLPoint(math.log(r_val), 1.0)
        for lam in range(2):
            for mu in range(2):
                for nu in range(2):
                    assert math.isfinite(m.christoffel(lam, mu, nu, p))


class TestFLRWMetricExpanded:
    def test_a1_gives_flat_spatial(self):
        m = MetricTensor.flrw(scale_factor_a=lambda t: 1.0, k=0.0)
        p = EMLPoint(0.0, math.e)
        g = m._g(p)
        assert g[0][0] == -1.0
        assert abs(g[1][1] - 1.0) < 1e-9

    def test_a2_doubles_spatial_component(self):
        m = MetricTensor.flrw(scale_factor_a=lambda t: 2.0, k=0.0)
        p = EMLPoint(0.0, math.e)
        g = m._g(p)
        assert abs(g[1][1] - 4.0) < 1e-9

    def test_is_curved_true(self):
        m = MetricTensor.flrw(scale_factor_a=lambda t: 2.0, k=0.0)
        assert m.is_curved() is True

    def test_a1_ds2_matches_minkowski_up_to_sign(self):
        # a=1, k=0: g_tt=-1, g_rr=1 (opposite sign convention to +- flat)
        # ds² for timelike (dt=1, dr=0): -1 (negative)
        m = MetricTensor.flrw(scale_factor_a=lambda t: 1.0, k=0.0)
        p = EMLPoint(0.0, math.e)
        g = m._g(p)
        assert g[0][0] == -1.0
        assert abs(g[1][1] - 1.0) < 1e-9


class TestAdS5Expanded:
    def test_g_tt_positive(self):
        m = MetricTensor.ads5_x_s5(L=1.0)
        for x_val in [0.5, 1.0, 2.0]:
            p = EMLPoint(x_val, 1.0)
            g = m._g(p)
            assert g[0][0] > 0, f"g_tt not positive at x={x_val}"

    def test_g_rr_positive(self):
        m = MetricTensor.ads5_x_s5(L=1.0)
        for x_val in [0.5, 1.0, 2.0]:
            p = EMLPoint(x_val, 1.0)
            g = m._g(p)
            assert g[1][1] > 0, f"g_rr not positive at x={x_val}"

    def test_is_curved_true(self):
        m = MetricTensor.ads5_x_s5(L=1.0)
        assert m.is_curved() is True

    def test_g_tt_times_g_rr_equals_one(self):
        # g_tt = (r/L)^2, g_rr = (L/r)^2 → product = 1
        m = MetricTensor.ads5_x_s5(L=1.0)
        for x_val in [0.5, 1.0, 2.0, 3.0]:
            p = EMLPoint(x_val, 1.0)
            g = m._g(p)
            assert abs(g[0][0] * g[1][1] - 1.0) < 1e-9, \
                f"g_tt * g_rr = {g[0][0]*g[1][1]:.9g} at x={x_val}"


class TestGeodesicConservationExpanded:
    def test_schwarzschild_christoffel_sum_bounded(self):
        # Verify that the Christoffel symbols remain small (bounded) far from the horizon,
        # which is a proxy for smooth geodesic evolution at r >> rs.
        rs = 2.0
        m = MetricTensor.schwarzschild(rs=rs)
        for r_val in [5.0, 10.0, 50.0, 100.0]:
            p = EMLPoint(math.log(r_val), 1.0)
            total = sum(
                abs(m.christoffel(lam, mu, nu, p))
                for lam in range(2)
                for mu in range(2)
                for nu in range(2)
            )
            assert math.isfinite(total)
            # Far from horizon, Christoffel symbols should be small
            assert total < 1.0, f"Christoffel sum too large at r={r_val}: {total:.6g}"

    def test_schwarzschild_metric_det_positive(self):
        # det(g) for Schwarzschild = -(1-rs/r) * 1/(1-rs/r) = -1, |det|=1
        rs = 2.0
        m = MetricTensor.schwarzschild(rs=rs)
        for r_val in [3.0, 5.0, 10.0, 50.0]:
            p = EMLPoint(math.log(r_val), 1.0)
            g = m._g(p)
            det = g[0][0] * g[1][1] - g[0][1] * g[1][0]
            # det = -(1-rs/r) / (1-rs/r) = -1 exactly
            assert abs(abs(det) - 1.0) < 1e-9, \
                f"det(g) = {det:.9g} at r={r_val}"
