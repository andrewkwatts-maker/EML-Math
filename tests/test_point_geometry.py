"""Tests for Sprint 1 geometric extensions on EMLPoint and EMLPair."""
from __future__ import annotations

import math
import pytest
from eml_math import EMLPoint, EMLPair


class TestPair:
    def test_returns_emlpair(self):
        p = EMLPoint(0.0, 1.0)
        result = p.pair()
        assert isinstance(result, EMLPair)

    def test_unit_point_pair_values(self):
        # EMLPoint(0, 1): exp(0)=1, ln(1)=0
        p = EMLPoint(0.0, 1.0)
        pair = p.pair()
        assert abs(pair.real_tension - 1.0) < 1e-12
        assert abs(pair.imag_tension - 0.0) < 1e-12

    def test_euler_point_pair_values(self):
        # EMLPoint(1, math.e): exp(1)=e, ln(e)=1
        p = EMLPoint(1.0, math.e)
        pair = p.pair()
        assert abs(pair.real_tension - math.e) < 1e-10
        assert abs(pair.imag_tension - 1.0) < 1e-10

    def test_axiom8_safety_negative_y(self):
        # y=-1: frame shift uses |y|=1, ln(1)=0
        p = EMLPoint(0.0, -1.0)
        pair = p.pair()
        assert pair.real_tension > 0
        assert math.isfinite(pair.imag_tension)

    def test_axiom8_safety_zero_y(self):
        p = EMLPoint(0.0, 0.0)
        pair = p.pair()
        assert math.isfinite(pair.real_tension)
        assert math.isfinite(pair.imag_tension)


class TestEuclideanDelta:
    def test_unit_point_delta(self):
        # pair=(1,0) → Δ=1
        assert abs(EMLPoint(0.0, 1.0).euclidean_delta() - 1.0) < 1e-12

    def test_euler_point_delta(self):
        # pair=(e,1) → Δ=√(e²+1)
        p = EMLPoint(1.0, math.e)
        expected = math.sqrt(math.e ** 2 + 1.0)
        assert abs(p.euclidean_delta() - expected) < 1e-10

    def test_symmetric_pair_delta(self):
        # EMLPoint(0, math.e): exp(0)=1, ln(e)=1 → Δ=√2
        p = EMLPoint(0.0, math.e)
        assert abs(p.euclidean_delta() - math.sqrt(2)) < 1e-12

    def test_positive_always(self):
        for x, y in [(1.0, 2.0), (2.0, 1.0), (0.5, 3.0), (-1.0, 0.5)]:
            assert EMLPoint(x, y).euclidean_delta() >= 0.0


class TestMinkowskiDelta:
    def test_timelike_point(self):
        # EMLPoint(0,1): exp(0)=1, ln(1)=0 → Δ_M = √(1-0) = 1
        p = EMLPoint(0.0, 1.0)
        assert abs(p.minkowski_delta() - 1.0) < 1e-12

    def test_lightlike_point(self):
        # EMLPoint(0, math.e): exp(0)=1, ln(e)=1 → Δ_M = √(1-1) = 0
        p = EMLPoint(0.0, math.e)
        assert p.minkowski_delta() < 1e-9

    def test_spacelike_point(self):
        # EMLPoint(0, math.exp(2)): exp(0)=1, ln(e²)=2 → Δ_M=√(4-1)=√3
        p = EMLPoint(0.0, math.exp(2.0))
        assert abs(p.minkowski_delta() - math.sqrt(3.0)) < 1e-10

    def test_signature_minus_plus(self):
        # Same point, different signature — result is the same (abs of ds²)
        p = EMLPoint(0.0, math.exp(2.0))
        plus = p.minkowski_delta(signature="+---")
        minus = p.minkowski_delta(signature="-+++")
        assert abs(plus - minus) < 1e-12

    def test_c_scaling(self):
        # With c=2: space component scaled by 2
        p = EMLPoint(1.0, math.e)  # exp(1)=e, ln(e)=1
        dm = p.minkowski_delta(c=2.0)
        # ds² = e² - (2*1)² = e²-4
        expected = math.sqrt(abs(math.e ** 2 - 4.0))
        assert abs(dm - expected) < 1e-10


class TestCausalClassification:
    def test_timelike(self):
        p = EMLPoint(0.0, 1.0)  # exp(0)=1 > ln(1)=0
        assert p.is_timelike()
        assert not p.is_spacelike()
        assert not p.is_lightlike()
        assert p.light_cone_type() == "timelike"

    def test_lightlike(self):
        p = EMLPoint(0.0, math.e)  # exp(0)=1, ln(e)=1
        assert p.is_lightlike()
        assert not p.is_timelike()
        assert p.light_cone_type() == "lightlike"

    def test_spacelike(self):
        p = EMLPoint(0.0, math.exp(2.0))  # exp(0)=1, ln(e²)=2
        assert p.is_spacelike()
        assert not p.is_timelike()
        assert p.light_cone_type() == "spacelike"

    def test_future_light_cone_timelike(self):
        # exp(x) is always positive, so any timelike point is in future light cone
        p = EMLPoint(0.0, 1.0)
        assert p.future_light_cone()

    def test_light_cone_coordinates(self):
        p = EMLPoint(0.0, 1.0)  # t=1, x=0
        u, v = p.light_cone_coordinates()
        # u = t + x = 1+0 = 1, v = t - x = 1-0 = 1
        assert abs(u - 1.0) < 1e-12
        assert abs(v - 1.0) < 1e-12


class TestBoost:
    def test_zero_rapidity_is_identity(self):
        p = EMLPoint(0.0, 1.0)
        boosted = p.boost(0.0)
        assert abs(boosted.x - p.x) < 1e-10
        assert abs(boosted.y - p.y) < 1e-10

    def test_boost_preserves_minkowski_delta(self):
        p = EMLPoint(0.0, 1.0)
        original_dm = p.minkowski_delta()
        for phi in [0.1, 0.5, 1.0, -0.3, -0.8]:
            boosted = p.boost(phi)
            assert abs(boosted.minkowski_delta() - original_dm) < 1e-8, \
                f"Δ_M changed after boost(phi={phi})"

    def test_boost_roundtrip(self):
        p = EMLPoint(0.0, 1.0)
        p2 = p.boost(0.5).boost(-0.5)
        assert abs(p2.x - p.x) < 1e-8
        assert abs(p2.y - p.y) < 1e-8

    def test_boost_velocity_subluminal(self):
        p = EMLPoint(0.0, 1.0)
        boosted = p.boost_velocity(0.5)
        assert abs(boosted.minkowski_delta() - p.minkowski_delta()) < 1e-8

    def test_boost_velocity_superluminal_raises(self):
        p = EMLPoint(0.0, 1.0)
        with pytest.raises(ValueError):
            p.boost_velocity(1.0)
        with pytest.raises(ValueError):
            p.boost_velocity(1.5)

    def test_rapidity_zero_for_rest(self):
        # EMLPoint(0,1): pair=(1,0), rapidity=atanh(0/1)=0
        p = EMLPoint(0.0, 1.0)
        assert abs(p.rapidity() - 0.0) < 1e-12

    def test_rapidity_raises_for_spacelike(self):
        p = EMLPoint(0.0, math.exp(2.0))  # spacelike
        with pytest.raises(ValueError):
            p.rapidity()

    def test_rest_energy_equals_minkowski_delta(self):
        p = EMLPoint(0.0, 1.0)
        assert abs(p.rest_energy() - p.minkowski_delta()) < 1e-12

    def test_proper_time(self):
        p = EMLPoint(0.0, 1.0)
        assert abs(p.proper_time() - p.minkowski_delta()) < 1e-12  # c=1


class TestCanonicalFrame:
    def test_frame0_matches_pair(self):
        p = EMLPoint(1.0, math.e)
        f0 = p.canonical_frame(0)
        raw = p.pair()
        assert abs(f0.real_tension - raw.real_tension) < 1e-12
        assert abs(f0.imag_tension - raw.imag_tension) < 1e-12

    def test_four_frames_same_euclidean_delta(self):
        p = EMLPoint(1.0, 2.0)
        ref_delta = p.euclidean_delta()
        for k in range(4):
            frame = p.canonical_frame(k)
            frame_delta = math.sqrt(
                frame.real_tension ** 2 + frame.imag_tension ** 2
            )
            assert abs(frame_delta - ref_delta) < 1e-10, \
                f"Frame {k} has different delta: {frame_delta} vs {ref_delta}"

    def test_frame_cycle_mod4(self):
        p = EMLPoint(1.0, 2.0)
        assert p.canonical_frame(0) == p.canonical_frame(4)
        assert p.canonical_frame(1) == p.canonical_frame(5)

    def test_frame1_is_quarter_rotation(self):
        # Frame 1 multiplies by i: (r, im) → (-im, r)
        p = EMLPoint(1.0, math.e)  # pair=(e, 1)
        f1 = p.canonical_frame(1)
        pair = p.pair()
        assert abs(f1.real_tension - (-pair.imag_tension)) < 1e-10
        assert abs(f1.imag_tension - pair.real_tension) < 1e-10


class TestEMLPairFrames:
    def test_returns_four_frames(self):
        p = EMLPoint(1.0, 2.0).pair()
        frames = p.frames()
        assert len(frames) == 4
        assert all(isinstance(f, EMLPair) for f in frames)

    def test_all_frames_same_modulus(self):
        p = EMLPoint(1.0, 2.0).pair()
        ref_mod = p.modulus
        for i, f in enumerate(p.frames()):
            assert abs(f.modulus - ref_mod) < 1e-10, \
                f"Frame {i} modulus {f.modulus} != {ref_mod}"

    def test_frame0_is_identity(self):
        p = EMLPoint(1.0, 2.0).pair()
        f0 = p.frames()[0]
        assert abs(f0.real_tension - p.real_tension) < 1e-12
        assert abs(f0.imag_tension - p.imag_tension) < 1e-12


# ── new expanded tests ────────────────────────────────────────────────────────

class TestMinkowskiBoostInvariance:
    @pytest.mark.parametrize("phi", [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0])
    def test_minkowski_delta_conserved(self, phi):
        p = EMLPoint(1.0, math.e)
        dm0 = p.minkowski_delta()
        boosted = p.boost(phi)
        assert abs(boosted.minkowski_delta() - dm0) < 1e-8, \
            f"Δ_M changed after boost(phi={phi}): {boosted.minkowski_delta()} vs {dm0}"

    @pytest.mark.parametrize("phi", [-0.5, -0.25, 0.0, 0.25, 0.5])
    def test_spacelike_point_invariant(self, phi):
        # EMLPoint(0, exp(2)): spacelike — use small rapidities to stay within clamping guard
        p = EMLPoint(0.0, math.exp(2.0))
        dm0 = p.minkowski_delta()
        boosted = p.boost(phi)
        assert abs(boosted.minkowski_delta() - dm0) < 1e-8


class TestMinkowskiDeltaScale:
    @pytest.mark.parametrize("x", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("y", [0.1, 1.0, math.e, 10.0, 100.0])
    def test_nonnegative_always(self, x, y):
        p = EMLPoint(x, y)
        dm = p.minkowski_delta()
        assert dm >= 0.0

    @pytest.mark.parametrize("x", [-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0])
    @pytest.mark.parametrize("y", [0.1, 1.0, math.e, 10.0, 100.0])
    def test_finite_always(self, x, y):
        p = EMLPoint(x, y)
        assert math.isfinite(p.minkowski_delta())


class TestLightConeEdgeCases:
    def test_exact_lightlike_unit(self):
        # EMLPoint(0, e): exp(0)=1, ln(e)=1, ds²=0
        p = EMLPoint(0.0, math.e)
        assert p.is_lightlike()
        assert p.light_cone_type() == "lightlike"

    def test_exact_lightlike_nontrivial(self):
        # exp(x) = ln(y) when y = exp(exp(x))
        x = 1.5
        y = math.exp(math.exp(x))
        p = EMLPoint(x, y)
        assert p.is_lightlike(tol=1e-6)

    def test_near_lightlike_tol_tight(self):
        # With very tight tol, the near-lightlike point is NOT lightlike
        p = EMLPoint(0.0, math.e * 1.001)
        assert not p.is_lightlike(tol=1e-12)

    def test_near_lightlike_tol_loose(self):
        # With loose tol it may classify as lightlike
        p = EMLPoint(0.0, math.e * (1.0 + 1e-7))
        assert p.is_lightlike(tol=1.0)

    def test_light_cone_type_timelike_string(self):
        p = EMLPoint(0.0, 1.0)
        assert p.light_cone_type() == "timelike"

    def test_light_cone_type_spacelike_string(self):
        p = EMLPoint(0.0, math.exp(2.0))
        assert p.light_cone_type() == "spacelike"

    def test_lightlike_minkowski_delta_near_zero(self):
        p = EMLPoint(0.0, math.e)
        assert p.minkowski_delta() < 1e-9


class TestBoostLargeRapidity:
    def test_large_positive_rapidity_finite(self):
        p = EMLPoint(1.0, math.e)
        boosted = p.boost(5.0)
        assert math.isfinite(boosted.x)
        assert math.isfinite(boosted.y)

    def test_large_negative_rapidity_finite(self):
        p = EMLPoint(1.0, math.e)
        boosted = p.boost(-5.0)
        assert math.isfinite(boosted.x)
        assert math.isfinite(boosted.y)

    def test_large_positive_rapidity_invariant(self):
        p = EMLPoint(1.0, math.e)
        dm0 = p.minkowski_delta()
        boosted = p.boost(5.0)
        assert abs(boosted.minkowski_delta() - dm0) < 1e-6

    def test_large_negative_rapidity_invariant(self):
        p = EMLPoint(1.0, math.e)
        dm0 = p.minkowski_delta()
        boosted = p.boost(-5.0)
        assert abs(boosted.minkowski_delta() - dm0) < 1e-6
