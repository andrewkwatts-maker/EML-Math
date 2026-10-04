//! `EMLPoint` -- the universal EML computation node, `eml(x, y) = exp(x) - ln(y)`.
//!
//! Every method here routes its domain guards through [`crate::guard`]. Five of
//! them previously inlined `y.abs().max(1e-300)`, which floors *every*
//! subnormal rather than only zero and so disagreed with the Python reference
//! (`eml_spectral.spacetime._xy_safe`) for inputs such as `y = -1e-320`.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::guard::{eml, x_guard, y_guard, OVERFLOW_THRESHOLD};
use crate::pair::EMLPair;

#[cfg_attr(feature = "python", pyclass(module = "eml_core"))]
#[derive(Clone, Debug)]
pub struct EMLPoint {
    pub x: f64,
    pub y: f64,
}

impl EMLPoint {
    pub fn new(x: f64, y: f64) -> Self {
        EMLPoint { x, y }
    }

    /// `eml(x, y) = exp(x) - ln(y)`, the EML Sheffer operator.
    pub fn tension(&self) -> f64 {
        debug_assert!(
            !self.x.is_nan(),
            "tension is undefined for a NaN x coordinate"
        );
        debug_assert!(
            !self.y.is_nan(),
            "tension is undefined for a NaN y coordinate"
        );
        eml(self.x, self.y)
    }

    /// Alias for [`EMLPoint::tension`].
    pub fn eml(&self) -> f64 {
        self.tension()
    }

    /// One EML iteration step: `x' = y_safe`, `y' = eml(x, y)`.
    pub fn mirror_pulse(&self) -> EMLPoint {
        debug_assert!(
            !self.x.is_nan(),
            "mirror_pulse is undefined for a NaN x coordinate"
        );
        debug_assert!(
            !self.y.is_nan(),
            "mirror_pulse is undefined for a NaN y coordinate"
        );
        let y_safe = y_guard(self.y);
        EMLPoint {
            x: y_safe,
            y: x_guard(self.x).exp() - y_safe.ln(),
        }
    }

    /// Alias for [`EMLPoint::mirror_pulse`].
    pub fn pulse(&self) -> EMLPoint {
        self.mirror_pulse()
    }

    /// Historical alias for [`EMLPoint::mirror_pulse`].
    ///
    /// This used `.max(1e-300)` while `mirror_pulse` used the zero-only floor,
    /// so the two aliases returned different points for subnormal `y`. They now
    /// agree, and both agree with Python.
    pub fn frame_shift(&self) -> EMLPoint {
        self.mirror_pulse()
    }

    /// True when `x` exceeds the point at which `exp(x)` would overflow.
    pub fn is_slipping(&self) -> bool {
        debug_assert!(
            OVERFLOW_THRESHOLD.exp().is_finite(),
            "the threshold must not overflow"
        );
        debug_assert!(!self.x.is_nan(), "a NaN x has no slipping state");
        self.x > OVERFLOW_THRESHOLD
    }

    /// Whether tension is conserved across the step into `next`, within `tol`.
    pub fn conserves_tension(&self, next: &EMLPoint, tol: f64) -> bool {
        debug_assert!(tol >= 0.0, "a negative tolerance can never be satisfied");
        debug_assert!(
            !tol.is_nan(),
            "a NaN tolerance makes the comparison meaningless"
        );
        let exp_x = x_guard(self.x).exp();
        if !exp_x.is_finite() {
            return true;
        }
        let ln_y = y_guard(next.y).ln();
        let t = exp_x - ln_y;
        (t + ln_y - exp_x).abs() < tol
    }

    pub fn __repr__(&self) -> String {
        debug_assert!(!self.x.is_nan(), "repr of a NaN point is not meaningful");
        debug_assert!(!self.y.is_nan(), "repr of a NaN point is not meaningful");
        format!("EMLPoint(x={:.6}, y={:.6})", self.x, self.y)
    }

    /// Canonical frame coordinates `(exp(x), ln(y))` as an [`EMLPair`].
    pub fn pair(&self) -> EMLPair {
        debug_assert!(!self.x.is_nan(), "pair is undefined for a NaN x coordinate");
        debug_assert!(!self.y.is_nan(), "pair is undefined for a NaN y coordinate");
        EMLPair {
            real: x_guard(self.x).exp(),
            imag: y_guard(self.y).ln(),
        }
    }

    /// Euclidean delta `sqrt(exp(2x) + (ln y)^2)`.
    pub fn euclidean_delta(&self) -> f64 {
        debug_assert!(!self.x.is_nan(), "euclidean_delta is undefined for a NaN x");
        debug_assert!(!self.y.is_nan(), "euclidean_delta is undefined for a NaN y");
        let ex = x_guard(self.x).exp();
        let ly = y_guard(self.y).ln();
        (ex * ex + ly * ly).sqrt()
    }

    /// Minkowski interval `sqrt(|exp(2x) - (c ln y)^2|)`.
    ///
    /// `plus_signature = true` means the (+---) convention: time-like when
    /// `exp(2x) > (c ln y)^2`.
    pub fn minkowski_delta(&self, plus_signature: bool, c: f64) -> f64 {
        debug_assert!(c.is_finite(), "the light speed parameter must be finite");
        debug_assert!(!self.y.is_nan(), "minkowski_delta is undefined for a NaN y");
        let t = x_guard(self.x).exp();
        let s = c * y_guard(self.y).ln();
        let ds2 = if plus_signature {
            t * t - s * s
        } else {
            s * s - t * t
        };
        ds2.abs().sqrt()
    }

    /// Lorentz boost by rapidity `phi`, preserving the Minkowski interval.
    ///
    /// The clamp on the boosted spatial component keeps `exp` in range; without
    /// it a large rapidity yields an infinite `y` coordinate.
    pub fn boost(&self, phi: f64, c: f64) -> EMLPoint {
        debug_assert!(
            c != 0.0,
            "dividing by the light speed parameter requires c != 0"
        );
        debug_assert!(phi.is_finite(), "rapidity must be finite for sinh and cosh");
        let t = x_guard(self.x).exp();
        let s = y_guard(self.y).ln();
        let (sh, ch) = (phi.sinh(), phi.cosh());
        let t_new = (t * ch - (s / c) * sh).max(1e-300);
        let s_new = (s * ch - t * c * sh).clamp(-709.0, 709.0);
        debug_assert!(
            t_new > 0.0,
            "the boosted time component must stay in the ln domain"
        );
        EMLPoint {
            x: t_new.ln(),
            y: s_new.exp(),
        }
    }
}

// PyO3's inner attributes (`#[new]`, `#[getter]`, `#[pyo3(get)]`) are consumed
// by the `pymethods` / `pyclass` macros as syntax -- they are not real
// attributes, so they cannot be produced by `cfg_attr`. The methods above are
// therefore plain Rust, always compiled, and this block exposes them to Python
// when the feature is on.
#[cfg(feature = "python")]
#[pymethods]
impl EMLPoint {
    #[new]
    fn py_new(x: f64, y: f64) -> Self {
        Self::new(x, y)
    }

    #[getter]
    fn get_x(&self) -> f64 {
        self.x
    }

    #[getter]
    fn get_y(&self) -> f64 {
        self.y
    }

    #[pyo3(name = "tension")]
    fn py_tension(&self) -> f64 {
        self.tension()
    }

    #[pyo3(name = "eml")]
    fn py_eml(&self) -> f64 {
        self.eml()
    }

    #[pyo3(name = "mirror_pulse")]
    fn py_mirror_pulse(&self) -> Self {
        self.mirror_pulse()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tension_matches_the_defining_identity() {
        assert!((EMLPoint::new(1.0, 1.0).tension() - std::f64::consts::E).abs() < 1e-12);
        assert!((EMLPoint::new(0.0, 1.0).tension() - 1.0).abs() < 1e-12);
        assert!((EMLPoint::new(2.0, 1.0).tension() - (2.0_f64).exp()).abs() < 1e-12);
    }

    #[test]
    fn frame_shift_and_mirror_pulse_now_agree_on_subnormals() {
        let p = EMLPoint::new(0.0, -1e-320);
        assert_eq!(p.frame_shift().x, p.mirror_pulse().x);
        // The zero-only floor keeps the subnormal; `.max(1e-300)` returned 1e-300.
        assert_eq!(p.mirror_pulse().x, 1e-320);
    }

    #[test]
    fn threshold_is_the_exact_ln_of_f64_max() {
        // 709.7813 sits inside the window where the old rounded literal 709.78
        // dampened and Python did not.
        assert!(!EMLPoint::new(709.7813, 1.0).is_slipping());
        assert!(EMLPoint::new(709.79, 1.0).is_slipping());
    }

    #[test]
    fn boost_preserves_the_minkowski_interval() {
        let p = EMLPoint::new(1.0, 2.0);
        let before = p.minkowski_delta(true, 1.0);
        let after = p.boost(0.5, 1.0).minkowski_delta(true, 1.0);
        assert!(
            (before - after).abs() < 1e-9 * before.max(1.0),
            "interval drifted: {before} -> {after}"
        );
    }

    #[test]
    fn pair_and_euclidean_delta_agree() {
        let p = EMLPoint::new(0.5, 3.0);
        let q = p.pair();
        let expected = (q.real * q.real + q.imag * q.imag).sqrt();
        assert!((p.euclidean_delta() - expected).abs() < 1e-12);
    }

    #[test]
    fn conserves_tension_is_true_for_an_exact_step() {
        let p = EMLPoint::new(1.0, 2.0);
        assert!(p.conserves_tension(&p.mirror_pulse(), 1e-9));
    }
}
