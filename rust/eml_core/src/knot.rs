//! `EMLKnot` -- an `EMLPoint` carrying a flip count and a phase.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::guard::{check_len, InputError};
use crate::point::EMLPoint;

const FLIP_YIELD: i64 = 2;
const PHASE_STEP: f64 = std::f64::consts::PI / 2.0;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

#[cfg_attr(feature = "python", pyclass(module = "eml_core"))]
#[derive(Clone, Debug)]
pub struct EMLKnot {
    pub point: EMLPoint,
    pub flip_count: i64,
    pub phase: f64,
}

impl EMLKnot {
    pub fn new(point: EMLPoint, n: i64, theta: f64) -> Self {
        EMLKnot {
            point,
            flip_count: n,
            phase: theta % TWO_PI,
        }
    }

    /// Tension magnitude at this knot.
    pub fn rho(&self) -> f64 {
        debug_assert!(
            !self.point.x.is_nan(),
            "rho is undefined at a NaN x coordinate"
        );
        debug_assert!(
            !self.point.y.is_nan(),
            "rho is undefined at a NaN y coordinate"
        );
        self.point.tension().abs()
    }

    /// One pulse: advance the point, increment the flip count, step the phase.
    pub fn mirror_pulse(&self) -> EMLKnot {
        debug_assert!(
            self.flip_count >= 0,
            "the flip count is a non-negative tally"
        );
        debug_assert!(
            self.flip_count < i64::MAX,
            "the flip count must have headroom before it is incremented"
        );
        EMLKnot {
            point: self.point.mirror_pulse(),
            flip_count: self.flip_count + 1,
            phase: (self.phase + PHASE_STEP) % TWO_PI,
        }
    }

    /// Alias for mirror_pulse().
    pub fn pulse(&self) -> EMLKnot {
        self.mirror_pulse()
    }

    /// The 3_1 (trefoil) flip: four pulses, which returns the phase to its
    /// starting value modulo 2 pi.
    pub fn three_one_flip(&self) -> EMLKnot {
        const PULSES_PER_FLIP: usize = 4;
        debug_assert!(
            self.phase.is_finite(),
            "a non-finite phase cannot be stepped four times"
        );
        debug_assert!(
            self.flip_count <= i64::MAX - PULSES_PER_FLIP as i64,
            "four increments must not overflow the flip count"
        );
        let mut k = self.clone();
        for _ in 0..PULSES_PER_FLIP {
            k = k.mirror_pulse();
        }
        k
    }

    /// Alias for three_one_flip().
    pub fn flip(&self) -> EMLKnot {
        self.three_one_flip()
    }

    /// Yield accumulated over completed flips.
    pub fn tread_yield(&self) -> i64 {
        debug_assert!(
            self.flip_count >= 0,
            "the flip count is a non-negative tally"
        );
        debug_assert!(
            self.flip_count / 4 <= i64::MAX / FLIP_YIELD,
            "the yield multiplication must not overflow"
        );
        (self.flip_count / 4) * FLIP_YIELD
    }

    pub fn __repr__(&self) -> String {
        format!(
            "EMLKnot(n={}, rho={:.6}, theta={:.4}, point={:?})",
            self.flip_count,
            self.rho(),
            self.phase,
            self.point
        )
    }
}

// PyO3's inner attributes are consumed by the `pymethods` / `pyclass` macros as
// syntax, not resolved as real attributes, so `cfg_attr` cannot produce them.
// The methods above are therefore plain Rust, always compiled; this block
// exposes them to Python when the feature is on.
#[cfg(feature = "python")]
#[pymethods]
impl EMLKnot {
    #[new]
    #[pyo3(signature = (point, n=0, theta=0.0))]
    fn py_new(point: EMLPoint, n: i64, theta: f64) -> Self {
        Self::new(point, n, theta)
    }

    #[getter]
    fn get_point(&self) -> EMLPoint {
        self.point.clone()
    }

    #[getter]
    fn get_flip_count(&self) -> i64 {
        self.flip_count
    }

    #[getter]
    fn get_phase(&self) -> f64 {
        self.phase
    }

    #[getter]
    #[pyo3(name = "rho")]
    fn py_rho(&self) -> f64 {
        EMLKnot::rho(self)
    }

    #[pyo3(name = "mirror_pulse")]
    fn py_mirror_pulse(&self) -> EMLKnot {
        self.mirror_pulse()
    }

    #[pyo3(name = "pulse")]
    fn py_pulse(&self) -> EMLKnot {
        self.pulse()
    }

    #[pyo3(name = "three_one_flip")]
    fn py_three_one_flip(&self) -> EMLKnot {
        self.three_one_flip()
    }
}

/// Upper bound on `n_pulses`, so a caller cannot request an unbounded loop.
///
/// The result vector holds `n_pulses + 1` triples, so this caps one call at
/// roughly 24 MB.
pub const MAX_PULSES: usize = 1_048_576;

/// Simulate `n_pulses` iterations, returning `(x, y, tension)` per step
/// including the initial state.
pub fn simulate_pulses_n(
    x0: f64,
    y0: f64,
    n_pulses: usize,
) -> Result<Vec<(f64, f64, f64)>, InputError> {
    check_len(n_pulses, MAX_PULSES)?;
    if !x0.is_finite() || !y0.is_finite() {
        return Err(InputError::Invalid {
            what: "x0 and y0 must be finite",
        });
    }
    debug_assert!(
        n_pulses <= MAX_PULSES,
        "check_len bounds the iteration count"
    );
    debug_assert!(
        x0.is_finite() && y0.is_finite(),
        "the seed point was validated above"
    );

    let mut results = Vec::with_capacity(n_pulses + 1);
    let mut p = EMLPoint::new(x0, y0);
    results.push((p.x, p.y, p.tension()));
    for _ in 0..n_pulses {
        p = p.mirror_pulse();
        results.push((p.x, p.y, p.tension()));
    }
    debug_assert_eq!(
        results.len(),
        n_pulses + 1,
        "one row per step plus the seed"
    );
    Ok(results)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_pulses_return_the_phase_to_its_start() {
        let k = EMLKnot::new(EMLPoint::new(1.0, 2.0), 0, 0.0);
        let f = k.three_one_flip();
        assert_eq!(f.flip_count, 4);
        assert!(
            f.phase.abs() < 1e-12,
            "phase should close modulo 2 pi, got {}",
            f.phase
        );
    }

    #[test]
    fn tread_yield_counts_completed_flips_only() {
        let p = EMLPoint::new(0.0, 1.0);
        assert_eq!(EMLKnot::new(p.clone(), 3, 0.0).tread_yield(), 0);
        assert_eq!(EMLKnot::new(p.clone(), 4, 0.0).tread_yield(), FLIP_YIELD);
        assert_eq!(EMLKnot::new(p, 8, 0.0).tread_yield(), 2 * FLIP_YIELD);
    }

    #[test]
    fn simulate_pulses_returns_one_row_per_step_plus_the_seed() {
        let rows = simulate_pulses_n(0.0, 1.0, 5).unwrap();
        assert_eq!(rows.len(), 6);
        assert_eq!(rows[0], (0.0, 1.0, 1.0));
    }

    #[test]
    fn simulate_pulses_rejects_unbounded_and_non_finite_input() {
        assert!(simulate_pulses_n(0.0, 1.0, MAX_PULSES + 1).is_err());
        assert!(simulate_pulses_n(f64::NAN, 1.0, 1).is_err());
    }
}
