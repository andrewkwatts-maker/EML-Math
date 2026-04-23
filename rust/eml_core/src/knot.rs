use pyo3::prelude::*;
use crate::point::EMLPoint;

const FLIP_YIELD: i64 = 2;
const PHASE_STEP: f64 = std::f64::consts::PI / 2.0;
const TWO_PI: f64 = std::f64::consts::PI * 2.0;

#[pyclass(module = "eml_core")]
#[derive(Clone, Debug)]
pub struct EMLKnot {
    #[pyo3(get)]
    pub point: EMLPoint,
    #[pyo3(get)]
    pub flip_count: i64,
    #[pyo3(get)]
    pub phase: f64,
}

#[pymethods]
impl EMLKnot {
    #[new]
    #[pyo3(signature = (point, n=0, theta=0.0))]
    pub fn new(point: EMLPoint, n: i64, theta: f64) -> Self {
        EMLKnot { point, flip_count: n, phase: theta % TWO_PI }
    }

    #[getter]
    pub fn rho(&self) -> f64 {
        self.point.tension().abs()
    }

    pub fn mirror_pulse(&self) -> EMLKnot {
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

    pub fn three_one_flip(&self) -> EMLKnot {
        let mut k = self.clone();
        for _ in 0..4 {
            k = k.mirror_pulse();
        }
        k
    }

    /// Alias for three_one_flip().
    pub fn flip(&self) -> EMLKnot {
        self.three_one_flip()
    }

    pub fn tread_yield(&self) -> i64 {
        (self.flip_count / 4) * FLIP_YIELD
    }

    pub fn __repr__(&self) -> String {
        format!(
            "EMLKnot(n={}, rho={:.6}, theta={:.4}, point={:?})",
            self.flip_count, self.rho(), self.phase, self.point
        )
    }
}

/// Simulate n_pulses of mirror_pulse, returning (x, y, tension) per step.
#[pyfunction]
pub fn simulate_pulses_n(x0: f64, y0: f64, n_pulses: usize) -> Vec<(f64, f64, f64)> {
    let mut results = Vec::with_capacity(n_pulses + 1);
    let mut p = EMLPoint::new(x0, y0);
    results.push((p.x, p.y, p.tension()));
    for _ in 0..n_pulses {
        p = p.mirror_pulse();
        results.push((p.x, p.y, p.tension()));
    }
    results
}
