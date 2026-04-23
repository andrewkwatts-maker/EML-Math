use pyo3::prelude::*;

const OVERFLOW_THRESHOLD: f64 = 709.78;

#[pyclass(module = "eml_core")]
#[derive(Clone, Debug)]
pub struct EMLPoint {
    #[pyo3(get)]
    pub x: f64,
    #[pyo3(get)]
    pub y: f64,
}

#[pymethods]
impl EMLPoint {
    #[new]
    pub fn new(x: f64, y: f64) -> Self {
        EMLPoint { x, y }
    }

    pub fn tension(&self) -> f64 {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 {
            self.y.abs().max(1e-300)
        } else {
            self.y
        };
        xv.exp() - y_safe.ln()
    }

    /// Alias for tension() — eml(x, y) = exp(x) - ln(y).
    pub fn eml(&self) -> f64 {
        self.tension()
    }

    pub fn mirror_pulse(&self) -> EMLPoint {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 {
            self.y.abs().max(1e-300)
        } else {
            self.y
        };
        let t = xv.exp() - y_safe.ln();
        EMLPoint { x: y_safe, y: t }
    }

    /// Alias for mirror_pulse().
    pub fn pulse(&self) -> EMLPoint {
        self.mirror_pulse()
    }

    pub fn frame_shift(&self) -> EMLPoint {
        let y_safe = self.y.abs().max(1e-300);
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let t = xv.exp() - y_safe.ln();
        EMLPoint { x: y_safe, y: t }
    }

    pub fn is_slipping(&self) -> bool {
        self.x > OVERFLOW_THRESHOLD
    }

    pub fn conserves_tension(&self, next: &EMLPoint, tol: f64) -> bool {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let exp_x = xv.exp();
        if !exp_x.is_finite() {
            return true;
        }
        let y_safe = if next.y <= 0.0 { next.y.abs().max(1e-300) } else { next.y };
        let t = exp_x - y_safe.ln();
        (t + y_safe.ln() - exp_x).abs() < tol
    }

    pub fn __repr__(&self) -> String {
        format!("EMLPoint(x={:.6}, y={:.6})", self.x, self.y)
    }
}
