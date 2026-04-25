use pyo3::prelude::*;
use crate::pair::EMLPair;

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

    /// Returns (exp(x), ln(|y|)) as an EMLPair — canonical frame coordinates.
    pub fn pair(&self) -> EMLPair {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 { self.y.abs().max(1e-300) } else { self.y };
        EMLPair { real: xv.exp(), imag: y_safe.ln() }
    }

    /// Euclidean delta: sqrt(exp(2x) + (ln y)^2).
    pub fn euclidean_delta(&self) -> f64 {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 { self.y.abs().max(1e-300) } else { self.y };
        let ex = xv.exp();
        let ly = y_safe.ln();
        (ex * ex + ly * ly).sqrt()
    }

    /// Minkowski interval sqrt(|exp(2x) - (c*ln y)^2|).
    /// plus_signature=true means (+---): time-like when exp(2x) > (c*ln y)^2.
    pub fn minkowski_delta(&self, plus_signature: bool, c: f64) -> f64 {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 { self.y.abs().max(1e-300) } else { self.y };
        let t = xv.exp();
        let s = c * y_safe.ln();
        let ds2 = if plus_signature { t * t - s * s } else { s * s - t * t };
        ds2.abs().sqrt()
    }

    /// Lorentz boost by rapidity phi. Returns new EMLPoint with identical Δ_M.
    /// Boost: t' = t*cosh(phi) - (x/c)*sinh(phi), x' = x*cosh(phi) - t*c*sinh(phi)
    /// where t = exp(x_coord), x = ln(y_coord).
    pub fn boost(&self, phi: f64, c: f64) -> EMLPoint {
        let xv = if self.x > OVERFLOW_THRESHOLD { self.x.ln() } else { self.x };
        let y_safe = if self.y <= 0.0 { self.y.abs().max(1e-300) } else { self.y };
        let t = xv.exp();
        let s = y_safe.ln();
        let (sh, ch) = (phi.sinh(), phi.cosh());
        let t_new = (t * ch - (s / c) * sh).max(1e-300);
        let s_new = (s * ch - t * c * sh).clamp(-709.0, 709.0);
        EMLPoint {
            x: t_new.ln(),
            y: s_new.exp(),
        }
    }
}
