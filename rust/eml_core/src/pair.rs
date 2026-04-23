use pyo3::prelude::*;

#[pyclass(module = "eml_core")]
#[derive(Clone, Debug)]
pub struct EMLPair {
    #[pyo3(get)]
    pub real: f64,
    #[pyo3(get)]
    pub imag: f64,
}

#[pymethods]
impl EMLPair {
    #[new]
    pub fn new(real: f64, imag: f64) -> Self {
        EMLPair { real, imag }
    }

    #[staticmethod]
    pub fn from_values(real: f64, imag: f64) -> Self {
        EMLPair { real, imag }
    }

    #[staticmethod]
    pub fn from_polar(r: f64, theta: f64) -> Self {
        EMLPair { real: r * theta.cos(), imag: r * theta.sin() }
    }

    #[staticmethod]
    pub fn unit_i() -> Self {
        EMLPair { real: 0.0, imag: 1.0 }
    }

    #[staticmethod]
    pub fn one() -> Self {
        EMLPair { real: 1.0, imag: 0.0 }
    }

    #[staticmethod]
    pub fn zero() -> Self {
        EMLPair { real: 0.0, imag: 0.0 }
    }

    #[getter]
    pub fn real_tension(&self) -> f64 {
        self.real
    }

    #[getter]
    pub fn imag_tension(&self) -> f64 {
        self.imag
    }

    #[getter]
    pub fn modulus(&self) -> f64 {
        (self.real * self.real + self.imag * self.imag).sqrt()
    }

    #[getter]
    pub fn argument(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    pub fn rotate_phase(&self, angle: f64) -> EMLPair {
        let (s, c) = angle.sin_cos();
        EMLPair {
            real: self.real * c - self.imag * s,
            imag: self.real * s + self.imag * c,
        }
    }

    pub fn conjugate(&self) -> EMLPair {
        EMLPair { real: self.real, imag: -self.imag }
    }

    pub fn __add__(&self, other: &EMLPair) -> EMLPair {
        EMLPair { real: self.real + other.real, imag: self.imag + other.imag }
    }

    pub fn __sub__(&self, other: &EMLPair) -> EMLPair {
        EMLPair { real: self.real - other.real, imag: self.imag - other.imag }
    }

    pub fn __mul__(&self, other: &EMLPair) -> EMLPair {
        EMLPair {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    pub fn __truediv__(&self, other: &EMLPair) -> EMLPair {
        let denom = other.real * other.real + other.imag * other.imag;
        let denom = if denom.abs() < 1e-300 { 1e-300 } else { denom };
        EMLPair {
            real: (self.real * other.real + self.imag * other.imag) / denom,
            imag: (self.imag * other.real - self.real * other.imag) / denom,
        }
    }

    pub fn __abs__(&self) -> f64 {
        self.modulus()
    }

    pub fn __eq__(&self, other: &EMLPair) -> bool {
        (self.real - other.real).abs() < 1e-9 && (self.imag - other.imag).abs() < 1e-9
    }

    pub fn __repr__(&self) -> String {
        let sign = if self.imag >= 0.0 { "+" } else { "-" };
        format!("EMLPair({:.6} {} {:.6}i)", self.real, sign, self.imag.abs())
    }
}

/// Batch Schrödinger potential-phase step: ψ_n → e^{-iV_n·dt/ħ}·ψ_n for all n.
/// Returns list of (real, imag) tuples.
#[pyfunction]
pub fn schrodinger_step_n(
    psi: Vec<(f64, f64)>,
    v: Vec<f64>,
    dt: f64,
    hbar: f64,
) -> Vec<(f64, f64)> {
    psi.iter()
        .zip(v.iter())
        .map(|((r, i), vn)| {
            let angle = -vn * dt / hbar;
            let (s, c) = angle.sin_cos();
            (r * c - i * s, r * s + i * c)
        })
        .collect()
}

/// Batch rotate_phase for a list of EMLPairs.
#[pyfunction]
pub fn rotate_phase_n(pairs: Vec<(f64, f64)>, angles: Vec<f64>) -> Vec<(f64, f64)> {
    pairs
        .iter()
        .zip(angles.iter())
        .map(|((r, im), &angle)| {
            let (s, c) = angle.sin_cos();
            (r * c - im * s, r * s + im * c)
        })
        .collect()
}
