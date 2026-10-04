//! `EMLPair` -- a two-real replacement for complex numbers, plus the batch
//! wrappers over it.
//!
//! The two batch functions used `zip`, which truncates to the shorter input.
//! `schrodinger_step_n(psi, v)` with a short potential array therefore returned
//! a short state vector rather than raising, which is a wrong answer dressed as
//! a correct one. Both now reject a length mismatch.

#[cfg(feature = "python")]
use pyo3::prelude::*;

use crate::guard::{check_pair, InputError, MAX_BATCH_LEN};

#[cfg_attr(feature = "python", pyclass(module = "eml_core"))]
#[derive(Clone, Debug)]
pub struct EMLPair {
    pub real: f64,
    pub imag: f64,
}

impl EMLPair {
    pub fn new(real: f64, imag: f64) -> Self {
        EMLPair { real, imag }
    }
    pub fn from_values(real: f64, imag: f64) -> Self {
        EMLPair { real, imag }
    }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        EMLPair {
            real: r * theta.cos(),
            imag: r * theta.sin(),
        }
    }
    pub fn unit_i() -> Self {
        EMLPair {
            real: 0.0,
            imag: 1.0,
        }
    }
    pub fn one() -> Self {
        EMLPair {
            real: 1.0,
            imag: 0.0,
        }
    }
    pub fn zero() -> Self {
        EMLPair {
            real: 0.0,
            imag: 0.0,
        }
    }
    pub fn real_tension(&self) -> f64 {
        self.real
    }
    pub fn imag_tension(&self) -> f64 {
        self.imag
    }
    pub fn modulus(&self) -> f64 {
        debug_assert!(
            !self.real.is_nan(),
            "modulus of a NaN real part is undefined"
        );
        debug_assert!(
            !self.imag.is_nan(),
            "modulus of a NaN imaginary part is undefined"
        );
        (self.real * self.real + self.imag * self.imag).sqrt()
    }
    pub fn argument(&self) -> f64 {
        self.imag.atan2(self.real)
    }

    /// Rotate by `angle` radians in the (real, imag) plane.
    pub fn rotate_phase(&self, angle: f64) -> EMLPair {
        debug_assert!(
            angle.is_finite(),
            "sin_cos of a non-finite angle yields NaN"
        );
        debug_assert!(
            self.modulus().is_finite(),
            "rotating an infinite pair is undefined"
        );
        let (s, c) = angle.sin_cos();
        EMLPair {
            real: self.real * c - self.imag * s,
            imag: self.real * s + self.imag * c,
        }
    }

    pub fn conjugate(&self) -> EMLPair {
        EMLPair {
            real: self.real,
            imag: -self.imag,
        }
    }

    pub fn __add__(&self, other: &EMLPair) -> EMLPair {
        EMLPair {
            real: self.real + other.real,
            imag: self.imag + other.imag,
        }
    }

    pub fn __sub__(&self, other: &EMLPair) -> EMLPair {
        EMLPair {
            real: self.real - other.real,
            imag: self.imag - other.imag,
        }
    }

    pub fn __mul__(&self, other: &EMLPair) -> EMLPair {
        EMLPair {
            real: self.real * other.real - self.imag * other.imag,
            imag: self.real * other.imag + self.imag * other.real,
        }
    }

    pub fn __truediv__(&self, other: &EMLPair) -> EMLPair {
        debug_assert!(
            !other.real.is_nan(),
            "division by a NaN real part is undefined"
        );
        debug_assert!(
            !other.imag.is_nan(),
            "division by a NaN imaginary part is undefined"
        );
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

// PyO3's inner attributes are consumed by the `pymethods` / `pyclass` macros as
// syntax, not resolved as real attributes, so `cfg_attr` cannot produce them.
// The methods above are therefore plain Rust, always compiled; this block
// exposes them to Python when the feature is on.
#[cfg(feature = "python")]
#[pymethods]
impl EMLPair {
    #[new]
    fn py_new(real: f64, imag: f64) -> Self {
        Self::new(real, imag)
    }

    #[getter]
    fn get_real(&self) -> f64 {
        self.real
    }

    #[getter]
    fn get_imag(&self) -> f64 {
        self.imag
    }

    #[staticmethod]
    #[pyo3(name = "from_values")]
    fn py_from_values(real: f64, imag: f64) -> Self {
        Self::from_values(real, imag)
    }

    #[staticmethod]
    #[pyo3(name = "from_polar")]
    fn py_from_polar(r: f64, theta: f64) -> Self {
        Self::from_polar(r, theta)
    }

    #[staticmethod]
    #[pyo3(name = "unit_i")]
    fn py_unit_i() -> Self {
        Self::unit_i()
    }

    #[staticmethod]
    #[pyo3(name = "one")]
    fn py_one() -> Self {
        Self::one()
    }

    #[staticmethod]
    #[pyo3(name = "zero")]
    fn py_zero() -> Self {
        Self::zero()
    }

    #[getter]
    #[pyo3(name = "real_tension")]
    fn py_real_tension(&self) -> f64 {
        EMLPair::real_tension(self)
    }

    #[getter]
    #[pyo3(name = "imag_tension")]
    fn py_imag_tension(&self) -> f64 {
        EMLPair::imag_tension(self)
    }

    #[getter]
    #[pyo3(name = "modulus")]
    fn py_modulus(&self) -> f64 {
        EMLPair::modulus(self)
    }
}

/// Batch Schrodinger potential-phase step: `psi_n -> exp(-i V_n dt / hbar) psi_n`.
///
/// Returns one `(real, imag)` tuple per input state.
pub fn schrodinger_step_n(
    psi: Vec<(f64, f64)>,
    v: Vec<f64>,
    dt: f64,
    hbar: f64,
) -> Result<Vec<(f64, f64)>, InputError> {
    check_pair(psi.len(), v.len())?;
    if hbar == 0.0 || !hbar.is_finite() {
        return Err(crate::guard::InputError::Invalid {
            what: "hbar must be finite and non-zero",
        });
    }
    if !dt.is_finite() {
        return Err(crate::guard::InputError::Invalid {
            what: "dt must be finite",
        });
    }
    debug_assert_eq!(psi.len(), v.len(), "check_pair guarantees equal lengths");
    debug_assert!(
        psi.len() <= MAX_BATCH_LEN,
        "check_pair enforces the batch cap"
    );
    Ok(psi
        .iter()
        .zip(v.iter())
        .map(|((r, i), vn)| {
            let (s, c) = (-vn * dt / hbar).sin_cos();
            (r * c - i * s, r * s + i * c)
        })
        .collect())
}

/// Batch [`EMLPair::rotate_phase`], one angle per pair.
pub fn rotate_phase_n(
    pairs: Vec<(f64, f64)>,
    angles: Vec<f64>,
) -> Result<Vec<(f64, f64)>, InputError> {
    check_pair(pairs.len(), angles.len())?;
    debug_assert_eq!(
        pairs.len(),
        angles.len(),
        "check_pair guarantees equal lengths"
    );
    debug_assert!(
        pairs.len() <= MAX_BATCH_LEN,
        "check_pair enforces the batch cap"
    );
    Ok(pairs
        .iter()
        .zip(angles.iter())
        .map(|((r, im), &angle)| {
            let (s, c) = angle.sin_cos();
            (r * c - im * s, r * s + im * c)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn multiplication_matches_complex_arithmetic() {
        let i = EMLPair::unit_i();
        let minus_one = i.__mul__(&i);
        assert!((minus_one.real + 1.0).abs() < 1e-15);
        assert!(minus_one.imag.abs() < 1e-15);
    }

    #[test]
    fn division_inverts_multiplication() {
        let a = EMLPair::new(3.0, -4.0);
        let b = EMLPair::new(1.0, 2.0);
        let back = a.__mul__(&b).__truediv__(&b);
        assert!((back.real - a.real).abs() < 1e-9);
        assert!((back.imag - a.imag).abs() < 1e-9);
    }

    #[test]
    fn rotation_preserves_modulus() {
        let a = EMLPair::from_polar(2.5, 0.3);
        let r = a.rotate_phase(1.234);
        assert!((a.modulus() - r.modulus()).abs() < 1e-12);
    }

    #[test]
    fn batch_wrappers_reject_a_length_mismatch() {
        assert!(schrodinger_step_n(vec![(1.0, 0.0), (0.0, 1.0)], vec![0.5], 0.1, 1.0).is_err());
        assert!(rotate_phase_n(vec![(1.0, 0.0)], vec![0.1, 0.2]).is_err());
        assert!(schrodinger_step_n(vec![(1.0, 0.0)], vec![0.5], 0.1, 0.0).is_err());
    }

    #[test]
    fn batch_wrappers_accept_matched_input() {
        let out = rotate_phase_n(vec![(1.0, 0.0)], vec![std::f64::consts::FRAC_PI_2]).unwrap();
        assert_eq!(out.len(), 1);
        assert!(out[0].0.abs() < 1e-15);
        assert!((out[0].1 - 1.0).abs() < 1e-15);
    }
}
