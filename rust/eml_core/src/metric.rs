//! Schwarzschild Christoffel symbols in the EML `(t, r)` slice.
//!
//! This module carried its own `OVERFLOW_THRESHOLD = 709.78` -- the rounded-down
//! literal that `point.rs` had already been corrected away from. For `x` in
//! `(709.78, 709.782712893384]` it therefore dampened `x` to `ln(x)` where
//! Python did not, so `r = exp(x)` came out around 6.6 instead of ~1.8e308 and
//! the returned symbol was wrong by roughly 300 orders of magnitude. It now
//! shares the single definition in [`crate::guard`].

use rayon::prelude::*;

use crate::guard::{check_len, x_guard, InputError, MAX_BATCH_LEN};

/// Number of coordinates in the `(t, r)` slice this module covers.
pub const SLICE_DIM: usize = 2;

/// Schwarzschild Christoffel symbol `Gamma^lam_{mu nu}` at radius `r`.
///
/// Index convention: upper index `lam`, lower indices `mu` and `nu`. Signature
/// `(-, +)`, so `g_tt = -(1 - rs/r) < 0` and `g_rr = 1/(1 - rs/r) > 0`. Returns
/// `0.0` inside the horizon and for index triples that are zero by symmetry.
pub fn schwarzschild_christoffel(lam: usize, mu: usize, nu: usize, r: f64, rs: f64) -> f64 {
    debug_assert!(!r.is_nan(), "the radial coordinate must not be NaN");
    debug_assert!(!rs.is_nan(), "the Schwarzschild radius must not be NaN");
    if r <= rs || r <= 0.0 {
        return 0.0;
    }
    debug_assert!(r > rs, "the branches below divide by (r - rs)");
    match (lam, mu, nu) {
        (0, 0, 1) | (0, 1, 0) => rs / (2.0 * r * (r - rs)),
        (1, 0, 0) => rs * (1.0 - rs / r) / (2.0 * r * r),
        (1, 1, 1) => -rs / (2.0 * r * (r - rs)),
        _ => 0.0,
    }
}

/// Batch Schwarzschild Christoffel evaluation over points whose `r = exp(x)`.
///
/// Out-of-range indices are rejected rather than silently returning `0.0`: a
/// typo in an index used to look exactly like a physically vanishing symbol.
pub fn christoffel_batch_n(
    points: Vec<(f64, f64)>,
    lam: usize,
    mu: usize,
    nu: usize,
    rs: f64,
) -> Result<Vec<f64>, InputError> {
    check_len(points.len(), MAX_BATCH_LEN)?;
    if lam >= SLICE_DIM || mu >= SLICE_DIM || nu >= SLICE_DIM {
        return Err(InputError::Invalid {
            what: "lam, mu and nu must each be 0 or 1 in the (t, r) slice",
        });
    }
    if !rs.is_finite() || rs < 0.0 {
        return Err(InputError::Invalid {
            what: "rs must be finite and non-negative",
        });
    }
    debug_assert!(
        points.len() <= MAX_BATCH_LEN,
        "check_len enforces the batch cap"
    );
    debug_assert!(
        lam < SLICE_DIM && mu < SLICE_DIM && nu < SLICE_DIM,
        "indices validated above"
    );

    Ok(points
        .par_iter()
        .map(|(x, _y)| {
            // Nudge off the horizon so the (r - rs) denominators stay finite.
            let r = x_guard(*x).exp().max(rs + 1e-9);
            schwarzschild_christoffel(lam, mu, nu, r, rs)
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbols_vanish_at_and_inside_the_horizon() {
        assert_eq!(schwarzschild_christoffel(0, 0, 1, 1.0, 2.0), 0.0);
        assert_eq!(schwarzschild_christoffel(0, 0, 1, 2.0, 2.0), 0.0);
        assert!(schwarzschild_christoffel(0, 0, 1, 4.0, 2.0) > 0.0);
    }

    #[test]
    fn the_symbol_is_symmetric_in_its_lower_indices() {
        let a = schwarzschild_christoffel(0, 0, 1, 5.0, 2.0);
        let b = schwarzschild_christoffel(0, 1, 0, 5.0, 2.0);
        assert_eq!(a, b);
    }

    #[test]
    fn threshold_now_matches_the_python_reference() {
        // Under the old 709.78 literal this x was dampened to ln(x) = 6.565,
        // giving r = 709.78 instead of ~1.8e308.
        let out = christoffel_batch_n(vec![(709.7813, 1.0)], 0, 0, 1, 2.0).unwrap();
        assert!(
            out[0] < 1e-300,
            "r = exp(709.7813) is enormous, so the symbol must underflow: {}",
            out[0]
        );
    }

    #[test]
    fn out_of_range_indices_are_an_error_not_a_zero() {
        assert!(christoffel_batch_n(vec![(1.0, 1.0)], 2, 0, 0, 2.0).is_err());
        assert!(christoffel_batch_n(vec![(1.0, 1.0)], 0, 0, 0, f64::NAN).is_err());
        assert!(christoffel_batch_n(vec![(1.0, 1.0)], 0, 0, 1, 2.0).is_ok());
    }
}
