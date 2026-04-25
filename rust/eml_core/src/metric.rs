use pyo3::prelude::*;
use rayon::prelude::*;

const OVERFLOW_THRESHOLD: f64 = 709.78;

/// Schwarzschild Christoffel symbol Γ^lam_{mu nu} at radial coordinate r.
/// Index convention: upper index lam (contravariant), lower indices mu, nu.
/// Metric signature: (-,+) — g_tt = -(1-rs/r) < 0, g_rr = 1/(1-rs/r) > 0.
/// Only the non-zero symbols in the 2D (t, r) slice are returned.
/// Returns 0.0 for r <= rs or unrecognised index combinations.
pub fn schwarzschild_christoffel(lam: usize, mu: usize, nu: usize, r: f64, rs: f64) -> f64 {
    if r <= rs || r <= 0.0 {
        return 0.0;
    }
    match (lam, mu, nu) {
        (0, 0, 1) | (0, 1, 0) => rs / (2.0 * r * (r - rs)),
        (1, 0, 0) => rs * (1.0 - rs / r) / (2.0 * r * r),
        (1, 1, 1) => -rs / (2.0 * r * (r - rs)),
        _ => 0.0,
    }
}

/// Batch Schwarzschild Christoffel evaluation.
/// points: Vec of (x, _) where r = exp(x).
/// Returns Vec of Γ^lam_{mu nu} values for each point.
#[pyfunction]
pub fn christoffel_batch_n(
    points: Vec<(f64, f64)>,
    lam: usize,
    mu: usize,
    nu: usize,
    rs: f64,
) -> Vec<f64> {
    points
        .par_iter()
        .map(|(x, _y)| {
            let xv = if *x > OVERFLOW_THRESHOLD { x.ln() } else { *x };
            let r = xv.exp().max(rs + 1e-9);
            schwarzschild_christoffel(lam, mu, nu, r, rs)
        })
        .collect()
}
