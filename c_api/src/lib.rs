//! C-compatible API for the EML Sheffer operator.
//!
//! Build as a static or shared library:
//!   cargo build --release -p eml_c_api
//!
//! Outputs (in target/release/):
//!   libeml_math.a       — static library (link with -leml_math)
//!   libeml_math.so      — shared library (Linux/macOS)
//!   eml_math.dll        — dynamic library (Windows)
//!
//! Include eml_math.h in your C/C++ project.

use std::os::raw::{c_double, c_int};

const OVERFLOW_THRESHOLD: f64 = 709.78;

// ── internal helpers ──────────────────────────────────────────────────────────

#[inline]
fn y_safe(y: f64) -> f64 {
    if y <= 0.0 { y.abs().max(1e-300) } else { y }
}

#[inline]
fn xv_safe(x: f64) -> f64 {
    if x > OVERFLOW_THRESHOLD { x.ln() } else { x }
}

// ── core EML operator ─────────────────────────────────────────────────────────

/// eml(x, y) = exp(x) - ln(y). Applies frame-shift guard (|y| when y ≤ 0).
#[no_mangle]
pub extern "C" fn eml_tension(x: c_double, y: c_double) -> c_double {
    let xv = xv_safe(x);
    xv.exp() - y_safe(y).ln()
}

// ── Mirror-Pulse iteration ────────────────────────────────────────────────────

/// One Mirror-Pulse step: (x, y) → (|y|, exp(x) − ln(|y|)).
/// Writes the new x and y values into *out_x and *out_y.
#[no_mangle]
pub unsafe extern "C" fn eml_mirror_pulse(
    x: c_double,
    y: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp() - ys.ln();
    *out_x = ys;
    *out_y = t;
}

/// Run n_pulses Mirror-Pulse steps from (x0, y0).
/// Caller must pre-allocate out_xs[n_pulses+1] and out_ys[n_pulses+1].
/// out_xs[0] = x0, out_ys[0] = y0; subsequent indices are the iterated states.
#[no_mangle]
pub unsafe extern "C" fn eml_simulate_pulses(
    x0: c_double,
    y0: c_double,
    n_pulses: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
) {
    *out_xs.add(0) = x0;
    *out_ys.add(0) = y0;
    let mut x = x0;
    let mut y = y0;
    for i in 1..=n_pulses {
        let xv = xv_safe(x);
        let ys = y_safe(y);
        let t = xv.exp() - ys.ln();
        x = ys;
        y = t;
        *out_xs.add(i) = x;
        *out_ys.add(i) = y;
    }
}

// ── Geometric invariants ──────────────────────────────────────────────────────

/// √(exp(2x) + (ln y)²) — Euclidean frame invariant.
#[no_mangle]
pub extern "C" fn eml_euclidean_delta(x: c_double, y: c_double) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let ex = xv.exp();
    let ly = ys.ln();
    (ex * ex + ly * ly).sqrt()
}

/// √|exp(2x) − (c·ln y)²| — Minkowski interval.
/// plus_signature=1 for (+−−−), 0 for (−+++) convention.
#[no_mangle]
pub extern "C" fn eml_minkowski_delta(
    x: c_double,
    y: c_double,
    plus_signature: c_int,
    c: c_double,
) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = c * ys.ln();
    let ds2 = if plus_signature != 0 {
        t * t - s * s
    } else {
        s * s - t * t
    };
    ds2.abs().sqrt()
}

/// Rapidity φ = atanh(ln(y) / exp(x)).
/// Returns f64::NAN if the point is not timelike (|ln y| ≥ |exp x|).
#[no_mangle]
pub extern "C" fn eml_rapidity(x: c_double, y: c_double) -> c_double {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = ys.ln();
    if t.abs() < 1e-300 { return f64::NAN; }
    let ratio = s / t;
    if ratio.abs() >= 1.0 { return f64::NAN; }
    ratio.atanh()
}

/// Causal classification: +1 = timelike, -1 = spacelike, 0 = lightlike.
/// tol is the tolerance for lightlike detection.
#[no_mangle]
pub extern "C" fn eml_causal_type(
    x: c_double,
    y: c_double,
    c: c_double,
    tol: c_double,
) -> c_int {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t2 = { let t = xv.exp(); t * t };
    let s2 = { let s = c * ys.ln(); s * s };
    if (t2 - s2).abs() < tol { 0 }
    else if t2 > s2 { 1 }
    else { -1 }
}

// ── Lorentz boost ─────────────────────────────────────────────────────────────

/// Apply a Lorentz boost by rapidity phi with speed of light c.
/// Writes the new EML coordinates into *out_x, *out_y.
#[no_mangle]
pub unsafe extern "C" fn eml_boost(
    x: c_double,
    y: c_double,
    phi: c_double,
    c: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
) {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp();
    let s = ys.ln();
    let sh = phi.sinh();
    let ch = phi.cosh();
    let t_new = (t * ch - (s / c) * sh).max(1e-300);
    let s_new = (s * ch - t * c * sh).clamp(-709.0, 709.0);
    *out_x = t_new.ln();
    *out_y = s_new.exp();
}

/// Batch boost. All arrays must have exactly n elements.
/// Applies eml_boost(xs[i], ys[i], phis[i], c) for each i.
#[no_mangle]
pub unsafe extern "C" fn eml_boost_batch(
    xs: *const c_double,
    ys: *const c_double,
    phis: *const c_double,
    c: c_double,
    n: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
) {
    for i in 0..n {
        eml_boost(*xs.add(i), *ys.add(i), *phis.add(i), c,
                  out_xs.add(i), out_ys.add(i));
    }
}

// ── Schwarzschild Christoffel symbols ────────────────────────────────────────

/// Analytic non-zero Γ^lam_{mu nu} for the Schwarzschild metric (2D slice).
/// Index convention: upper index lam (contravariant), lower indices mu, nu.
/// Metric signature: (-,+) — g_tt < 0, g_rr > 0.
/// r = radial coordinate, rs = Schwarzschild radius (r must be > rs).
#[no_mangle]
pub extern "C" fn eml_schwarzschild_christoffel(
    lam: usize,
    mu: usize,
    nu: usize,
    r: c_double,
    rs: c_double,
) -> c_double {
    if r <= rs || r <= 0.0 { return 0.0; }
    match (lam, mu, nu) {
        (0, 0, 1) | (0, 1, 0) => rs / (2.0 * r * (r - rs)),
        (1, 0, 0) => rs * (1.0 - rs / r) / (2.0 * r * r),
        (1, 1, 1) => -rs / (2.0 * r * (r - rs)),
        _ => 0.0,
    }
}

// ── Octonion multiplication ───────────────────────────────────────────────────

const FANO_LINES: [(usize, usize, usize); 7] =
    [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(1,5,6),(2,6,7),(1,3,7)];

const fn build_oct_table() -> [[(i8, usize); 8]; 8] {
    let mut t = [[(0i8, 0usize); 8]; 8];
    let mut i = 0;
    while i < 8 { t[0][i] = (1, i); t[i][0] = (1, i); i += 1; }
    let mut i = 1;
    while i < 8 { t[i][i] = (-1, 0); i += 1; }
    let mut li = 0;
    while li < 7 {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c); t[b][a] = (-1, c);
        t[b][c] = (1, a); t[c][b] = (-1, a);
        t[c][a] = (1, b); t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); 8]; 8] = build_oct_table();

/// Multiply two octonions. a and b point to 8-element f64 arrays.
/// Result is written to out (also 8 elements). Caller must allocate out.
#[no_mangle]
pub unsafe extern "C" fn eml_octonion_mul(
    a: *const c_double,
    b: *const c_double,
    out: *mut c_double,
) {
    let a = std::slice::from_raw_parts(a, 8);
    let b = std::slice::from_raw_parts(b, 8);
    let out = std::slice::from_raw_parts_mut(out, 8);
    for v in out.iter_mut() { *v = 0.0; }
    for i in 0..8 {
        if a[i] == 0.0 { continue; }
        for j in 0..8 {
            if b[j] == 0.0 { continue; }
            let (sign, k) = OCT_TABLE[i][j];
            out[k] += (sign as f64) * a[i] * b[j];
        }
    }
}
