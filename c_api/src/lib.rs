//! C-compatible API for the EML Sheffer operator library.
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

/// Largest element count a batch entry point is documented to accept.
///
/// The C ABI cannot see the caller's allocation, so this is a stated contract
/// backed by a debug assertion rather than an enforceable bound.
pub const MAX_C_BATCH: usize = 1_048_576;

/// `f64::MAX.ln()`.
///
/// This file carried the rounded-down literal `709.78`, so for `x` in
/// `(709.78, 709.782712893384]` the C API dampened `x` to `ln(x)` where both
/// Python and `eml_core` did not -- a divergence of order 1e308. Keep equal to
/// `eml_core::guard::OVERFLOW_THRESHOLD` and to Python's
/// `eml_math.constants.OVERFLOW_THRESHOLD`.
const OVERFLOW_THRESHOLD: f64 = 709.782712893384;

// ── internal helpers ──────────────────────────────────────────────────────────

/// Frame-shift guard: map `y` into the domain of `ln`.
///
/// The floor applies only to zero. The previous `y.abs().max(1e-300)` also
/// crushed legitimate subnormals such as `y = -1e-320` up to `1e-300`, which
/// shifts `ln(y)` by a factor of 46.
#[inline]
fn y_safe(y: f64) -> f64 {
    debug_assert!(
        !y.is_nan(),
        "y_safe received NaN; the caller must screen it"
    );
    let a = if y <= 0.0 { y.abs() } else { y };
    debug_assert!(
        a >= 0.0,
        "the frame shift must produce a non-negative magnitude"
    );
    if a == 0.0 {
        1e-300
    } else {
        a
    }
}

/// Slipping-Wheel dampening, so `exp(x)` stays finite.
#[inline]
fn xv_safe(x: f64) -> f64 {
    debug_assert!(
        !x.is_nan(),
        "xv_safe received NaN; the caller must screen it"
    );
    debug_assert!(
        x.is_finite() || x.is_infinite(),
        "a non-NaN float is either finite or infinite; NaN was excluded above"
    );
    if x > OVERFLOW_THRESHOLD {
        x.ln()
    } else {
        x
    }
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
/// # Safety
///
/// The caller guarantees that:
/// - `out_x` is non-null and correctly aligned for `c_double`.
/// - `out_y` is non-null and correctly aligned for `c_double`.
/// - the output pointers do not alias each other.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_mirror_pulse(
    x: c_double,
    y: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
) {
    debug_assert!(!out_x.is_null(), "out_x must not be null");
    debug_assert!(!out_y.is_null(), "out_y must not be null");
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t = xv.exp() - ys.ln();
    *out_x = ys;
    *out_y = t;
}

/// Run n_pulses Mirror-Pulse steps from (x0, y0).
/// Caller must pre-allocate out_xs[n_pulses+1] and out_ys[n_pulses+1].
/// out_xs[0] = x0, out_ys[0] = y0; subsequent indices are the iterated states.
/// # Safety
///
/// The caller guarantees that:
/// - `out_xs` is non-null and correctly aligned for `c_double`.
/// - `out_ys` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_simulate_pulses(
    x0: c_double,
    y0: c_double,
    n_pulses: usize,
    out_xs: *mut c_double,
    out_ys: *mut c_double,
) {
    debug_assert!(!out_xs.is_null(), "out_xs must not be null");
    debug_assert!(!out_ys.is_null(), "out_ys must not be null");
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

// ── Elementary arithmetic operators ──────────────────────────────────────────

/// exp(x) = eml(x, 1). Safe: caps x at overflow threshold.
#[no_mangle]
pub extern "C" fn eml_exp(x: c_double) -> c_double {
    xv_safe(x).exp()
}

/// ln(y). Safe: applies frame-shift guard for y ≤ 0.
#[no_mangle]
pub extern "C" fn eml_ln(y: c_double) -> c_double {
    y_safe(y).ln()
}

/// a + b (addition of tension values).
#[no_mangle]
pub extern "C" fn eml_add(a: c_double, b: c_double) -> c_double {
    a + b
}

/// a - b (subtraction of tension values).
#[no_mangle]
pub extern "C" fn eml_sub(a: c_double, b: c_double) -> c_double {
    a - b
}

/// a * b (multiplication of tension values).
#[no_mangle]
pub extern "C" fn eml_mul(a: c_double, b: c_double) -> c_double {
    a * b
}

/// a / b. Returns NaN when |b| < 1e-300.
#[no_mangle]
pub extern "C" fn eml_div(a: c_double, b: c_double) -> c_double {
    if b.abs() < 1e-300 {
        return f64::NAN;
    }
    a / b
}

/// sqrt(|x|) — square root with Axiom-8 absolute value guard.
#[no_mangle]
pub extern "C" fn eml_sqrt(x: c_double) -> c_double {
    x.abs().sqrt()
}

/// x² (squaring).
#[no_mangle]
pub extern "C" fn eml_sqr(x: c_double) -> c_double {
    x * x
}

/// |x|^exp — power function with absolute-value base guard.
#[no_mangle]
pub extern "C" fn eml_pow(base: c_double, exp: c_double) -> c_double {
    base.abs().powf(exp)
}

/// -x (negation).
#[no_mangle]
pub extern "C" fn eml_neg(x: c_double) -> c_double {
    -x
}

/// 1/x. Returns NaN when |x| < 1e-300.
#[no_mangle]
pub extern "C" fn eml_inv(x: c_double) -> c_double {
    if x.abs() < 1e-300 {
        return f64::NAN;
    }
    1.0 / x
}

/// x / 2.
#[no_mangle]
pub extern "C" fn eml_half(x: c_double) -> c_double {
    x * 0.5
}

/// Logistic function: 1 / (1 + exp(-x)).
#[no_mangle]
pub extern "C" fn eml_logistic(x: c_double) -> c_double {
    1.0 / (1.0 + (-x).exp())
}

/// sqrt(a² + b²) — hypotenuse without overflow.
#[no_mangle]
pub extern "C" fn eml_hypot(a: c_double, b: c_double) -> c_double {
    a.hypot(b)
}

/// Arithmetic mean: (a + b) / 2.
#[no_mangle]
pub extern "C" fn eml_avg(a: c_double, b: c_double) -> c_double {
    (a + b) * 0.5
}

/// log_base(x) = ln(x) / ln(base). Returns NaN for invalid base (0, 1, negative).
#[no_mangle]
pub extern "C" fn eml_log(base: c_double, x: c_double) -> c_double {
    if base <= 0.0 || (base - 1.0).abs() < 1e-300 {
        return f64::NAN;
    }
    y_safe(x).ln() / y_safe(base).ln()
}

// ── Trigonometric functions ───────────────────────────────────────────────────

/// sin(x).
#[no_mangle]
pub extern "C" fn eml_sin(x: c_double) -> c_double {
    x.sin()
}

/// cos(x).
#[no_mangle]
pub extern "C" fn eml_cos(x: c_double) -> c_double {
    x.cos()
}

/// tan(x). Returns NaN near poles (|cos x| < 1e-15).
#[no_mangle]
pub extern "C" fn eml_tan(x: c_double) -> c_double {
    let c = x.cos();
    if c.abs() < 1e-15 {
        return f64::NAN;
    }
    x.sin() / c
}

/// arcsin(x). Returns NaN for |x| > 1.
#[no_mangle]
pub extern "C" fn eml_arcsin(x: c_double) -> c_double {
    if x.abs() > 1.0 {
        return f64::NAN;
    }
    x.asin()
}

/// arccos(x). Returns NaN for |x| > 1.
#[no_mangle]
pub extern "C" fn eml_arccos(x: c_double) -> c_double {
    if x.abs() > 1.0 {
        return f64::NAN;
    }
    x.acos()
}

/// arctan(x).
#[no_mangle]
pub extern "C" fn eml_arctan(x: c_double) -> c_double {
    x.atan()
}

/// arctan2(y, x) — four-quadrant arctangent.
#[no_mangle]
pub extern "C" fn eml_arctan2(y: c_double, x: c_double) -> c_double {
    y.atan2(x)
}

// ── Hyperbolic functions ──────────────────────────────────────────────────────

/// sinh(x) = (exp(x) - exp(-x)) / 2.
#[no_mangle]
pub extern "C" fn eml_sinh(x: c_double) -> c_double {
    x.sinh()
}

/// cosh(x) = (exp(x) + exp(-x)) / 2.
#[no_mangle]
pub extern "C" fn eml_cosh(x: c_double) -> c_double {
    x.cosh()
}

/// tanh(x).
#[no_mangle]
pub extern "C" fn eml_tanh(x: c_double) -> c_double {
    x.tanh()
}

/// arsinh(x) = ln(x + sqrt(x² + 1)).
#[no_mangle]
pub extern "C" fn eml_arsinh(x: c_double) -> c_double {
    x.asinh()
}

/// arcosh(x) = ln(x + sqrt(x² - 1)). Returns NaN for x < 1.
#[no_mangle]
pub extern "C" fn eml_arcosh(x: c_double) -> c_double {
    if x < 1.0 {
        return f64::NAN;
    }
    x.acosh()
}

/// artanh(x) = ln((1+x)/(1-x)) / 2. Returns NaN for |x| >= 1.
#[no_mangle]
pub extern "C" fn eml_artanh(x: c_double) -> c_double {
    if x.abs() >= 1.0 {
        return f64::NAN;
    }
    x.atanh()
}

// ── Batch arithmetic (auto-vectorisable scalar loops) ─────────────────────────

/// Batch exp: out[i] = eml_exp(xs[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_exp_batch(n: usize, xs: *const c_double, out: *mut c_double) {
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = eml_exp(*xs.add(i));
    }
}

/// Batch ln: out[i] = eml_ln(ys[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `ys` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_ln_batch(n: usize, ys: *const c_double, out: *mut c_double) {
    debug_assert!(!ys.is_null(), "ys must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = eml_ln(*ys.add(i));
    }
}

/// Batch add: out[i] = as_[i] + bs[i] for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `as_` is non-null and correctly aligned for `c_double`.
/// - `bs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_add_batch(
    n: usize,
    as_: *const c_double,
    bs: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!as_.is_null(), "as_ must not be null");
    debug_assert!(!bs.is_null(), "bs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = *as_.add(i) + *bs.add(i);
    }
}

/// Batch sub: out[i] = as_[i] - bs[i] for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `as_` is non-null and correctly aligned for `c_double`.
/// - `bs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_sub_batch(
    n: usize,
    as_: *const c_double,
    bs: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!as_.is_null(), "as_ must not be null");
    debug_assert!(!bs.is_null(), "bs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = *as_.add(i) - *bs.add(i);
    }
}

/// Batch mul: out[i] = as_[i] * bs[i] for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `as_` is non-null and correctly aligned for `c_double`.
/// - `bs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_mul_batch(
    n: usize,
    as_: *const c_double,
    bs: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!as_.is_null(), "as_ must not be null");
    debug_assert!(!bs.is_null(), "bs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = *as_.add(i) * *bs.add(i);
    }
}

/// Batch div: out[i] = as_[i] / bs[i]. Writes NaN when |bs[i]| < 1e-300.
/// # Safety
///
/// The caller guarantees that:
/// - `as_` is non-null and correctly aligned for `c_double`.
/// - `bs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_div_batch(
    n: usize,
    as_: *const c_double,
    bs: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!as_.is_null(), "as_ must not be null");
    debug_assert!(!bs.is_null(), "bs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = eml_div(*as_.add(i), *bs.add(i));
    }
}

/// Batch sqrt: out[i] = eml_sqrt(xs[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_sqrt_batch(n: usize, xs: *const c_double, out: *mut c_double) {
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = eml_sqrt(*xs.add(i));
    }
}

/// Batch sin: out[i] = sin(xs[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_sin_batch(n: usize, xs: *const c_double, out: *mut c_double) {
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = (*xs.add(i)).sin();
    }
}

/// Batch cos: out[i] = cos(xs[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_cos_batch(n: usize, xs: *const c_double, out: *mut c_double) {
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = (*xs.add(i)).cos();
    }
}

/// Batch eml_tension: out[i] = eml(xs[i], ys[i]) for i in 0..n.
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `ys` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_tension_batch(
    n: usize,
    xs: *const c_double,
    ys: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!ys.is_null(), "ys must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    for i in 0..n {
        *out.add(i) = eml_tension(*xs.add(i), *ys.add(i));
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
    if t.abs() < 1e-300 {
        return f64::NAN;
    }
    let ratio = s / t;
    if ratio.abs() >= 1.0 {
        return f64::NAN;
    }
    ratio.atanh()
}

/// Causal classification: +1 = timelike, -1 = spacelike, 0 = lightlike.
/// tol is the tolerance for lightlike detection.
#[no_mangle]
pub extern "C" fn eml_causal_type(x: c_double, y: c_double, c: c_double, tol: c_double) -> c_int {
    let xv = xv_safe(x);
    let ys = y_safe(y);
    let t2 = {
        let t = xv.exp();
        t * t
    };
    let s2 = {
        let s = c * ys.ln();
        s * s
    };
    if (t2 - s2).abs() < tol {
        0
    } else if t2 > s2 {
        1
    } else {
        -1
    }
}

// ── Lorentz boost ─────────────────────────────────────────────────────────────

/// Apply a Lorentz boost by rapidity phi with speed of light c.
/// Writes the new EML coordinates into *out_x, *out_y.
/// # Safety
///
/// The caller guarantees that:
/// - `out_x` is non-null and correctly aligned for `c_double`.
/// - `out_y` is non-null and correctly aligned for `c_double`.
/// - the output pointers do not alias each other.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_boost(
    x: c_double,
    y: c_double,
    phi: c_double,
    c: c_double,
    out_x: *mut c_double,
    out_y: *mut c_double,
) {
    debug_assert!(!out_x.is_null(), "out_x must not be null");
    debug_assert!(!out_y.is_null(), "out_y must not be null");
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
/// # Safety
///
/// The caller guarantees that:
/// - `xs` is non-null and correctly aligned for `c_double`.
/// - `ys` is non-null and correctly aligned for `c_double`.
/// - `phis` is non-null and correctly aligned for `c_double`.
/// - `out_xs` is non-null and correctly aligned for `c_double`.
/// - `out_ys` is non-null and correctly aligned for `c_double`.
/// - every array above is valid for the full element count requested,
///   and the output arrays do not alias the inputs.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
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
    debug_assert!(!xs.is_null(), "xs must not be null");
    debug_assert!(!ys.is_null(), "ys must not be null");
    debug_assert!(!phis.is_null(), "phis must not be null");
    debug_assert!(!out_xs.is_null(), "out_xs must not be null");
    debug_assert!(!out_ys.is_null(), "out_ys must not be null");
    for i in 0..n {
        eml_boost(
            *xs.add(i),
            *ys.add(i),
            *phis.add(i),
            c,
            out_xs.add(i),
            out_ys.add(i),
        );
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

// ── Octonion multiplication ───────────────────────────────────────────────────

const FANO_LINES: [(usize, usize, usize); 7] = [
    (1, 2, 4),
    (2, 3, 5),
    (3, 4, 6),
    (4, 5, 7),
    (1, 5, 6),
    (2, 6, 7),
    (1, 3, 7),
];

const fn build_oct_table() -> [[(i8, usize); 8]; 8] {
    let mut t = [[(0i8, 0usize); 8]; 8];
    let mut i = 0;
    while i < 8 {
        t[0][i] = (1, i);
        t[i][0] = (1, i);
        i += 1;
    }
    let mut i = 1;
    while i < 8 {
        t[i][i] = (-1, 0);
        i += 1;
    }
    let mut li = 0;
    while li < 7 {
        let (a, b, c) = FANO_LINES[li];
        t[a][b] = (1, c);
        t[b][a] = (-1, c);
        t[b][c] = (1, a);
        t[c][b] = (-1, a);
        t[c][a] = (1, b);
        t[a][c] = (-1, b);
        li += 1;
    }
    t
}

const OCT_TABLE: [[(i8, usize); 8]; 8] = build_oct_table();

/// Multiply two octonions. a and b point to 8-element f64 arrays.
/// Result is written to out (also 8 elements). Caller must allocate out.
/// # Safety
///
/// The caller guarantees that:
/// - `a` is non-null and correctly aligned for `c_double`.
/// - `b` is non-null and correctly aligned for `c_double`.
/// - `out` is non-null and correctly aligned for `c_double`.
/// - the output pointers do not alias each other.
///
/// Violating any of these is undefined behaviour; the debug assertions in the
/// body catch null pointers in a debug build only.
#[no_mangle]
pub unsafe extern "C" fn eml_octonion_mul(
    a: *const c_double,
    b: *const c_double,
    out: *mut c_double,
) {
    debug_assert!(!a.is_null(), "a must not be null");
    debug_assert!(!b.is_null(), "b must not be null");
    debug_assert!(!out.is_null(), "out must not be null");
    let a = std::slice::from_raw_parts(a, 8);
    let b = std::slice::from_raw_parts(b, 8);
    let out = std::slice::from_raw_parts_mut(out, 8);
    for v in out.iter_mut() {
        *v = 0.0;
    }
    // The index is the basis-element number, so it is read from `enumerate`.
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj == 0.0 {
                continue;
            }
            let (sign, k) = OCT_TABLE[i][j];
            debug_assert!(k < 8, "the Fano table must only name real basis indices");
            out[k] += (sign as f64) * ai * bj;
        }
    }
}
