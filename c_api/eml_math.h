/**
 * eml_math.h — C API for the EML Sheffer operator library.
 *
 * The EML operator: eml(x, y) = exp(x) - ln(y)
 *
 * Build the library:
 *   cargo build --release -p eml_c_api
 *
 * Link in C/C++:
 *   gcc your_program.c -L./target/release -leml_math -lm -o your_program
 *   g++ your_program.cpp -L./target/release -leml_math -lm -o your_program
 *
 * In Rust (Cargo.toml):
 *   [dependencies]
 *   eml_c_api = { path = "path/to/c_api" }
 *
 * Frame-shift guard (Axiom 8): y ≤ 0 is replaced by |y|, so all functions
 * accepting a y-coordinate are well-defined on the full real line.
 */

#ifndef EML_MATH_H
#define EML_MATH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ─── Core EML operator ──────────────────────────────────────────────────── */

/** eml(x, y) = exp(x) - ln(y). Applies frame-shift guard (|y| when y <= 0). */
double eml_tension(double x, double y);

/* ─── Mirror-Pulse iteration ─────────────────────────────────────────────── */

/** One Mirror-Pulse step: (x, y) -> (|y|, exp(x) - ln(|y|)).
 *  Writes new coordinates into *out_x and *out_y. */
void eml_mirror_pulse(double x, double y, double *out_x, double *out_y);

/** Run n_pulses from (x0, y0). Caller must allocate out_xs[n_pulses+1]
 *  and out_ys[n_pulses+1]. Index 0 stores the initial state. */
void eml_simulate_pulses(double x0, double y0, size_t n_pulses,
                         double *out_xs, double *out_ys);

/* ─── Elementary arithmetic operators ───────────────────────────────────── */

/** exp(x) = eml(x, 1). Caps x at the overflow threshold (~709.78). */
double eml_exp(double x);

/** ln(y). Applies frame-shift guard: ln(|y|) for y <= 0. */
double eml_ln(double y);

/** a + b */
double eml_add(double a, double b);

/** a - b */
double eml_sub(double a, double b);

/** a * b */
double eml_mul(double a, double b);

/** a / b. Returns NaN when |b| < 1e-300. */
double eml_div(double a, double b);

/** sqrt(|x|) — square root with Axiom-8 absolute value guard. */
double eml_sqrt(double x);

/** x^2 — squaring. */
double eml_sqr(double x);

/** |base|^exp — power function with absolute-value base guard. */
double eml_pow(double base, double exp);

/** -x */
double eml_neg(double x);

/** 1/x. Returns NaN when |x| < 1e-300. */
double eml_inv(double x);

/** x / 2. */
double eml_half(double x);

/** 1 / (1 + exp(-x)) — logistic / sigmoid function. */
double eml_logistic(double x);

/** sqrt(a^2 + b^2) — hypotenuse without overflow. */
double eml_hypot(double a, double b);

/** (a + b) / 2 — arithmetic mean. */
double eml_avg(double a, double b);

/** log_base(x) = ln(x) / ln(base). Returns NaN for base <= 0 or base == 1. */
double eml_log(double base, double x);

/* ─── Trigonometric functions ────────────────────────────────────────────── */

/** sin(x) */
double eml_sin(double x);

/** cos(x) */
double eml_cos(double x);

/** tan(x). Returns NaN near poles (|cos x| < 1e-15). */
double eml_tan(double x);

/** arcsin(x). Returns NaN for |x| > 1. Range: [-π/2, π/2]. */
double eml_arcsin(double x);

/** arccos(x). Returns NaN for |x| > 1. Range: [0, π]. */
double eml_arccos(double x);

/** arctan(x). Range: (-π/2, π/2). */
double eml_arctan(double x);

/** arctan2(y, x) — four-quadrant arctangent. Range: (-π, π]. */
double eml_arctan2(double y, double x);

/* ─── Hyperbolic functions ───────────────────────────────────────────────── */

/** sinh(x) = (exp(x) - exp(-x)) / 2 */
double eml_sinh(double x);

/** cosh(x) = (exp(x) + exp(-x)) / 2 */
double eml_cosh(double x);

/** tanh(x) */
double eml_tanh(double x);

/** arsinh(x) = ln(x + sqrt(x^2 + 1)) */
double eml_arsinh(double x);

/** arcosh(x) = ln(x + sqrt(x^2 - 1)). Returns NaN for x < 1. */
double eml_arcosh(double x);

/** artanh(x) = ln((1+x)/(1-x)) / 2. Returns NaN for |x| >= 1. */
double eml_artanh(double x);

/* ─── Batch arithmetic (scalar loops, auto-vectorisable) ─────────────────── */

/** Batch exp: out[i] = eml_exp(xs[i]) for i in [0, n). */
void eml_exp_batch(size_t n, const double *xs, double *out);

/** Batch ln: out[i] = eml_ln(ys[i]) for i in [0, n). */
void eml_ln_batch(size_t n, const double *ys, double *out);

/** Batch add: out[i] = as_[i] + bs[i] for i in [0, n). */
void eml_add_batch(size_t n, const double *as_, const double *bs, double *out);

/** Batch sub: out[i] = as_[i] - bs[i] for i in [0, n). */
void eml_sub_batch(size_t n, const double *as_, const double *bs, double *out);

/** Batch mul: out[i] = as_[i] * bs[i] for i in [0, n). */
void eml_mul_batch(size_t n, const double *as_, const double *bs, double *out);

/** Batch div: out[i] = as_[i] / bs[i]. Writes NaN when |bs[i]| < 1e-300. */
void eml_div_batch(size_t n, const double *as_, const double *bs, double *out);

/** Batch sqrt: out[i] = eml_sqrt(xs[i]) for i in [0, n). */
void eml_sqrt_batch(size_t n, const double *xs, double *out);

/** Batch sin: out[i] = sin(xs[i]) for i in [0, n). */
void eml_sin_batch(size_t n, const double *xs, double *out);

/** Batch cos: out[i] = cos(xs[i]) for i in [0, n). */
void eml_cos_batch(size_t n, const double *xs, double *out);

/** Batch eml_tension: out[i] = eml(xs[i], ys[i]) for i in [0, n). */
void eml_tension_batch(size_t n, const double *xs, const double *ys, double *out);

/* ─── Geometric invariants ───────────────────────────────────────────────── */

/** sqrt(exp(2x) + (ln y)^2) — Euclidean frame invariant. */
double eml_euclidean_delta(double x, double y);

/** sqrt(|exp(2x) - (c*ln y)^2|) — Minkowski spacetime interval.
 *  plus_signature=1 for (+---), 0 for (-+++) convention. */
double eml_minkowski_delta(double x, double y, int plus_signature, double c);

/** Rapidity φ (phi) = atanh(ln(y) / exp(x)).
 *  φ is additive under sequential boosts in the same direction.
 *  Returns NaN if the point is not timelike (|ln y| >= |exp x|). */
double eml_rapidity(double x, double y);

/** Causal type: +1 = timelike, -1 = spacelike, 0 = lightlike.
 *  tol is the tolerance for lightlike detection (e.g. 1e-9). */
int eml_causal_type(double x, double y, double c, double tol);

/* ─── Lorentz boost ──────────────────────────────────────────────────────── */

/** Apply a Lorentz boost by rapidity phi with speed of light c.
 *  Writes boosted EML coordinates into *out_x, *out_y.
 *  The Minkowski delta is preserved: eml_minkowski_delta is invariant. */
void eml_boost(double x, double y, double phi, double c,
               double *out_x, double *out_y);

/** Batch boost: apply boost independently to n (x,y) pairs.
 *  xs, ys, phis: input arrays of length n.
 *  out_xs, out_ys: output arrays of length n (caller-allocated). */
void eml_boost_batch(const double *xs, const double *ys,
                     const double *phis, double c,
                     size_t n,
                     double *out_xs, double *out_ys);

/* ─── Schwarzschild Christoffel symbols ──────────────────────────────────── */

/** Analytic Gamma^lam_{mu nu} for the Schwarzschild metric (2D radial slice).
 *  Index convention: upper index lam (contravariant), lower indices mu, nu.
 *  r = radial coordinate, rs = Schwarzschild radius (2GM/c^2).
 *  Metric signature used: (-,+) i.e. g_tt < 0, g_rr > 0.
 *  Returns 0 for r <= rs or unrecognised index combinations. */
double eml_schwarzschild_christoffel(size_t lam, size_t mu, size_t nu,
                                     double r, double rs);

/* ─── Octonion multiplication ────────────────────────────────────────────── */

/** Multiply two octonions using the Fano-plane multiplication table.
 *  a[8]: left operand coefficients (e_0 ... e_7).
 *  b[8]: right operand coefficients.
 *  out[8]: result (caller-allocated). Norm is multiplicative: |ab|=|a|*|b|. */
void eml_octonion_mul(const double a[8], const double b[8], double out[8]);

#ifdef __cplusplus
}
#endif

#endif /* EML_MATH_H */
