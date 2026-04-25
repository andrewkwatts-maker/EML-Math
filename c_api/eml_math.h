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
