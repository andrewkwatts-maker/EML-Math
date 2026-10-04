//! Levenberg-Marquardt fitting of tunable constants inside a candidate
//! expression.
//!
//! # This code is currently unreachable at runtime
//!
//! `search::score_level` always passes an empty parameter vector, and the
//! candidate builder never constructs a `Node::Param`, so `n_params` is always
//! zero and `optimize` returns on its first branch. The whole damped
//! least-squares body below therefore never executes in a released build. It is
//! left in place, correct and unit-tested, rather than deleted: making it live
//! means seeding the search with a `Node::Param` leaf, which changes the
//! formulas the search returns and so is a behavioural change, not a fix.
//!
//! The practical consequence for callers is that `find_formula` performs
//! **structure search only**; it never fits numeric coefficients, and the
//! `params` list on a result is always empty.

use crate::discover::expr::Expr;

/// Damped least-squares iteration cap. Standard 2: fixed loop bounds.
const MAX_ITERS: usize = 20;

/// Forward-difference step for the numerical Jacobian.
const EPSILON: f64 = 1e-6;

/// Largest parameter vector the solver will accept.
///
/// The Jacobian is `y.len() * n_params`, so an unbounded parameter count is an
/// unbounded allocation.
pub const MAX_PARAMS: usize = 64;

/// Levenberg-Marquardt constant optimisation.
///
/// Returns `(optimized_params, final_rmse)`. On a degenerate input the
/// parameters are returned unchanged with the RMSE they produce, so a caller
/// never receives silently fabricated coefficients.
pub fn optimize(
    expr: &Expr,
    data: &[Vec<f64>],
    y: &[f64],
    initial_params: &[f64],
) -> (Vec<f64>, f64) {
    // Caller input, not an internal invariant -- see the note in `search`.
    // The function already refuses these by returning the initial parameters
    // unchanged, which is the documented behaviour; asserting first made that
    // refusal unreachable under test.
    let n_params = initial_params.len();
    if n_params == 0 || n_params > MAX_PARAMS || y.is_empty() {
        let rmse = compute_rmse(expr, data, y, initial_params);
        return (initial_params.to_vec(), rmse);
    }
    debug_assert!(!y.is_empty(), "the guard above should have returned");
    debug_assert!(n_params <= MAX_PARAMS, "parameter cap not enforced");

    let mut params = initial_params.to_vec();
    let mut lambda = 1e-3_f64;
    let mut best_rmse = compute_rmse(expr, data, y, &params);

    for _ in 0..MAX_ITERS {
        // Numerical Jacobian: J[i][j] = d(residual_i)/d(param_j)
        let residuals = match compute_residuals(expr, data, y, &params) {
            Some(r) => r,
            None => break,
        };

        // Forward-difference Jacobian, stored column-major so each parameter
        // owns a contiguous run: J[j][i] = d(residual_i) / d(param_j).
        let mut jacobian = vec![vec![0.0_f64; y.len()]; n_params];
        for (j, column) in jacobian.iter_mut().enumerate() {
            let mut params_plus = params.clone();
            params_plus[j] += EPSILON;
            let res_plus = match compute_residuals(expr, data, y, &params_plus) {
                Some(r) => r,
                None => continue,
            };
            debug_assert_eq!(
                res_plus.len(),
                residuals.len(),
                "residual vectors must align"
            );
            for (slot, (plus, base)) in column.iter_mut().zip(res_plus.iter().zip(residuals.iter()))
            {
                *slot = (plus - base) / EPSILON;
            }
        }

        // Gradient g = J^T r, and the diagonal of J^T J, in one pass per column.
        let mut gradient = vec![0.0_f64; n_params];
        let mut diag = vec![0.0_f64; n_params];
        for ((g, d), column) in gradient
            .iter_mut()
            .zip(diag.iter_mut())
            .zip(jacobian.iter())
        {
            for (value, residual) in column.iter().zip(residuals.iter()) {
                *g += value * residual;
                *d += value * value;
            }
        }

        // Damped update: params -= g / (diag + lambda).
        let mut new_params = params.clone();
        for ((slot, g), d) in new_params.iter_mut().zip(gradient.iter()).zip(diag.iter()) {
            *slot -= g / (d + lambda);
        }

        let new_rmse = compute_rmse(expr, data, y, &new_params);
        if new_rmse < best_rmse {
            params = new_params;
            best_rmse = new_rmse;
            lambda /= 10.0;
        } else {
            lambda *= 10.0;
        }

        if best_rmse < 1e-12 {
            break;
        }
    }

    (params, best_rmse)
}

/// Residual vector `predicted - target`, or `None` if the expression left its
/// domain at any sample.
fn compute_residuals(
    expr: &Expr,
    data: &[Vec<f64>],
    y: &[f64],
    params: &[f64],
) -> Option<Vec<f64>> {
    debug_assert!(!y.is_empty(), "an empty target vector has no residuals");
    debug_assert!(!data.is_empty(), "residuals need at least one input column");
    let predicted = expr.eval_batch(data, params)?;
    if predicted.len() != y.len() {
        return None;
    }
    Some(predicted.iter().zip(y.iter()).map(|(p, t)| p - t).collect())
}

/// Root-mean-square error, or infinity when the expression is undefined
/// anywhere on the data. Infinity is a comparison sentinel here, not a silently
/// substituted default: it can only lose to a real candidate.
fn compute_rmse(expr: &Expr, data: &[Vec<f64>], y: &[f64], params: &[f64]) -> f64 {
    // Reached on the refusal path of `optimize`, which scores an over-long
    // parameter vector rather than truncating it -- so an assert on the
    // parameter count here fires on exactly the input that path exists to
    // handle. Infinity already covers the degenerate cases.
    if y.is_empty() {
        return f64::INFINITY;
    }
    debug_assert!(!y.is_empty(), "the guard above should have returned");
    debug_assert!(!data.is_empty() || y.is_empty(), "scoring needs samples");
    match expr.eval_batch(data, params) {
        None => f64::INFINITY,
        Some(pred) => {
            let mse = pred
                .iter()
                .zip(y.iter())
                .map(|(p, t)| (p - t).powi(2))
                .sum::<f64>()
                / y.len() as f64;
            mse.sqrt()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::expr::{Node, Op};

    /// `c0 * x`, with `c0` seeded away from its true value.
    fn scaled_var(initial: f64) -> Expr {
        Expr {
            nodes: vec![Node::Param(0, initial), Node::Var(0), Node::Op(Op::Mul)],
            n_vars: 1,
        }
    }

    #[test]
    fn an_empty_parameter_vector_is_returned_unchanged() {
        let expr = Expr::new_var(0);
        let data = vec![vec![1.0, 2.0, 3.0]];
        let (params, rmse) = optimize(&expr, &data, &[1.0, 2.0, 3.0], &[]);
        assert!(params.is_empty());
        assert!(
            rmse < 1e-12,
            "identity on identity must fit exactly, got {rmse}"
        );
    }

    #[test]
    fn fitting_recovers_a_linear_coefficient() {
        let data = vec![vec![1.0, 2.0, 3.0, 4.0]];
        let y: Vec<f64> = data[0].iter().map(|x| 3.0 * x).collect();
        let (params, rmse) = optimize(&scaled_var(1.0), &data, &y, &[1.0]);
        assert_eq!(params.len(), 1);
        assert!((params[0] - 3.0).abs() < 1e-4, "recovered {}", params[0]);
        assert!(rmse < 1e-4, "residual RMSE was {rmse}");
    }

    #[test]
    fn an_oversized_parameter_vector_is_refused_not_truncated() {
        let data = vec![vec![1.0, 2.0]];
        let big = vec![0.0; MAX_PARAMS + 1];
        let (params, _) = optimize(&scaled_var(1.0), &data, &[1.0, 2.0], &big);
        assert_eq!(
            params.len(),
            big.len(),
            "parameters must come back untouched"
        );
    }

    #[test]
    fn rmse_is_infinite_when_the_expression_leaves_its_domain() {
        let ln = Expr {
            nodes: vec![Node::Var(0), Node::Op(Op::Ln)],
            n_vars: 1,
        };
        let rmse = compute_rmse(&ln, &[vec![-1.0, -2.0]], &[0.0, 0.0], &[]);
        assert!(rmse.is_infinite());
    }
}
