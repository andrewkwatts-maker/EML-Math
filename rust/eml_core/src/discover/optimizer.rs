use crate::discover::expr::Expr;

const MAX_ITERS: usize = 20;
const EPSILON: f64 = 1e-6;

/// Levenberg-Marquardt constant optimization.
/// Returns (optimized_params, final_rmse).
pub fn optimize(
    expr: &Expr,
    data: &[Vec<f64>],
    y: &[f64],
    initial_params: &[f64],
) -> (Vec<f64>, f64) {
    let n_params = initial_params.len();
    if n_params == 0 {
        let rmse = compute_rmse(expr, data, y, initial_params);
        return (initial_params.to_vec(), rmse);
    }

    let mut params = initial_params.to_vec();
    let mut lambda = 1e-3_f64;
    let mut best_rmse = compute_rmse(expr, data, y, &params);

    for _ in 0..MAX_ITERS {
        // Numerical Jacobian: J[i][j] = d(residual_i)/d(param_j)
        let residuals = match compute_residuals(expr, data, y, &params) {
            Some(r) => r,
            None => break,
        };

        let mut jacobian = vec![vec![0.0_f64; n_params]; y.len()];
        for j in 0..n_params {
            let mut params_plus = params.clone();
            params_plus[j] += EPSILON;
            let res_plus = match compute_residuals(expr, data, y, &params_plus) {
                Some(r) => r,
                None => continue,
            };
            for i in 0..y.len() {
                jacobian[i][j] = (res_plus[i] - residuals[i]) / EPSILON;
            }
        }

        // Gradient: g = J^T · r
        let mut gradient = vec![0.0_f64; n_params];
        for j in 0..n_params {
            for i in 0..y.len() {
                gradient[j] += jacobian[i][j] * residuals[i];
            }
        }

        // Diagonal of J^T·J
        let mut diag = vec![0.0_f64; n_params];
        for j in 0..n_params {
            for i in 0..y.len() {
                diag[j] += jacobian[i][j] * jacobian[i][j];
            }
        }

        // Update: params -= gradient / (diag + lambda)
        let mut new_params = params.clone();
        for j in 0..n_params {
            new_params[j] -= gradient[j] / (diag[j] + lambda);
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

fn compute_residuals(
    expr: &Expr,
    data: &[Vec<f64>],
    y: &[f64],
    params: &[f64],
) -> Option<Vec<f64>> {
    let predicted = expr.eval_batch(data, params)?;
    Some(predicted.iter().zip(y.iter()).map(|(p, t)| p - t).collect())
}

fn compute_rmse(expr: &Expr, data: &[Vec<f64>], y: &[f64], params: &[f64]) -> f64 {
    match expr.eval_batch(data, params) {
        None => f64::INFINITY,
        Some(pred) => {
            let mse = pred.iter().zip(y.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>()
                / y.len() as f64;
            mse.sqrt()
        }
    }
}
