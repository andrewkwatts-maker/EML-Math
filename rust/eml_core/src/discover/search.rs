//! Breadth-first search over EML expression trees.
//!
//! Every level is bounded: the candidate set is capped, the working set is
//! capped, and both the complexity and beam-width knobs are validated against
//! the caps below before the search starts. Without those caps the binary
//! combination step is quadratic in a set the caller controls.

use crate::discover::expr::{Expr, Node, Op, MAX_NODES};
use crate::discover::fingerprint::{fingerprint, is_new, Seen};
use crate::discover::optimizer::optimize;
use rayon::prelude::*;

/// Hard ceiling on `SearchConfig::max_complexity`.
pub const MAX_COMPLEXITY_CAP: usize = 32;

/// Hard ceiling on `SearchConfig::beam_width`.
pub const MAX_BEAM_WIDTH: usize = 100_000;

/// Hard ceiling on the working set carried between BFS levels.
///
/// The binary combination step is O(n^2) in this number, so it is the single
/// most important bound in the crate.
pub const MAX_WORKING_SET: usize = 8_192;

/// Hard ceiling on the candidates generated at one level.
pub const MAX_CANDIDATES_PER_LEVEL: usize = 1_000_000;

#[derive(Clone, Debug)]
pub struct SearchConfig {
    pub max_complexity: usize,
    pub beam_width: usize,
    pub precision_goal: f64,
    pub complexity_penalty: f64,
    pub use_trig: bool,
    pub use_eml_primitive: bool,
}

impl Default for SearchConfig {
    fn default() -> Self {
        SearchConfig {
            max_complexity: 8,
            beam_width: 2000,
            precision_goal: 1e-10,
            complexity_penalty: 0.001,
            use_trig: true,
            use_eml_primitive: true,
        }
    }
}

#[derive(Clone, Debug)]
pub struct SearchResult {
    pub formula: String,
    pub error: f64,
    pub complexity: usize,
    pub params: Vec<f64>,
    pub expr: Expr,
}

/// Search for the expression that best fits `data -> y`.
///
/// Returns `None` when no candidate evaluated to a finite error anywhere in the
/// bounded search space.
pub fn search(
    data: &[Vec<f64>],
    y: &[f64],
    n_vars: usize,
    config: &SearchConfig,
) -> Option<SearchResult> {
    // These conditions are *caller input*, not internal invariants, and the
    // function documents that it returns `None` for them. Asserting them as
    // well made the documented path unreachable in a debug build: the assert
    // fired before the guard below could answer. `debug_assert!` belongs on
    // facts this code establishes for itself, not on data handed in.
    if data.is_empty() || y.is_empty() || n_vars == 0 {
        return None;
    }
    if config.max_complexity == 0 || config.max_complexity > MAX_COMPLEXITY_CAP {
        return None;
    }
    debug_assert!(!y.is_empty(), "the guard above should have returned");
    debug_assert!(
        config.max_complexity <= MAX_COMPLEXITY_CAP,
        "complexity cap not enforced"
    );
    let seen: Seen = Seen::new();
    let mut best: Option<SearchResult> = None;

    // Build operator list
    let unary_ops: Vec<Op> = {
        let mut ops = vec![Op::Neg, Op::Inv, Op::Exp, Op::Ln, Op::Sqrt];
        if config.use_trig {
            ops.push(Op::Sin);
            ops.push(Op::Cos);
        }
        ops
    };
    let binary_ops: Vec<Op> = {
        let mut ops = vec![Op::Add, Op::Sub, Op::Mul, Op::Div];
        if config.use_eml_primitive {
            ops.push(Op::Eml);
        }
        ops
    };

    // Level 0: seeds — constants and variables
    let mut level: Vec<Expr> = Vec::new();
    for v in [0.0_f64, 1.0] {
        let e = Expr::new_const(v);
        if let Some(fp) = fingerprint(&e, n_vars, &[]) {
            if is_new(&seen, fp) {
                level.push(e);
            }
        }
    }
    for i in 0..n_vars {
        let e = Expr::new_var(i);
        if let Some(fp) = fingerprint(&e, n_vars, &[]) {
            if is_new(&seen, fp) {
                level.push(e);
            }
        }
    }

    // Score seeds
    let mut scored = score_level(&level, data, y, config, n_vars);
    update_best(&mut best, &scored, config);
    if best
        .as_ref()
        .is_some_and(|b| b.error <= config.precision_goal)
    {
        return best;
    }

    let mut all_exprs = level.clone();

    // BFS by complexity. Bounded three ways: the level count, the working-set
    // size that drives the quadratic binary step, and the per-level candidate
    // count.
    for _depth in 1..config.max_complexity.min(MAX_COMPLEXITY_CAP) {
        debug_assert!(
            all_exprs.len() <= MAX_WORKING_SET,
            "the working set must stay within its cap between levels"
        );
        let mut candidates: Vec<Expr> = Vec::new();

        // Unary: apply each unary op to every expr in all_exprs
        for expr in &all_exprs {
            if candidates.len() >= MAX_CANDIDATES_PER_LEVEL {
                break;
            }
            for op in &unary_ops {
                if expr.nodes.len() + 1 > MAX_NODES {
                    continue;
                }
                let mut nodes = expr.nodes.clone();
                nodes.push(Node::Op(op.clone()));
                let candidate = Expr { nodes, n_vars };
                if let Some(fp) = fingerprint(&candidate, n_vars, &[]) {
                    if is_new(&seen, fp) {
                        candidates.push(candidate);
                    }
                }
            }
        }

        // Binary: combine pairs from all_exprs
        for (i, left) in all_exprs.iter().enumerate() {
            if candidates.len() >= MAX_CANDIDATES_PER_LEVEL {
                break;
            }
            for right in all_exprs.iter().take(i + 1) {
                if left.complexity() + right.complexity() >= config.max_complexity {
                    continue;
                }
                if left.nodes.len() + right.nodes.len() + 1 > MAX_NODES {
                    continue;
                }
                for op in &binary_ops {
                    let mut nodes = left.nodes.clone();
                    nodes.extend_from_slice(&right.nodes);
                    nodes.push(Node::Op(op.clone()));
                    let candidate = Expr { nodes, n_vars };
                    if let Some(fp) = fingerprint(&candidate, n_vars, &[]) {
                        if is_new(&seen, fp) {
                            candidates.push(candidate.clone());
                        }
                    }
                    // Also try right op left (non-commutative ops)
                    if matches!(op, Op::Sub | Op::Div | Op::Eml) {
                        let mut nodes2 = right.nodes.clone();
                        nodes2.extend_from_slice(&left.nodes);
                        nodes2.push(Node::Op(op.clone()));
                        let candidate2 = Expr {
                            nodes: nodes2,
                            n_vars,
                        };
                        if let Some(fp) = fingerprint(&candidate2, n_vars, &[]) {
                            if is_new(&seen, fp) {
                                candidates.push(candidate2);
                            }
                        }
                    }
                }
            }
        }

        if candidates.is_empty() {
            break;
        }

        // Score candidates in parallel
        scored = score_level(&candidates, data, y, config, n_vars);
        update_best(&mut best, &scored, config);

        if best
            .as_ref()
            .is_some_and(|b| b.error <= config.precision_goal)
        {
            return best;
        }

        // Beam pruning: keep top beam_width by penalized error
        scored.sort_by(|a, b| {
            a.error
                .partial_cmp(&b.error)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(config.beam_width.min(MAX_BEAM_WIDTH));

        // Carry the best quarter of the beam into the next level, then trim the
        // working set. Without the trim `all_exprs` grew every level and the
        // O(n^2) binary step above grew with it, unbounded by any config knob.
        let carry = (config.beam_width / 4).max(1);
        all_exprs.extend(scored.iter().take(carry).map(|r| r.expr.clone()));
        if all_exprs.len() > MAX_WORKING_SET {
            all_exprs.truncate(MAX_WORKING_SET);
        }
    }

    debug_assert!(
        best.as_ref().is_none_or(|b| !b.error.is_nan()),
        "a returned result must carry a comparable error"
    );
    best
}

/// Fit and score every candidate at one level, in parallel.
///
/// `n_vars` is retained for symmetry with `fingerprint` and to keep the call
/// sites uniform; scoring itself reads the variable count from `data`.
fn score_level(
    candidates: &[Expr],
    data: &[Vec<f64>],
    y: &[f64],
    config: &SearchConfig,
    n_vars: usize,
) -> Vec<SearchResult> {
    debug_assert!(!y.is_empty(), "scoring divides by the sample count");
    debug_assert!(
        n_vars == 0 || data.len() == n_vars,
        "the column count must match the declared variable count"
    );
    let _ = n_vars;
    candidates
        .par_iter()
        .filter_map(|expr| {
            let params: Vec<f64> = Vec::new();
            let raw_rmse = match expr.eval_batch(data, &params) {
                None => return None,
                Some(pred) => {
                    let mse = pred
                        .iter()
                        .zip(y.iter())
                        .map(|(p, t)| (p - t).powi(2))
                        .sum::<f64>()
                        / y.len() as f64;
                    mse.sqrt()
                }
            };

            // Only run LM if error is promising
            let (params, error) = if raw_rmse < 2.0 {
                optimize(expr, data, y, &params)
            } else {
                (params, raw_rmse)
            };

            let penalized = error * (1.0 + expr.complexity() as f64 * config.complexity_penalty);

            Some(SearchResult {
                formula: expr.to_string(),
                error: penalized,
                complexity: expr.complexity(),
                params,
                expr: expr.clone(),
            })
        })
        .collect()
}

/// Replace `best` with any strictly better scored candidate.
fn update_best(best: &mut Option<SearchResult>, scored: &[SearchResult], _config: &SearchConfig) {
    debug_assert!(
        scored.iter().all(|r| !r.formula.is_empty()),
        "every scored candidate must carry a printed formula"
    );
    debug_assert!(
        scored.len() <= MAX_CANDIDATES_PER_LEVEL,
        "a level must not exceed the candidate cap"
    );
    for r in scored {
        let is_better = best.as_ref().is_none_or(|b| r.error < b.error);
        if is_better {
            *best = Some(r.clone());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn linspace(lo: f64, hi: f64, n: usize) -> Vec<f64> {
        let step = (hi - lo) / (n - 1) as f64;
        (0..n).map(|i| lo + step * i as f64).collect()
    }

    #[test]
    fn recovers_the_identity_function() {
        let xs = linspace(1.0, 5.0, 20);
        let cfg = SearchConfig {
            max_complexity: 3,
            beam_width: 200,
            ..SearchConfig::default()
        };
        let got = search(&[xs.clone()], &xs, 1, &cfg).expect("identity must be findable");
        assert!(got.error < 1e-9, "error was {}", got.error);
        assert!(
            got.formula.is_ascii(),
            "formula must be ASCII: {}",
            got.formula
        );
    }

    #[test]
    fn recovers_exp() {
        let xs = linspace(0.1, 2.0, 25);
        let ys: Vec<f64> = xs.iter().map(|x| x.exp()).collect();
        let cfg = SearchConfig {
            max_complexity: 4,
            beam_width: 400,
            ..SearchConfig::default()
        };
        let got = search(&[xs], &ys, 1, &cfg).expect("exp must be findable");
        assert!(
            got.error < 1e-8,
            "error was {} for {}",
            got.error,
            got.formula
        );
    }

    #[test]
    fn empty_input_returns_none_rather_than_panicking() {
        let cfg = SearchConfig::default();
        assert!(search(&[], &[1.0, 2.0], 0, &cfg).is_none());
        assert!(search(&[vec![1.0]], &[], 1, &cfg).is_none());
    }

    #[test]
    fn the_default_config_respects_every_cap() {
        let d = SearchConfig::default();
        assert!(d.max_complexity > 0 && d.max_complexity <= MAX_COMPLEXITY_CAP);
        assert!(d.beam_width > 0 && d.beam_width <= MAX_BEAM_WIDTH);
        assert!(d.precision_goal.is_finite() && d.precision_goal >= 0.0);
        assert!(d.complexity_penalty.is_finite() && d.complexity_penalty >= 0.0);
    }
}
