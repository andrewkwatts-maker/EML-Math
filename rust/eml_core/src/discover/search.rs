use rayon::prelude::*;
use crate::discover::expr::{Expr, Node, Op};
use crate::discover::fingerprint::{fingerprint, is_new, Seen};
use crate::discover::optimizer::optimize;

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

pub fn search(
    data: &[Vec<f64>],
    y: &[f64],
    n_vars: usize,
    config: &SearchConfig,
) -> Option<SearchResult> {
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
    if best.as_ref().map_or(false, |b| b.error <= config.precision_goal) {
        return best;
    }

    let mut all_exprs = level.clone();

    // BFS by complexity
    for _depth in 1..config.max_complexity {
        let mut candidates: Vec<Expr> = Vec::new();

        // Unary: apply each unary op to every expr in all_exprs
        for expr in &all_exprs {
            for op in &unary_ops {
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
            for right in all_exprs.iter().take(i + 1) {
                if left.complexity() + right.complexity() >= config.max_complexity {
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
                        let candidate2 = Expr { nodes: nodes2, n_vars };
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

        if best.as_ref().map_or(false, |b| b.error <= config.precision_goal) {
            return best;
        }

        // Beam pruning: keep top beam_width by penalized error
        scored.sort_by(|a, b| a.error.partial_cmp(&b.error).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(config.beam_width);

        // Add best candidates to all_exprs for next level
        all_exprs.extend(scored.iter().take(config.beam_width / 4).map(|r| r.expr.clone()));
    }

    best
}

fn score_level(
    candidates: &[Expr],
    data: &[Vec<f64>],
    y: &[f64],
    config: &SearchConfig,
    n_vars: usize,
) -> Vec<SearchResult> {
    candidates
        .par_iter()
        .filter_map(|expr| {
            let params: Vec<f64> = Vec::new();
            let raw_rmse = match expr.eval_batch(data, &params) {
                None => return None,
                Some(pred) => {
                    let mse = pred.iter().zip(y.iter()).map(|(p, t)| (p - t).powi(2)).sum::<f64>()
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

fn update_best(best: &mut Option<SearchResult>, scored: &[SearchResult], _config: &SearchConfig) {
    for r in scored {
        let is_better = best.as_ref().map_or(true, |b| r.error < b.error);
        if is_better {
            *best = Some(r.clone());
        }
    }
}
