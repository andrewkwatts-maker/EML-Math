//! Flat RPN expression trees for the formula search.
//!
//! `Op::Eml` used to inline `709.78` and `.max(1e-300)`, so the search
//! primitive disagreed with `EMLPoint::tension` for the same inputs. It now
//! calls the single definition in [`crate::guard`].

use crate::guard::{eml, MAX_SAMPLES};

/// Largest number of nodes an expression may hold.
///
/// The RPN evaluator pushes at most one stack slot per node, so this also
/// bounds the evaluation stack.
pub const MAX_NODES: usize = 4096;

/// RPN expression tree node.
/// A node of the RPN program.
///
/// `Param` and `Op::Abs` are constructed by no code path today: the search
/// builder emits only `Const`, `Var` and the operators it lists explicitly.
/// They are kept because removing them would also remove the only description
/// of the interface the Levenberg-Marquardt optimiser expects. See the module
/// note in `optimizer.rs`.
#[allow(dead_code)]
#[derive(Clone, Debug, PartialEq)]
pub enum Node {
    Const(f64),        // fixed constant
    Param(usize, f64), // tunable parameter: (id, initial_value)
    Var(usize),        // input variable index
    Op(Op),
}

#[allow(dead_code)] // `Abs` has no builder yet; see the note on `Node`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Op {
    // Unary
    Neg,
    Inv,
    Exp,
    Ln,
    Sqrt,
    Sin,
    Cos,
    Abs,
    // Binary
    Eml, // exp(a) - ln(b)  — the Sheffer primitive
    Add,
    Sub,
    Mul,
    Div,
}

impl Op {
    pub fn arity(&self) -> usize {
        match self {
            Op::Neg | Op::Inv | Op::Exp | Op::Ln | Op::Sqrt | Op::Sin | Op::Cos | Op::Abs => 1,
            Op::Eml | Op::Add | Op::Sub | Op::Mul | Op::Div => 2,
        }
    }
}

/// Flat RPN expression: evaluated left-to-right with a stack.
///
/// `n_vars` is carried for the benefit of callers that rebuild an `Expr` from
/// parts; the evaluator reads variable indices from the nodes themselves.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Expr {
    pub nodes: Vec<Node>,
    pub n_vars: usize,
}

impl Expr {
    /// A leaf holding a fixed constant.
    pub fn new_const(v: f64) -> Self {
        debug_assert!(v.is_finite(), "a constant leaf must be finite");
        debug_assert!(
            !v.is_nan(),
            "a NaN constant would make every fingerprint of this leaf collide"
        );
        Expr {
            nodes: vec![Node::Const(v)],
            n_vars: 0,
        }
    }

    /// A leaf reading input variable `idx`.
    pub fn new_var(idx: usize) -> Self {
        debug_assert!(
            idx < crate::guard::MAX_VARS,
            "the variable index must be nameable"
        );
        debug_assert!(idx < usize::MAX, "the n_vars increment must not overflow");
        Expr {
            nodes: vec![Node::Var(idx)],
            n_vars: idx + 1,
        }
    }

    /// Node count, used as the complexity measure.
    pub fn complexity(&self) -> usize {
        debug_assert!(
            !self.nodes.is_empty(),
            "an expression must have at least one node"
        );
        debug_assert!(
            self.nodes.len() <= MAX_NODES,
            "expressions are capped at MAX_NODES"
        );
        self.nodes.len()
    }

    /// Evaluate at a single point. `vars` must have length >= `n_vars`.
    ///
    /// Returns `None` whenever the expression leaves its domain (log of a
    /// non-positive number, division by an underflowing divisor, a non-finite
    /// intermediate) so the search can discard the candidate.
    pub fn eval(&self, vars: &[f64], params: &[f64]) -> Option<f64> {
        debug_assert!(
            !self.nodes.is_empty(),
            "an expression must have at least one node"
        );
        debug_assert!(
            self.nodes.len() <= MAX_NODES,
            "the builder caps expressions at MAX_NODES"
        );
        let mut stack: Vec<f64> = Vec::with_capacity(self.nodes.len());
        for node in &self.nodes {
            match node {
                Node::Const(v) => stack.push(*v),
                Node::Param(id, init) => {
                    let v = params.get(*id).copied().unwrap_or(*init);
                    stack.push(v);
                }
                Node::Var(i) => {
                    stack.push(*vars.get(*i)?);
                }
                Node::Op(op) => {
                    if stack.len() < op.arity() {
                        return None;
                    }
                    let result = match op {
                        Op::Neg => {
                            let a = stack.pop()?;
                            -a
                        }
                        Op::Inv => {
                            let a = stack.pop()?;
                            if a.abs() < 1e-300 {
                                return None;
                            }
                            1.0 / a
                        }
                        Op::Exp => {
                            let a = stack.pop()?;
                            a.exp()
                        }
                        Op::Ln => {
                            let a = stack.pop()?;
                            if a <= 0.0 {
                                return None;
                            }
                            a.ln()
                        }
                        Op::Sqrt => {
                            let a = stack.pop()?;
                            if a < 0.0 {
                                return None;
                            }
                            a.sqrt()
                        }
                        Op::Sin => {
                            let a = stack.pop()?;
                            a.sin()
                        }
                        Op::Cos => {
                            let a = stack.pop()?;
                            a.cos()
                        }
                        Op::Abs => {
                            let a = stack.pop()?;
                            a.abs()
                        }
                        Op::Eml => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            // guard::eml asserts on NaN, so screen it here.
                            if a.is_nan() || b.is_nan() {
                                return None;
                            }
                            eml(a, b)
                        }
                        Op::Add => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            a + b
                        }
                        Op::Sub => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            a - b
                        }
                        Op::Mul => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            a * b
                        }
                        Op::Div => {
                            let b = stack.pop()?;
                            let a = stack.pop()?;
                            if b.abs() < 1e-300 {
                                return None;
                            }
                            a / b
                        }
                    };
                    if !result.is_finite() {
                        return None;
                    }
                    stack.push(result);
                }
            }
        }
        if stack.len() == 1 {
            Some(stack[0])
        } else {
            None
        }
    }

    /// Evaluate at all data points. Returns `None` if any point fails.
    ///
    /// `data[0]` was indexed unconditionally, so an empty column list aborted
    /// the process with a Rust panic that reached Python as an uncatchable
    /// `PanicException`. `find_formula` now rejects that input up front; this
    /// guard keeps the invariant local to the evaluator too.
    pub fn eval_batch(&self, data: &[Vec<f64>], params: &[f64]) -> Option<Vec<f64>> {
        let n = data.first()?.len();
        debug_assert!(n <= MAX_SAMPLES, "callers must cap the sample count");
        debug_assert!(
            data.iter().all(|c| c.len() == n),
            "callers must reject ragged columns before evaluating"
        );
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let vars: Vec<f64> = data.iter().map(|col| col[i]).collect();
            out.push(self.eval(&vars, params)?);
        }
        Some(out)
    }

    /// Human-readable infix representation.
    ///
    /// Deliberately an inherent method rather than a `Display` impl: the search
    /// calls it directly on every surviving candidate, never through a trait
    /// object, and `Display` would drag in a formatter allocation per node.
    #[allow(clippy::inherent_to_string)]
    pub fn to_string(&self) -> String {
        debug_assert!(
            !self.nodes.is_empty(),
            "an expression must have at least one node"
        );
        debug_assert!(
            self.nodes.len() <= MAX_NODES,
            "expressions are capped at MAX_NODES"
        );
        let mut stack: Vec<String> = Vec::new();
        for node in &self.nodes {
            match node {
                Node::Const(v) => {
                    // ASCII names only: this string reaches Python as
                    // `SearchResult.formula` and is re-parsed by `to_python`.
                    let s = if *v == std::f64::consts::PI {
                        "pi".to_string()
                    } else if *v == std::f64::consts::E {
                        "e".to_string()
                    } else {
                        format!("{v:.4}")
                    };
                    stack.push(s);
                }
                Node::Param(id, v) => stack.push(format!("c{id}({v:.4})")),
                Node::Var(i) => {
                    let name = (b'x' + *i as u8) as char;
                    stack.push(name.to_string());
                }
                Node::Op(op) => match op {
                    Op::Neg => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("(-{a})"));
                    }
                    Op::Inv => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("(1/{a})"));
                    }
                    Op::Exp => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("exp({a})"));
                    }
                    Op::Ln => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("ln({a})"));
                    }
                    Op::Sqrt => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("sqrt({a})"));
                    }
                    Op::Sin => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("sin({a})"));
                    }
                    Op::Cos => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("cos({a})"));
                    }
                    Op::Abs => {
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("abs({a})"));
                    }
                    Op::Eml => {
                        let b = stack.pop().unwrap_or_default();
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("eml({a}, {b})"));
                    }
                    Op::Add => {
                        let b = stack.pop().unwrap_or_default();
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("({a} + {b})"));
                    }
                    Op::Sub => {
                        let b = stack.pop().unwrap_or_default();
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("({a} - {b})"));
                    }
                    Op::Mul => {
                        let b = stack.pop().unwrap_or_default();
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("({a} * {b})"));
                    }
                    Op::Div => {
                        let b = stack.pop().unwrap_or_default();
                        let a = stack.pop().unwrap_or_default();
                        stack.push(format!("({a} / {b})"));
                    }
                },
            }
        }
        stack.pop().unwrap_or_else(|| "?".to_string())
    }

    /// Highest parameter id in this expression, plus one.
    ///
    /// NOTE: the search never constructs a `Node::Param`, so this is always 0
    /// in practice and the Levenberg-Marquardt optimiser is never exercised.
    /// See the module note in `optimizer.rs`.
    #[allow(dead_code)] // Only the unreachable optimiser would call this.
    pub fn param_count(&self) -> usize {
        debug_assert!(
            self.nodes.len() <= MAX_NODES,
            "expressions are capped at MAX_NODES"
        );
        debug_assert!(
            !self.nodes.is_empty(),
            "an expression must have at least one node"
        );
        self.nodes
            .iter()
            .filter_map(|n| {
                if let Node::Param(id, _) = n {
                    Some(*id + 1)
                } else {
                    None
                }
            })
            .max()
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn eml_expr() -> Expr {
        Expr {
            nodes: vec![Node::Var(0), Node::Var(0), Node::Op(Op::Eml)],
            n_vars: 1,
        }
    }

    #[test]
    fn eml_op_matches_the_shared_guard() {
        let e = eml_expr();
        for &x in &[0.5, 1.0, 2.0, 709.7813] {
            assert_eq!(e.eval(&[x], &[]), Some(eml(x, x)));
        }
    }

    #[test]
    fn eval_batch_on_empty_data_returns_none_instead_of_panicking() {
        assert_eq!(eml_expr().eval_batch(&[], &[]), None);
    }

    #[test]
    fn domain_violations_return_none() {
        let ln = Expr {
            nodes: vec![Node::Var(0), Node::Op(Op::Ln)],
            n_vars: 1,
        };
        assert_eq!(ln.eval(&[-1.0], &[]), None);
        assert_eq!(ln.eval(&[std::f64::consts::E], &[]), Some(1.0));
    }

    #[test]
    fn the_printer_emits_ascii_only() {
        let e = Expr {
            nodes: vec![
                Node::Const(std::f64::consts::PI),
                Node::Var(0),
                Node::Op(Op::Add),
            ],
            n_vars: 1,
        };
        let s = e.to_string();
        assert!(s.is_ascii(), "formula must be ASCII: {s}");
        assert_eq!(s, "(pi + x)");
    }

    #[test]
    fn arity_agrees_with_the_evaluator() {
        for op in [
            Op::Neg,
            Op::Inv,
            Op::Exp,
            Op::Ln,
            Op::Sqrt,
            Op::Sin,
            Op::Cos,
            Op::Abs,
        ] {
            assert_eq!(op.arity(), 1);
        }
        for op in [Op::Eml, Op::Add, Op::Sub, Op::Mul, Op::Div] {
            assert_eq!(op.arity(), 2);
        }
    }

    #[test]
    fn an_unbalanced_stack_yields_none() {
        let bad = Expr {
            nodes: vec![Node::Var(0), Node::Var(0)],
            n_vars: 1,
        };
        assert_eq!(bad.eval(&[1.0], &[]), None);
    }
}
