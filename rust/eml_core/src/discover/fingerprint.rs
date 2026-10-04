//! Behavioural fingerprints, used to deduplicate the BFS frontier.
//!
//! Two structurally different expressions that agree on every probe are
//! numerically the same function, so only the first is kept. The probes are
//! mutually transcendental so an accidental collision is not an algebraic
//! identity in disguise.

use dashmap::DashMap;

use crate::discover::expr::Expr;

/// Five algebraically independent probe values, named in Latin letters.
const PROBES: [f64; 5] = [
    0.577_215_664_901_532_9, // Euler-Mascheroni gamma
    1.282_427_129_100_622_6, // Glaisher-Kinkelin A
    1.618_033_988_749_895,   // golden ratio phi
    std::f64::consts::LN_2,  // ln(2)
    1.202_056_903_159_594_2, // Apery zeta(3)
];

/// Number of `i64` slots in a fingerprint: two per probe.
const SLOTS: usize = 2 * PROBES.len();

/// The (high, low) halves of the IEEE-754 bit pattern at each probe.
#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct Fingerprint([i64; SLOTS]);

/// Evaluate `expr` at every probe and pack the results.
///
/// Returns `None` if the expression leaves its domain at any probe, which also
/// discards it from the search.
pub fn fingerprint(expr: &Expr, n_vars: usize, params: &[f64]) -> Option<Fingerprint> {
    debug_assert!(
        n_vars <= crate::guard::MAX_VARS,
        "callers must cap the variable count"
    );
    debug_assert_eq!(
        SLOTS,
        2 * PROBES.len(),
        "each probe contributes exactly two slots"
    );
    let mut parts = [0i64; SLOTS];
    // Bounded: PROBES has a fixed length known at compile time.
    for (k, &probe) in PROBES.iter().enumerate() {
        let vars: Vec<f64> = (0..n_vars).map(|i| probe * (i + 1) as f64).collect();
        let v = expr.eval(&vars, params)?;
        if !v.is_finite() {
            return None;
        }
        let bits = v.to_bits() as i64;
        parts[2 * k] = bits >> 32;
        parts[2 * k + 1] = bits & 0xFFFF_FFFF;
    }
    Some(Fingerprint(parts))
}

/// Concurrent set of fingerprints already visited.
pub type Seen = DashMap<Fingerprint, ()>;

/// Record `fp` and report whether it had not been seen before.
pub fn is_new(seen: &Seen, fp: Fingerprint) -> bool {
    debug_assert!(
        seen.len() < usize::MAX,
        "the seen set must have insertion headroom"
    );
    debug_assert_eq!(
        fp.0.len(),
        SLOTS,
        "a fingerprint always carries every probe"
    );
    seen.insert(fp, ()).is_none()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::expr::{Node, Op};

    #[test]
    fn probes_are_distinct_and_finite() {
        for (i, a) in PROBES.iter().enumerate() {
            assert!(a.is_finite() && *a > 0.0);
            for b in PROBES.iter().skip(i + 1) {
                assert!((a - b).abs() > 1e-6, "probes must be well separated");
            }
        }
    }

    #[test]
    fn equivalent_expressions_share_a_fingerprint() {
        // x and (x + 0) are the same function.
        let x = Expr::new_var(0);
        let x_plus_zero = Expr {
            nodes: vec![Node::Var(0), Node::Const(0.0), Node::Op(Op::Add)],
            n_vars: 1,
        };
        assert_eq!(fingerprint(&x, 1, &[]), fingerprint(&x_plus_zero, 1, &[]));
    }

    #[test]
    fn different_expressions_do_not_share_a_fingerprint() {
        let x = Expr::new_var(0);
        let exp_x = Expr {
            nodes: vec![Node::Var(0), Node::Op(Op::Exp)],
            n_vars: 1,
        };
        assert_ne!(fingerprint(&x, 1, &[]), fingerprint(&exp_x, 1, &[]));
    }

    #[test]
    fn out_of_domain_expressions_have_no_fingerprint() {
        // ln of a negative probe is undefined at every probe.
        let ln_neg = Expr {
            nodes: vec![Node::Var(0), Node::Op(Op::Neg), Node::Op(Op::Ln)],
            n_vars: 1,
        };
        assert!(fingerprint(&ln_neg, 1, &[]).is_none());
    }

    #[test]
    fn is_new_reports_only_the_first_insertion() {
        let seen: Seen = Seen::new();
        let fp = fingerprint(&Expr::new_var(0), 1, &[]).unwrap();
        assert!(is_new(&seen, fp.clone()));
        assert!(!is_new(&seen, fp));
    }
}
