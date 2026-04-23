use dashmap::DashMap;
use ordered_float::OrderedFloat;
use crate::discover::expr::Expr;

// Five algebraically independent probe values (irrational, mutually transcendental).
const PROBES: [f64; 5] = [
    0.5772156649015328,  // Euler-Mascheroni γ
    1.2824271291006226,  // Glaisher-Kinkelin A
    1.6180339887498949,  // Golden ratio φ
    0.6931471805599453,  // ln(2)
    1.2020569031595942,  // Apéry ζ(3)
];

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct Fingerprint([i64; 10]); // (mantissa, exponent) × 5 probes

pub fn fingerprint(expr: &Expr, n_vars: usize, params: &[f64]) -> Option<Fingerprint> {
    let mut parts = [0i64; 10];
    for (k, &probe) in PROBES.iter().enumerate() {
        let vars: Vec<f64> = (0..n_vars).map(|i| probe * (i + 1) as f64).collect();
        let v = expr.eval(&vars, params)?;
        if !v.is_finite() { return None; }
        let bits = v.to_bits() as i64;
        parts[2 * k] = bits >> 32;
        parts[2 * k + 1] = bits & 0xFFFF_FFFF;
    }
    Some(Fingerprint(parts))
}

pub type Seen = DashMap<Fingerprint, ()>;

pub fn is_new(seen: &Seen, fp: Fingerprint) -> bool {
    seen.insert(fp, ()).is_none()
}
