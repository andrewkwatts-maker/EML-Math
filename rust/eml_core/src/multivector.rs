use pyo3::prelude::*;
use rayon::prelude::*;

/// Compute e_A * e_B for basis blades encoded as bitmasks.
/// Returns (sign, result_mask).
/// signature: array of +1/-1 values, one per basis vector (up to 8).
fn blade_product(a_mask: usize, b_mask: usize, signature: &[i8]) -> (f64, usize) {
    let mut sign = 1.0f64;
    let result = a_mask ^ b_mask;

    let mut b = b_mask;
    while b != 0 {
        let lsb = b & b.wrapping_neg();
        let bit_pos = lsb.trailing_zeros() as usize;

        // Count set bits in a_mask strictly above bit_pos
        let above = (a_mask >> (bit_pos + 1)).count_ones();
        if above % 2 == 1 {
            sign = -sign;
        }

        // If bit was in both a and b: apply metric signature
        if a_mask & lsb != 0 {
            sign *= signature[bit_pos] as f64;
        }

        b &= b - 1;
    }

    (sign, result)
}

/// Single geometric product of two multivectors with given signature.
/// components_a, components_b: coefficient arrays of length 2^n.
/// signature: +1/-1 array of length n.
fn geometric_product_single(
    a: &[f64],
    b: &[f64],
    signature: &[i8],
) -> Vec<f64> {
    let dim = a.len();
    let mut result = vec![0.0f64; dim];
    for i in 0..dim {
        if a[i] == 0.0 { continue; }
        for j in 0..dim {
            if b[j] == 0.0 { continue; }
            let (sign, k) = blade_product(i, j, signature);
            result[k] += sign * a[i] * b[j];
        }
    }
    result
}

/// Batch geometric product (Rayon parallel).
/// a_batch, b_batch: each is a Vec of flat coefficient arrays.
/// signature: +1/-1 per basis vector (length n, algebra dimension = 2^n).
/// Returns Vec of result coefficient arrays.
#[pyfunction]
pub fn geometric_product_n(
    a_batch: Vec<Vec<f64>>,
    b_batch: Vec<Vec<f64>>,
    signature: Vec<i8>,
) -> Vec<Vec<f64>> {
    a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| geometric_product_single(a, b, &signature))
        .collect()
}
