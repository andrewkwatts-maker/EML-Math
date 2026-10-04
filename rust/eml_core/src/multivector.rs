//! Clifford-algebra geometric product over bitmask-encoded basis blades.
//!
//! Two panics used to reach the caller from here as an uncatchable
//! `PanicException`: a `signature` shorter than the algebra indexed
//! `signature[bit_pos]` out of range, and operands of different lengths indexed
//! `b[j]` out of range. Both are now rejected as `ValueError` before any
//! indexing happens.

use rayon::prelude::*;

use crate::guard::{check_len, check_pair, InputError, MAX_BATCH_LEN};

/// Largest number of basis vectors accepted, so `2^n` stays a sane allocation.
///
/// A blade mask is a `usize` bit pattern, and the coefficient array has `2^n`
/// entries; 16 basis vectors is already a 65536-element multivector.
pub const MAX_BASIS_VECTORS: usize = 16;

/// Compute `e_A * e_B` for basis blades encoded as bitmasks.
///
/// Returns `(sign, result_mask)`. The sign accumulates one swap per pair of
/// generators that must cross, plus the metric factor for repeated generators.
fn blade_product(a_mask: usize, b_mask: usize, signature: &[i8]) -> (f64, usize) {
    debug_assert!(
        a_mask < (1usize << signature.len()),
        "the left blade mask must fit the algebra's generator count"
    );
    debug_assert!(
        b_mask < (1usize << signature.len()),
        "the right blade mask must fit the algebra's generator count"
    );
    let mut sign = 1.0f64;
    let result = a_mask ^ b_mask;

    // Bounded: each iteration clears one set bit of `b`, so this runs at most
    // `usize::BITS` times and in practice at most `signature.len()` times.
    let mut b = b_mask;
    while b != 0 {
        let lsb = b & b.wrapping_neg();
        let bit_pos = lsb.trailing_zeros() as usize;
        debug_assert!(
            bit_pos < signature.len(),
            "callers must validate the signature length"
        );

        // Every generator of `a` above this position must be crossed.
        if (a_mask >> (bit_pos + 1)).count_ones() % 2 == 1 {
            sign = -sign;
        }
        // A generator present in both squares to its metric value.
        if a_mask & lsb != 0 {
            sign *= signature[bit_pos] as f64;
        }
        b &= b - 1;
    }

    (sign, result)
}

/// Geometric product of two multivectors of the same dimension.
fn geometric_product_single(a: &[f64], b: &[f64], signature: &[i8]) -> Vec<f64> {
    debug_assert_eq!(a.len(), b.len(), "callers must validate operand dimensions");
    debug_assert_eq!(
        a.len(),
        1usize << signature.len(),
        "a multivector has 2^n coefficients for n generators"
    );
    let dim = a.len();
    let mut result = vec![0.0f64; dim];
    // The index *is* the blade mask, not a mere cursor, so both loops read it
    // from `enumerate` rather than a bare range.
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj == 0.0 {
                continue;
            }
            let (sign, k) = blade_product(i, j, signature);
            debug_assert!(k < dim, "the XOR of two in-range masks stays in range");
            result[k] += sign * ai * bj;
        }
    }
    result
}

/// Reject a batch whose shapes do not describe a Clifford algebra.
fn validate_shapes(
    a_batch: &[Vec<f64>],
    b_batch: &[Vec<f64>],
    signature: &[i8],
) -> Result<(), InputError> {
    debug_assert!(
        signature.len() <= usize::BITS as usize,
        "a signature longer than the word size cannot index a blade mask"
    );
    debug_assert!(
        a_batch.len() <= usize::MAX / 2,
        "the batch length must be plausible before it is checked"
    );
    check_pair(a_batch.len(), b_batch.len())?;
    if signature.is_empty() {
        return Err(InputError::Empty { what: "signature" });
    }
    check_len(signature.len(), MAX_BASIS_VECTORS)?;
    if signature.iter().any(|&s| s != 1 && s != -1 && s != 0) {
        return Err(InputError::Invalid {
            what: "signature entries must each be -1, 0 or 1",
        });
    }
    let dim = 1usize << signature.len();
    // Bounded: a_batch.len() was capped by check_pair above.
    for (a, b) in a_batch.iter().zip(b_batch.iter()) {
        if a.len() != dim || b.len() != dim {
            return Err(InputError::Invalid {
                what: "every multivector must have exactly 2^len(signature) coefficients",
            });
        }
    }
    Ok(())
}

/// Batch geometric product over a shared metric signature.
pub fn geometric_product_n(
    a_batch: Vec<Vec<f64>>,
    b_batch: Vec<Vec<f64>>,
    signature: Vec<i8>,
) -> Result<Vec<Vec<f64>>, InputError> {
    validate_shapes(&a_batch, &b_batch, &signature)?;
    debug_assert_eq!(
        a_batch.len(),
        b_batch.len(),
        "validate_shapes pairs the batches"
    );
    debug_assert!(
        a_batch.len() <= MAX_BATCH_LEN,
        "validate_shapes enforces the batch cap"
    );
    Ok(a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| geometric_product_single(a, b, &signature))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cl(1,1): generators e1 with e1^2 = +1 and e2 with e2^2 = -1.
    const MINKOWSKI_2D: [i8; 2] = [1, -1];

    #[test]
    fn generators_square_to_their_signature() {
        // e1 is mask 0b01, e2 is mask 0b10; the scalar slot is index 0.
        let e1 = vec![0.0, 1.0, 0.0, 0.0];
        let e2 = vec![0.0, 0.0, 1.0, 0.0];
        let s1 = geometric_product_single(&e1, &e1, &MINKOWSKI_2D);
        let s2 = geometric_product_single(&e2, &e2, &MINKOWSKI_2D);
        assert_eq!(s1[0], 1.0);
        assert_eq!(s2[0], -1.0);
    }

    #[test]
    fn distinct_generators_anticommute() {
        let e1 = vec![0.0, 1.0, 0.0, 0.0];
        let e2 = vec![0.0, 0.0, 1.0, 0.0];
        let ab = geometric_product_single(&e1, &e2, &MINKOWSKI_2D);
        let ba = geometric_product_single(&e2, &e1, &MINKOWSKI_2D);
        for k in 0..4 {
            assert_eq!(ab[k], -ba[k], "slot {k} must anticommute");
        }
    }

    #[test]
    fn the_scalar_one_is_an_identity() {
        let one = vec![1.0, 0.0, 0.0, 0.0];
        let v = vec![0.5, 1.0, -2.0, 3.0];
        assert_eq!(geometric_product_single(&one, &v, &MINKOWSKI_2D), v);
        assert_eq!(geometric_product_single(&v, &one, &MINKOWSKI_2D), v);
    }

    #[test]
    fn a_short_signature_is_an_error_not_a_panic() {
        // This combination used to abort with an index-out-of-bounds panic.
        let a = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let b = vec![vec![1.0, 0.0, 0.0, 0.0]];
        assert!(geometric_product_n(a.clone(), b.clone(), vec![1]).is_err());
        assert!(geometric_product_n(a, b, vec![1, 1]).is_ok());
    }

    #[test]
    fn mismatched_operand_dimensions_are_an_error_not_a_panic() {
        let a = vec![vec![1.0, 0.0, 0.0, 0.0]];
        let b = vec![vec![1.0, 0.0]];
        assert!(geometric_product_n(a, b, vec![1, 1]).is_err());
    }

    #[test]
    fn an_oversized_signature_is_rejected() {
        let sig = vec![1i8; MAX_BASIS_VECTORS + 1];
        assert!(geometric_product_n(vec![vec![1.0]], vec![vec![1.0]], sig).is_err());
    }
}
