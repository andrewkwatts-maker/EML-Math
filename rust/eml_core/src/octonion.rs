//! Octonion multiplication over the Fano-plane table.

use rayon::prelude::*;

use crate::guard::{check_pair, InputError, MAX_BATCH_LEN};

/// Number of octonion basis elements, `e_0` through `e_7`.
pub const DIM: usize = 8;

// Fano-plane multiplication table.
// TABLE[i][j] = (sign, result_index) for e_i * e_j.
// e_0 is the real unit. e_1..e_7 follow the lines (124)(235)(346)(457)(156)(267)(137).
const TABLE: [[(i8, usize); 8]; 8] = build_table();

const fn fano_lines() -> [(usize, usize, usize); 7] {
    [
        (1, 2, 4),
        (2, 3, 5),
        (3, 4, 6),
        (4, 5, 7),
        (1, 5, 6),
        (2, 6, 7),
        (1, 3, 7),
    ]
}

const fn build_table() -> [[(i8, usize); 8]; 8] {
    let mut t = [[(0i8, 0usize); 8]; 8];

    // e_0 is identity
    let mut i = 0;
    while i < 8 {
        t[0][i] = (1, i);
        t[i][0] = (1, i);
        i += 1;
    }

    // e_i * e_i = -e_0 for i > 0
    let mut i = 1;
    while i < 8 {
        t[i][i] = (-1, 0);
        i += 1;
    }

    // Fill from Fano lines
    let lines = fano_lines();
    let mut li = 0;
    while li < 7 {
        let (a, b, c) = lines[li];
        t[a][b] = (1, c);
        t[b][a] = (-1, c);
        t[b][c] = (1, a);
        t[c][b] = (-1, a);
        t[c][a] = (1, b);
        t[a][c] = (-1, b);
        li += 1;
    }

    t
}

/// Multiply two octonions given as coefficient arrays over `e_0..e_7`.
///
/// Both loops are fixed at [`DIM`]; the zero-skip is a speed optimisation only,
/// never a bound.
#[inline]
pub fn mul_octonion(a: &[f64; DIM], b: &[f64; DIM]) -> [f64; DIM] {
    debug_assert_eq!(a.len(), DIM, "an octonion has exactly eight components");
    debug_assert_eq!(
        TABLE.len(),
        DIM,
        "the Fano table must cover every basis pair"
    );
    let mut result = [0.0f64; DIM];
    // The index is the basis-element number, so it is read from `enumerate`.
    for (i, &ai) in a.iter().enumerate() {
        if ai == 0.0 {
            continue;
        }
        for (j, &bj) in b.iter().enumerate() {
            if bj == 0.0 {
                continue;
            }
            let (sign, k) = TABLE[i][j];
            debug_assert!(k < DIM, "the Fano table must only name real basis indices");
            result[k] += (sign as f64) * ai * bj;
        }
    }
    result
}

/// Batch octonion multiplication.
///
/// `zip` previously truncated to the shorter batch, so a caller passing 100
/// left operands and 99 right ones silently got 99 products back.
pub fn octonion_mul_n(
    a_batch: Vec<[f64; DIM]>,
    b_batch: Vec<[f64; DIM]>,
) -> Result<Vec<[f64; DIM]>, InputError> {
    check_pair(a_batch.len(), b_batch.len())?;
    debug_assert_eq!(
        a_batch.len(),
        b_batch.len(),
        "check_pair guarantees equal lengths"
    );
    debug_assert!(
        a_batch.len() <= MAX_BATCH_LEN,
        "check_pair enforces the batch cap"
    );
    Ok(a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| mul_octonion(a, b))
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(i: usize) -> [f64; DIM] {
        let mut v = [0.0; DIM];
        v[i] = 1.0;
        v
    }

    #[test]
    fn the_real_unit_is_a_two_sided_identity() {
        for i in 0..DIM {
            assert_eq!(mul_octonion(&unit(0), &unit(i)), unit(i));
            assert_eq!(mul_octonion(&unit(i), &unit(0)), unit(i));
        }
    }

    #[test]
    fn imaginary_units_square_to_minus_one() {
        let mut minus_one = [0.0; DIM];
        minus_one[0] = -1.0;
        for i in 1..DIM {
            assert_eq!(mul_octonion(&unit(i), &unit(i)), minus_one, "e_{i} squared");
        }
    }

    #[test]
    fn imaginary_units_anticommute() {
        for i in 1..DIM {
            for j in 1..DIM {
                if i == j {
                    continue;
                }
                let ab = mul_octonion(&unit(i), &unit(j));
                let ba = mul_octonion(&unit(j), &unit(i));
                for k in 0..DIM {
                    assert_eq!(ab[k], -ba[k], "e_{i} e_{j} must anticommute at slot {k}");
                }
            }
        }
    }

    #[test]
    fn the_norm_is_multiplicative() {
        let a = [1.0, 2.0, -1.0, 0.5, 0.0, 3.0, -2.0, 1.0];
        let b = [0.0, 1.0, 1.0, -1.0, 2.0, 0.0, 1.0, -3.0];
        let n = |v: &[f64; DIM]| v.iter().map(|x| x * x).sum::<f64>();
        let prod = mul_octonion(&a, &b);
        assert!((n(&prod) - n(&a) * n(&b)).abs() < 1e-9);
    }

    #[test]
    fn the_batch_wrapper_rejects_a_length_mismatch() {
        assert!(octonion_mul_n(vec![unit(1), unit(2)], vec![unit(1)]).is_err());
        assert!(octonion_mul_n(vec![unit(1)], vec![unit(1)]).is_ok());
    }
}
