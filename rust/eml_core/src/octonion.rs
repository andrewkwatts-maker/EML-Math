use pyo3::prelude::*;
use rayon::prelude::*;

// Fano-plane multiplication table.
// TABLE[i][j] = (sign, result_index) for e_i * e_j.
// e_0 is the real unit. e_1..e_7 follow the lines (124)(235)(346)(457)(156)(267)(137).
const TABLE: [[(i8, usize); 8]; 8] = build_table();

const fn fano_lines() -> [(usize, usize, usize); 7] {
    [(1,2,4),(2,3,5),(3,4,6),(4,5,7),(1,5,6),(2,6,7),(1,3,7)]
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

/// Multiply two octonions given as [f64; 8] coefficient arrays.
#[inline]
fn mul_octonion(a: &[f64; 8], b: &[f64; 8]) -> [f64; 8] {
    let mut result = [0.0f64; 8];
    for i in 0..8 {
        if a[i] == 0.0 { continue; }
        for j in 0..8 {
            if b[j] == 0.0 { continue; }
            let (sign, k) = TABLE[i][j];
            result[k] += (sign as f64) * a[i] * b[j];
        }
    }
    result
}

/// Batch octonion multiplication (Rayon parallel).
/// a_batch, b_batch: each is a Vec of 8-element float lists.
/// Returns Vec of 8-element results.
#[pyfunction]
pub fn octonion_mul_n(
    a_batch: Vec<[f64; 8]>,
    b_batch: Vec<[f64; 8]>,
) -> Vec<[f64; 8]> {
    a_batch
        .par_iter()
        .zip(b_batch.par_iter())
        .map(|(a, b)| mul_octonion(a, b))
        .collect()
}
