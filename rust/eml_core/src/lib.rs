use pyo3::prelude::*;
use rayon::prelude::*;

mod point;
mod pair;
mod knot;
mod discover;
mod metric;
mod octonion;
mod multivector;

use point::EMLPoint;
use pair::{EMLPair, schrodinger_step_n, rotate_phase_n};
use knot::{EMLKnot, simulate_pulses_n};
use metric::christoffel_batch_n;
use octonion::octonion_mul_n;
use multivector::geometric_product_n;
use discover::search::{search, SearchConfig};

/// Python-facing formula discovery result.
#[pyclass(module = "eml_core")]
struct PySearchResult {
    #[pyo3(get)]
    formula: String,
    #[pyo3(get)]
    error: f64,
    #[pyo3(get)]
    complexity: usize,
    #[pyo3(get)]
    params: Vec<f64>,
}

#[pymethods]
impl PySearchResult {
    pub fn to_latex(&self) -> String {
        // Simple substitutions — a full LaTeX renderer would be more elaborate
        self.formula
            .replace("exp(", r"\exp(")
            .replace("ln(", r"\ln(")
            .replace("sqrt(", r"\sqrt{")
            .replace("sin(", r"\sin(")
            .replace("cos(", r"\cos(")
            .replace("eml(", r"\mathrm{eml}(")
            .replace("pi", r"\pi")
    }

    pub fn to_python(&self) -> String {
        let expr = self.formula
            .replace("exp(", "math.exp(")
            .replace("ln(", "math.log(")
            .replace("sqrt(", "math.sqrt(")
            .replace("sin(", "math.sin(")
            .replace("cos(", "math.cos(");
        format!("import math\nf = lambda x: {}", expr)
    }

    pub fn __repr__(&self) -> String {
        format!(
            "SearchResult(formula='{}', error={:.2e}, complexity={})",
            self.formula, self.error, self.complexity
        )
    }
}

/// Find a formula fitting x_data → y_data.
/// x_data: list of column vectors (one per variable), or a single 1D list for univariate.
/// y_data: target values (1D list).
#[pyfunction]
#[pyo3(signature = (x_data, y_data, max_complexity=8, beam_width=2000, precision_goal=1e-10, use_trig=true, use_eml=true, complexity_penalty=0.001))]
fn find_formula(
    x_data: Vec<Vec<f64>>,
    y_data: Vec<f64>,
    max_complexity: usize,
    beam_width: usize,
    precision_goal: f64,
    use_trig: bool,
    use_eml: bool,
    complexity_penalty: f64,
) -> Option<PySearchResult> {
    let n_vars = x_data.len();
    let config = SearchConfig {
        max_complexity,
        beam_width,
        precision_goal,
        complexity_penalty,
        use_trig,
        use_eml_primitive: use_eml,
    };
    search(&x_data, &y_data, n_vars, &config).map(|r| PySearchResult {
        formula: r.formula,
        error: r.error,
        complexity: r.complexity,
        params: r.params,
    })
}

/// Batch Lorentz boost: apply boost(phi_i, c) to each (x_i, y_i) point in parallel.
/// Returns Vec of (x', y') tuples.
#[pyfunction]
fn boost_n(points: Vec<(f64, f64)>, phis: Vec<f64>, c: f64) -> Vec<(f64, f64)> {
    points
        .par_iter()
        .zip(phis.par_iter())
        .map(|((x, y), phi)| {
            let p = EMLPoint::new(*x, *y);
            let b = p.boost(*phi, c);
            (b.x, b.y)
        })
        .collect()
}

// ── Batch arithmetic operators (Rayon parallel) ───────────────────────────────

const OVERFLOW_THRESHOLD: f64 = 709.78;

#[inline]
fn y_safe(y: f64) -> f64 { if y <= 0.0 { y.abs().max(1e-300) } else { y } }

#[inline]
fn xv_safe(x: f64) -> f64 { if x > OVERFLOW_THRESHOLD { x.ln() } else { x } }

/// Batch exp: [exp(x) for x in xs] — Rayon parallel.
#[pyfunction]
fn exp_n(xs: Vec<f64>) -> Vec<f64> {
    xs.par_iter().map(|&x| xv_safe(x).exp()).collect()
}

/// Batch ln: [ln(y) for y in ys] — Rayon parallel, frame-shift guard applied.
#[pyfunction]
fn ln_n(ys: Vec<f64>) -> Vec<f64> {
    ys.par_iter().map(|&y| y_safe(y).ln()).collect()
}

/// Batch add: [a + b for (a, b) in zip(as_, bs)] — Rayon parallel.
#[pyfunction]
fn add_n(as_: Vec<f64>, bs: Vec<f64>) -> Vec<f64> {
    as_.par_iter().zip(bs.par_iter()).map(|(a, b)| a + b).collect()
}

/// Batch sub: [a - b for (a, b) in zip(as_, bs)] — Rayon parallel.
#[pyfunction]
fn sub_n(as_: Vec<f64>, bs: Vec<f64>) -> Vec<f64> {
    as_.par_iter().zip(bs.par_iter()).map(|(a, b)| a - b).collect()
}

/// Batch mul: [a * b for (a, b) in zip(as_, bs)] — Rayon parallel.
#[pyfunction]
fn mul_n(as_: Vec<f64>, bs: Vec<f64>) -> Vec<f64> {
    as_.par_iter().zip(bs.par_iter()).map(|(a, b)| a * b).collect()
}

/// Batch div: [a / b] — NaN for |b| < 1e-300 — Rayon parallel.
#[pyfunction]
fn div_n(as_: Vec<f64>, bs: Vec<f64>) -> Vec<f64> {
    as_.par_iter().zip(bs.par_iter()).map(|(a, b)| {
        if b.abs() < 1e-300 { f64::NAN } else { a / b }
    }).collect()
}

/// Batch sqrt: [sqrt(|x|) for x in xs] — Rayon parallel.
#[pyfunction]
fn sqrt_n(xs: Vec<f64>) -> Vec<f64> {
    xs.par_iter().map(|&x| x.abs().sqrt()).collect()
}

/// Batch sin: [sin(x) for x in xs] — Rayon parallel.
#[pyfunction]
fn sin_n(xs: Vec<f64>) -> Vec<f64> {
    xs.par_iter().map(|&x| x.sin()).collect()
}

/// Batch cos: [cos(x) for x in xs] — Rayon parallel.
#[pyfunction]
fn cos_n(xs: Vec<f64>) -> Vec<f64> {
    xs.par_iter().map(|&x| x.cos()).collect()
}

/// Batch eml_tension: [eml(x, y) for (x, y) in zip(xs, ys)] — Rayon parallel.
#[pyfunction]
fn tension_n(xs: Vec<f64>, ys: Vec<f64>) -> Vec<f64> {
    xs.par_iter().zip(ys.par_iter()).map(|(&x, &y)| {
        xv_safe(x).exp() - y_safe(y).ln()
    }).collect()
}

/// Batch pow: [|base|^exp for (base, exp) in zip(bases, exps)] — Rayon parallel.
#[pyfunction]
fn pow_n(bases: Vec<f64>, exps: Vec<f64>) -> Vec<f64> {
    bases.par_iter().zip(exps.par_iter()).map(|(b, e)| b.abs().powf(*e)).collect()
}

#[pymodule]
fn eml_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<EMLPoint>()?;
    m.add_class::<EMLPair>()?;
    m.add_class::<EMLKnot>()?;
    // Batch functions
    m.add_function(wrap_pyfunction!(schrodinger_step_n, m)?)?;
    m.add_function(wrap_pyfunction!(rotate_phase_n, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_pulses_n, m)?)?;
    m.add_function(wrap_pyfunction!(boost_n, m)?)?;
    // Batch arithmetic operators
    m.add_function(wrap_pyfunction!(exp_n, m)?)?;
    m.add_function(wrap_pyfunction!(ln_n, m)?)?;
    m.add_function(wrap_pyfunction!(add_n, m)?)?;
    m.add_function(wrap_pyfunction!(sub_n, m)?)?;
    m.add_function(wrap_pyfunction!(mul_n, m)?)?;
    m.add_function(wrap_pyfunction!(div_n, m)?)?;
    m.add_function(wrap_pyfunction!(sqrt_n, m)?)?;
    m.add_function(wrap_pyfunction!(sin_n, m)?)?;
    m.add_function(wrap_pyfunction!(cos_n, m)?)?;
    m.add_function(wrap_pyfunction!(tension_n, m)?)?;
    m.add_function(wrap_pyfunction!(pow_n, m)?)?;
    m.add_function(wrap_pyfunction!(christoffel_batch_n, m)?)?;
    m.add_function(wrap_pyfunction!(octonion_mul_n, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_product_n, m)?)?;
    // Formula discovery
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(find_formula, m)?)?;
    Ok(())
}
