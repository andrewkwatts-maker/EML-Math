use pyo3::prelude::*;

mod point;
mod pair;
mod knot;
mod discover;

use point::EMLPoint;
use pair::{EMLPair, schrodinger_step_n, rotate_phase_n};
use knot::{EMLKnot, simulate_pulses_n};
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
    // Formula discovery
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(find_formula, m)?)?;
    Ok(())
}
