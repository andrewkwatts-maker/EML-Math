//! `eml_core` -- the Rust implementation behind the `eml-math` Python package.
//!
//! The crate is a complete library on its own; the `python` feature is purely
//! additive and only adds the PyO3 surface. `extension-module` is a *separate*
//! feature that only the wheel build enables, so `cargo test` can still link
//! libpython and actually run the `#[cfg(test)]` tests in these bindings.
//!
//! ## Conventions
//!
//! - **Two runtime assertions minimum** per function, checking preconditions the
//!   Python boundary cannot enforce through types alone.
//! - **Bounded loops.** Every iteration over caller-supplied data checks its
//!   length against [`guard::MAX_BATCH_LEN`] first.
//! - **Errors surface as Python exceptions**, never as a default value, and
//!   never as a silently truncated result. `Iterator::zip` on two sequences of
//!   different lengths returns the shorter one, which turns a caller's
//!   off-by-one into a wrong answer; every paired wrapper below rejects that.
//! - **ASCII only** in identifiers and in every string that reaches a caller.

// PyO3 0.22 expands every `#[pyfunction] -> PyResult<T>` into code that calls
// `Into::into` on an error that is already a `PyErr`. The conversion is in
// macro-generated code we do not own, so the lint is silenced once here rather
// than annotated on ~20 wrappers.
#![allow(clippy::useless_conversion)]

#[cfg(feature = "python")]
use pyo3::exceptions::PyValueError;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use rayon::prelude::*;

pub mod guard;

// Public, so the crate is usable as a Rust library.
//
// These were all private, which meant every public item in `eml_core` was a
// `#[pyfunction]`: with the `python` feature off there was no API at all, and
// the whole crate compiled to dead code. A "standalone Rust core" that nothing
// in Rust can call is not standalone, it is just unreachable.
pub mod discover;
pub mod knot;
pub mod metric;
pub mod multivector;
pub mod octonion;
pub mod pair;
pub mod point;

// The `arithmos_bridge.rs` skeleton in this directory is NOT wired up. It was
// declared behind `#[cfg(feature = "with-arithmos")]`, but that feature was
// never added to `Cargo.toml` and `arithmos_core` was never added to
// `[dependencies]`, so the module -- and the three `#[test]`s inside it -- could
// not compile under any feature combination, including `--all-features`. It is
// left on disk untouched rather than deleted, but it is dead: declaring the
// feature would only turn an invisible defect into a build failure, because the
// crate it imports is absent. The upstream is now `Arithma`, not `arithmos`.

#[cfg(feature = "python")]
use discover::search::{search, SearchConfig};
use guard::{check_len, check_pair, InputError};
#[cfg(feature = "python")]
use guard::{eml, x_guard, y_guard, MAX_BATCH_LEN};
#[cfg(feature = "python")]
use knot::{simulate_pulses_n, EMLKnot};
#[cfg(feature = "python")]
use metric::christoffel_batch_n;
#[cfg(feature = "python")]
use multivector::geometric_product_n;
#[cfg(feature = "python")]
use octonion::octonion_mul_n;
#[cfg(feature = "python")]
use pair::{rotate_phase_n, schrodinger_step_n, EMLPair};
#[cfg(feature = "python")]
use point::EMLPoint;

/// Crate version, kept in lockstep with `eml_math.__version__`.
///
/// `eml_math.assert_rust_backend()` compares the two and refuses to start on a
/// mismatch. A stale `eml_core.pyd` left in the source tree by an earlier build
/// is easy to miss and produces baffling behaviour.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Translate a core-side input rejection into a Python `ValueError`.
/// Bridge the crate's own error into a Python exception.
///
/// Declared once here so the core functions can return `Result<_, InputError>`
/// and stay usable from Rust; `?` converts at the binding boundary.
#[cfg(feature = "python")]
impl From<InputError> for PyErr {
    fn from(err: InputError) -> Self {
        PyValueError::new_err(err.to_string())
    }
}

#[cfg(feature = "python")]
pub(crate) fn to_py_err(err: InputError) -> PyErr {
    let message = err.to_string();
    debug_assert!(
        !message.is_empty(),
        "every InputError must carry a diagnostic message"
    );
    debug_assert!(message.is_ascii(), "diagnostics must stay ASCII");
    PyValueError::new_err(message)
}

/// Version of the compiled extension. Used for the Python version handshake.
#[cfg(feature = "python")]
#[pyfunction]
fn version_rust() -> &'static str {
    // Both checks read a compile-time constant, so they are compile-time facts
    // rather than runtime preconditions; a const block states that honestly and
    // fails the build rather than a debug run.
    const _: () = assert!(!VERSION.is_empty());
    VERSION
}

/// Always `true` in the compiled extension.
///
/// The pure-Python stand-in returns `false`, so callers can branch on the real
/// backend rather than on whether an import happened to succeed.
#[cfg(feature = "python")]
#[pyfunction]
fn is_rust_backend() -> bool {
    true
}

/// Python-facing formula discovery result.
#[cfg(feature = "python")]
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

#[cfg(feature = "python")]
#[pymethods]
impl PySearchResult {
    /// Render the formula as LaTeX.
    ///
    /// Textual substitution, not a parser: it recognises the function names the
    /// search emits and leaves everything else alone.
    pub fn to_latex(&self) -> String {
        debug_assert!(
            !self.formula.is_empty(),
            "a search result must carry a formula"
        );
        debug_assert!(
            self.formula.is_ascii(),
            "the formula printer must emit ASCII only"
        );
        self.formula
            .replace("exp(", r"\exp(")
            .replace("ln(", r"\ln(")
            .replace("sqrt(", r"\sqrt{")
            .replace("sin(", r"\sin(")
            .replace("cos(", r"\cos(")
            .replace("eml(", r"\mathrm{eml}(")
            .replace("pi", r"\pi")
    }

    /// Render the formula as a runnable Python lambda.
    pub fn to_python(&self) -> String {
        debug_assert!(
            !self.formula.is_empty(),
            "a search result must carry a formula"
        );
        debug_assert!(
            self.error.is_finite(),
            "a reported result must have finite error"
        );
        let expr = self
            .formula
            .replace("exp(", "math.exp(")
            .replace("ln(", "math.log(")
            .replace("sqrt(", "math.sqrt(")
            .replace("sin(", "math.sin(")
            .replace("cos(", "math.cos(");
        format!("import math\nf = lambda x: {expr}")
    }

    pub fn __repr__(&self) -> String {
        debug_assert!(
            self.complexity > 0,
            "complexity counts nodes and is at least one"
        );
        debug_assert!(
            !self.error.is_nan(),
            "error must never be NaN in a reported result"
        );
        format!(
            "SearchResult(formula='{}', error={:.2e}, complexity={})",
            self.formula, self.error, self.complexity
        )
    }
}

/// Tunable knobs for [`find_formula`], grouped so the wrapper stays under the
/// argument limit and so Python can build one config and reuse it.
#[cfg_attr(feature = "python", pyclass(module = "eml_core"))]
#[derive(Clone, Debug)]
pub struct SearchOptions {
    pub max_complexity: usize,
    pub beam_width: usize,
    pub precision_goal: f64,
    pub use_trig: bool,
    pub use_eml: bool,
    pub complexity_penalty: f64,
}

impl SearchOptions {
    pub fn new(
        max_complexity: usize,
        beam_width: usize,
        precision_goal: f64,
        use_trig: bool,
        use_eml: bool,
        complexity_penalty: f64,
    ) -> Result<Self, InputError> {
        let opts = SearchOptions {
            max_complexity,
            beam_width,
            precision_goal,
            use_trig,
            use_eml,
            complexity_penalty,
        };
        opts.validate()?;
        Ok(opts)
    }

    pub fn __repr__(&self) -> String {
        debug_assert!(
            self.max_complexity > 0,
            "validated options have positive complexity"
        );
        debug_assert!(
            self.beam_width > 0,
            "validated options have positive beam width"
        );
        format!(
            "SearchOptions(max_complexity={}, beam_width={}, precision_goal={:.3e})",
            self.max_complexity, self.beam_width, self.precision_goal
        )
    }
}

// PyO3's inner attributes are macro syntax, not real attributes, so `cfg_attr`
// cannot produce them. `SearchOptions` above is plain Rust and always
// compiled; this block is its Python view.
#[cfg(feature = "python")]
#[pymethods]
impl SearchOptions {
    #[new]
    #[pyo3(signature = (max_complexity=8, beam_width=2000, precision_goal=1e-10,
                        use_trig=true, use_eml=true, complexity_penalty=0.001))]
    fn py_new(
        max_complexity: usize,
        beam_width: usize,
        precision_goal: f64,
        use_trig: bool,
        use_eml: bool,
        complexity_penalty: f64,
    ) -> Result<Self, InputError> {
        Self::new(
            max_complexity,
            beam_width,
            precision_goal,
            use_trig,
            use_eml,
            complexity_penalty,
        )
    }

    #[getter]
    fn get_max_complexity(&self) -> usize {
        self.max_complexity
    }

    #[getter]
    fn get_beam_width(&self) -> usize {
        self.beam_width
    }

    #[getter]
    fn get_precision_goal(&self) -> f64 {
        self.precision_goal
    }

    #[getter]
    fn get_use_trig(&self) -> bool {
        self.use_trig
    }

    #[getter]
    fn get_use_eml(&self) -> bool {
        self.use_eml
    }

    #[getter]
    fn get_complexity_penalty(&self) -> f64 {
        self.complexity_penalty
    }

    #[pyo3(name = "__repr__")]
    fn py_repr(&self) -> String {
        self.__repr__()
    }
}

impl SearchOptions {
    /// Reject configurations that would loop unboundedly or never terminate.
    pub fn validate(&self) -> Result<(), InputError> {
        // The caps themselves are compile-time facts, checked once at build
        // time; a debug_assert on a constant is folded away and asserts nothing.
        const _: () = assert!(discover::search::MAX_COMPLEXITY_CAP > 0);
        const _: () = assert!(discover::search::MAX_BEAM_WIDTH > 0);
        // No debug_assert on the knobs themselves: this *is* the function that
        // rejects them, so asserting a bound here panics on exactly the input
        // it exists to turn into an `Err`. A validator must not assume its
        // argument is already valid.
        if self.max_complexity == 0 || self.max_complexity > discover::search::MAX_COMPLEXITY_CAP {
            return Err(InputError::Invalid {
                what: "max_complexity must be in 1..=32",
            });
        }
        if self.beam_width == 0 || self.beam_width > discover::search::MAX_BEAM_WIDTH {
            return Err(InputError::Invalid {
                what: "beam_width must be in 1..=100000",
            });
        }
        if !self.precision_goal.is_finite() || self.precision_goal < 0.0 {
            return Err(InputError::Invalid {
                what: "precision_goal must be finite and non-negative",
            });
        }
        if !self.complexity_penalty.is_finite() || self.complexity_penalty < 0.0 {
            return Err(InputError::Invalid {
                what: "complexity_penalty must be finite and non-negative",
            });
        }
        Ok(())
    }
}

/// Find a formula fitting `x_data -> y_data`.
///
/// `x_data` is a list of column vectors, one per input variable; `y_data` holds
/// the targets. Every column must be the same length as `y_data`. Previously an
/// empty `x_data` indexed `data[0]` and aborted the interpreter with a Rust
/// panic; it is now a `ValueError`.
#[cfg(feature = "python")]
#[pyfunction]
#[allow(clippy::too_many_arguments)] // Keeps the historical keyword-argument API.
fn find_formula(
    x_data: Vec<Vec<f64>>,
    y_data: Vec<f64>,
    max_complexity: usize,
    beam_width: usize,
    precision_goal: f64,
    use_trig: bool,
    use_eml: bool,
    complexity_penalty: f64,
) -> PyResult<Option<PySearchResult>> {
    let opts = SearchOptions::new(
        max_complexity,
        beam_width,
        precision_goal,
        use_trig,
        use_eml,
        complexity_penalty,
    )?;
    Ok(find_formula_with(x_data, y_data, &opts)?)
}

#[cfg(feature = "python")]
/// Shared body for [`find_formula`], split out to keep each function short.
fn find_formula_with(
    x_data: Vec<Vec<f64>>,
    y_data: Vec<f64>,
    opts: &SearchOptions,
) -> Result<Option<PySearchResult>, InputError> {
    validate_dataset(&x_data, &y_data)?;
    debug_assert!(
        !x_data.is_empty(),
        "validate_dataset rejects an empty column list"
    );
    debug_assert!(
        x_data.iter().all(|c| c.len() == y_data.len()),
        "validate_dataset rejects ragged columns"
    );

    let config = SearchConfig {
        max_complexity: opts.max_complexity,
        beam_width: opts.beam_width,
        precision_goal: opts.precision_goal,
        complexity_penalty: opts.complexity_penalty,
        use_trig: opts.use_trig,
        use_eml_primitive: opts.use_eml,
    };
    let n_vars = x_data.len();
    Ok(
        search(&x_data, &y_data, n_vars, &config).map(|r| PySearchResult {
            formula: r.formula,
            error: r.error,
            complexity: r.complexity,
            params: r.params,
        }),
    )
}

/// Reject datasets the search cannot consume: empty, ragged, oversized or
/// non-finite. Non-finite inputs are rejected up front because the guards in
/// `expr.rs` assert on NaN.
pub fn validate_dataset(x_data: &[Vec<f64>], y_data: &[f64]) -> Result<(), InputError> {
    debug_assert!(
        x_data.len() <= usize::MAX / 2,
        "the column count must be a plausible size before it is checked"
    );
    debug_assert!(
        y_data.len() <= usize::MAX / 2,
        "the sample count must be a plausible size before it is checked"
    );
    if x_data.is_empty() {
        return Err(InputError::Empty { what: "x_data" });
    }
    if y_data.is_empty() {
        return Err(InputError::Empty { what: "y_data" });
    }
    check_len(x_data.len(), guard::MAX_VARS)?;
    check_len(y_data.len(), guard::MAX_SAMPLES)?;
    // Bounded: x_data.len() is capped at MAX_VARS by the check above.
    for column in x_data.iter() {
        check_pair(column.len(), y_data.len())?;
        if column.iter().any(|v| !v.is_finite()) {
            return Err(InputError::Invalid {
                what: "x_data contains a non-finite value",
            });
        }
    }
    if y_data.iter().any(|v| !v.is_finite()) {
        return Err(InputError::Invalid {
            what: "y_data contains a non-finite value",
        });
    }
    Ok(())
}

/// Batch Lorentz boost: apply `boost(phi_i, c)` to each `(x_i, y_i)` point.
#[cfg(feature = "python")]
#[pyfunction]
fn boost_n(points: Vec<(f64, f64)>, phis: Vec<f64>, c: f64) -> PyResult<Vec<(f64, f64)>> {
    check_pair(points.len(), phis.len()).map_err(to_py_err)?;
    if c == 0.0 || !c.is_finite() {
        return Err(to_py_err(InputError::Invalid {
            what: "c must be finite and non-zero",
        }));
    }
    debug_assert_eq!(
        points.len(),
        phis.len(),
        "check_pair guarantees equal lengths"
    );
    debug_assert!(
        points.len() <= MAX_BATCH_LEN,
        "check_pair enforces the batch cap"
    );
    Ok(points
        .par_iter()
        .zip(phis.par_iter())
        .map(|((x, y), phi)| {
            let b = EMLPoint::new(*x, *y).boost(*phi, c);
            (b.x, b.y)
        })
        .collect())
}

// -- Batch arithmetic operators (Rayon parallel) ------------------------------
//
// Each of these previously used `zip`, which truncates to the shorter input, so
// `add_n([1,2,3], [1])` quietly returned a one-element list. They now reject a
// length mismatch.

/// Apply `f` to every element of a bounded batch, in parallel.
#[cfg(feature = "python")]
fn map_unary(xs: Vec<f64>, f: impl Fn(f64) -> f64 + Sync + Send) -> PyResult<Vec<f64>> {
    check_len(xs.len(), MAX_BATCH_LEN).map_err(to_py_err)?;
    debug_assert!(
        xs.len() <= MAX_BATCH_LEN,
        "check_len enforces the batch cap"
    );
    let out: Vec<f64> = xs.par_iter().map(|&x| f(x)).collect();
    debug_assert_eq!(out.len(), xs.len(), "a unary map preserves batch length");
    Ok(out)
}

/// Apply `f` elementwise to two batches of equal, bounded length, in parallel.
#[cfg(feature = "python")]
fn map_binary(
    a: Vec<f64>,
    b: Vec<f64>,
    f: impl Fn(f64, f64) -> f64 + Sync + Send,
) -> PyResult<Vec<f64>> {
    check_pair(a.len(), b.len()).map_err(to_py_err)?;
    debug_assert_eq!(a.len(), b.len(), "check_pair guarantees equal lengths");
    debug_assert!(
        a.len() <= MAX_BATCH_LEN,
        "check_pair enforces the batch cap"
    );
    let out: Vec<f64> = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(&x, &y)| f(x, y))
        .collect();
    debug_assert_eq!(out.len(), a.len(), "a binary map preserves batch length");
    Ok(out)
}

/// Batch exp with the Slipping-Wheel guard applied.
#[cfg(feature = "python")]
#[pyfunction]
fn exp_n(xs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_unary(xs, |x| x_guard(x).exp())
}

/// Batch ln with the frame-shift guard applied.
#[cfg(feature = "python")]
#[pyfunction]
fn ln_n(ys: Vec<f64>) -> PyResult<Vec<f64>> {
    map_unary(ys, |y| y_guard(y).ln())
}

/// Batch elementwise add.
#[cfg(feature = "python")]
#[pyfunction]
fn add_n(as_: Vec<f64>, bs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(as_, bs, |a, b| a + b)
}

/// Batch elementwise subtract.
#[cfg(feature = "python")]
#[pyfunction]
fn sub_n(as_: Vec<f64>, bs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(as_, bs, |a, b| a - b)
}

/// Batch elementwise multiply.
#[cfg(feature = "python")]
#[pyfunction]
fn mul_n(as_: Vec<f64>, bs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(as_, bs, |a, b| a * b)
}

/// Batch elementwise divide; NaN where the divisor underflows.
#[cfg(feature = "python")]
#[pyfunction]
fn div_n(as_: Vec<f64>, bs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(
        as_,
        bs,
        |a, b| if b.abs() < 1e-300 { f64::NAN } else { a / b },
    )
}

/// Batch `sqrt(|x|)`.
#[cfg(feature = "python")]
#[pyfunction]
fn sqrt_n(xs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_unary(xs, |x| x.abs().sqrt())
}

/// Batch sin.
#[cfg(feature = "python")]
#[pyfunction]
fn sin_n(xs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_unary(xs, f64::sin)
}

/// Batch cos.
#[cfg(feature = "python")]
#[pyfunction]
fn cos_n(xs: Vec<f64>) -> PyResult<Vec<f64>> {
    map_unary(xs, f64::cos)
}

/// Batch `eml(x, y) = exp(x) - ln(y)`.
#[cfg(feature = "python")]
#[pyfunction]
fn tension_n(xs: Vec<f64>, ys: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(xs, ys, eml)
}

/// Batch `|base| ** exp`.
#[cfg(feature = "python")]
#[pyfunction]
fn pow_n(bases: Vec<f64>, exps: Vec<f64>) -> PyResult<Vec<f64>> {
    map_binary(bases, exps, |b, e| b.abs().powf(e))
}

// Thin bindings for the batch APIs. Those functions are plain Rust now --
// they return `Result<_, InputError>` so Rust callers are not forced to speak
// PyO3 -- and `?` converts at this boundary through `From<InputError>`.

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "octonion_mul_n")]
fn py_octonion_mul_n(a_batch: Vec<[f64; 8]>, b_batch: Vec<[f64; 8]>) -> PyResult<Vec<[f64; 8]>> {
    Ok(octonion_mul_n(a_batch, b_batch)?)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "geometric_product_n")]
fn py_geometric_product_n(
    a_batch: Vec<Vec<f64>>,
    b_batch: Vec<Vec<f64>>,
    signature: Vec<i8>,
) -> PyResult<Vec<Vec<f64>>> {
    Ok(geometric_product_n(a_batch, b_batch, signature)?)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "christoffel_batch_n")]
fn py_christoffel_batch_n(
    points: Vec<(f64, f64)>,
    lam: usize,
    mu: usize,
    nu: usize,
    rs: f64,
) -> PyResult<Vec<f64>> {
    Ok(christoffel_batch_n(points, lam, mu, nu, rs)?)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "simulate_pulses_n")]
fn py_simulate_pulses_n(x0: f64, y0: f64, n_pulses: usize) -> PyResult<Vec<(f64, f64, f64)>> {
    Ok(simulate_pulses_n(x0, y0, n_pulses)?)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "rotate_phase_n")]
fn py_rotate_phase_n(pairs: Vec<(f64, f64)>, angles: Vec<f64>) -> PyResult<Vec<(f64, f64)>> {
    Ok(rotate_phase_n(pairs, angles)?)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "schrodinger_step_n")]
fn py_schrodinger_step_n(
    psi: Vec<(f64, f64)>,
    v: Vec<f64>,
    dt: f64,
    hbar: f64,
) -> PyResult<Vec<(f64, f64)>> {
    Ok(schrodinger_step_n(psi, v, dt, hbar)?)
}

#[cfg(feature = "python")]
#[pymodule]
fn eml_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // Core types
    m.add_class::<EMLPoint>()?;
    m.add_class::<EMLPair>()?;
    m.add_class::<EMLKnot>()?;
    m.add_class::<SearchOptions>()?;
    // Backend handshake
    m.add_function(wrap_pyfunction!(version_rust, m)?)?;
    m.add_function(wrap_pyfunction!(is_rust_backend, m)?)?;
    m.add("VERSION", VERSION)?;
    m.add("MAX_BATCH_LEN", MAX_BATCH_LEN)?;
    // Batch functions
    m.add_function(wrap_pyfunction!(py_schrodinger_step_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_rotate_phase_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_simulate_pulses_n, m)?)?;
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
    m.add_function(wrap_pyfunction!(py_christoffel_batch_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_octonion_mul_n, m)?)?;
    m.add_function(wrap_pyfunction!(py_geometric_product_n, m)?)?;
    // Formula discovery
    m.add_class::<PySearchResult>()?;
    m.add_function(wrap_pyfunction!(find_formula, m)?)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_is_a_three_part_ascii_number() {
        assert!(VERSION.is_ascii());
        assert_eq!(
            VERSION.split('.').count(),
            3,
            "version must be major.minor.patch"
        );
        assert!(VERSION.split('.').all(|p| p.parse::<u32>().is_ok()));
    }

    #[test]
    fn dataset_validation_rejects_the_panic_cases() {
        // The empty-column case used to abort with an index-out-of-bounds panic.
        assert_eq!(
            validate_dataset(&[], &[1.0]),
            Err(InputError::Empty { what: "x_data" })
        );
        assert_eq!(
            validate_dataset(&[vec![1.0, 2.0]], &[1.0]),
            Err(InputError::LengthMismatch { left: 2, right: 1 })
        );
        assert_eq!(
            validate_dataset(&[vec![1.0]], &[]),
            Err(InputError::Empty { what: "y_data" })
        );
        assert!(validate_dataset(&[vec![1.0, 2.0]], &[3.0, 4.0]).is_ok());
    }

    #[test]
    fn dataset_validation_rejects_non_finite() {
        assert!(validate_dataset(&[vec![f64::NAN, 1.0]], &[1.0, 2.0]).is_err());
        assert!(validate_dataset(&[vec![1.0, 2.0]], &[f64::INFINITY, 2.0]).is_err());
    }

    #[test]
    fn search_options_reject_unbounded_configurations() {
        let base = SearchOptions {
            max_complexity: 8,
            beam_width: 2000,
            precision_goal: 1e-10,
            use_trig: true,
            use_eml: true,
            complexity_penalty: 0.001,
        };
        assert!(base.validate().is_ok());
        assert!(SearchOptions {
            max_complexity: 0,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(SearchOptions {
            beam_width: usize::MAX,
            ..base.clone()
        }
        .validate()
        .is_err());
        assert!(SearchOptions {
            precision_goal: f64::NAN,
            ..base
        }
        .validate()
        .is_err());
    }
}
