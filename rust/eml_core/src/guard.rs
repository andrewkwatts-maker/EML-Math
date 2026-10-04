//! Shared numeric guards and input bounds for `eml_core`.
//!
//! Three copies of the EML overflow threshold used to live in this crate
//! (`point.rs`, `lib.rs`, `metric.rs`) and a fourth in `discover/expr.rs`.
//! Two of them carried the rounded-down literal `709.78`, so for
//! `x` in `(709.78, 709.782712893384]` those paths dampened `x` to `ln(x)`
//! while Python and the other two paths did not -- a divergence of order
//! 1e308. One definition removes the class of bug entirely.

/// `f64::MAX.ln()`, the largest `x` for which `exp(x)` is finite.
///
/// Must equal Python's `eml_math.constants.OVERFLOW_THRESHOLD` exactly.
pub const OVERFLOW_THRESHOLD: f64 = 709.782712893384;

/// Upper bound on elements accepted from a Python sequence in one call.
///
/// Safety-critical standard 2: every loop has a fixed, checked bound. Each
/// batch wrapper checks its input length against this before iterating, so a
/// malformed or hostile payload fails fast instead of allocating unboundedly.
pub const MAX_BATCH_LEN: usize = 1_048_576;

/// Upper bound on the number of data rows accepted by the formula search.
pub const MAX_SAMPLES: usize = 1_048_576;

/// Upper bound on input variables accepted by the formula search. The formula
/// printer names variables `x`, `y`, `z`, ... from `b'x'`, so more than this
/// would produce names outside the printable ASCII range.
pub const MAX_VARS: usize = 26;

/// Slipping-Wheel dampening: `x` above the threshold is folded to `ln(x)` so
/// `exp(x)` stays finite.
///
/// Byte-for-byte equivalent to the Python guard in `EMLPoint.tension`.
#[inline]
pub fn x_guard(x: f64) -> f64 {
    debug_assert!(
        !x.is_nan(),
        "x_guard received NaN; callers must reject NaN first"
    );
    debug_assert!(
        x.is_finite() || x.is_infinite(),
        "a non-NaN float is either finite or infinite; NaN was excluded above"
    );
    if x > OVERFLOW_THRESHOLD {
        x.ln()
    } else {
        x
    }
}

/// Frame-shift guard: map `y` into the domain of `ln`.
///
/// Equivalent to Python's `y_safe = abs(y) if y <= 0 else y; if y_safe == 0:
/// y_safe = 1e-300`. The floor applies **only to zero**. Writing this as
/// `y.abs().max(1e-300)` -- as five methods in `point.rs` still did -- also
/// crushes legitimate subnormals such as `y = -1e-320` up to `1e-300`, which
/// changes `ln(y)` by a factor of 46 in the result.
#[inline]
pub fn y_guard(y: f64) -> f64 {
    debug_assert!(
        !y.is_nan(),
        "y_guard received NaN; callers must reject NaN first"
    );
    let a = if y <= 0.0 { y.abs() } else { y };
    debug_assert!(
        a >= 0.0,
        "frame shift must produce a non-negative magnitude"
    );
    if a == 0.0 {
        1e-300
    } else {
        a
    }
}

/// Evaluate `eml(x, y) = exp(x) - ln(y)` with both guards applied.
#[inline]
pub fn eml(x: f64, y: f64) -> f64 {
    debug_assert!(!x.is_nan(), "eml received NaN x");
    debug_assert!(!y.is_nan(), "eml received NaN y");
    let ys = y_guard(y);
    debug_assert!(
        ys > 0.0,
        "y_guard must return a strictly positive value for ln"
    );
    x_guard(x).exp() - ys.ln()
}

/// Error returned when a batch wrapper rejects its input.
///
/// Kept as a plain enum so the numeric core stays free of PyO3; `lib.rs`
/// converts it to a Python exception at the boundary. Standard 7: errors
/// surface as exceptions, never as a default value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InputError {
    /// Two parallel sequences had different lengths.
    LengthMismatch { left: usize, right: usize },
    /// A sequence exceeded [`MAX_BATCH_LEN`] (or another stated cap).
    TooLong { len: usize, max: usize },
    /// A required sequence was empty.
    Empty { what: &'static str },
    /// A structural precondition the other variants do not cover.
    Invalid { what: &'static str },
}

impl core::fmt::Display for InputError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            InputError::LengthMismatch { left, right } => write!(
                f,
                "sequence lengths differ: {left} vs {right}; zipping them would \
                 silently truncate to the shorter one"
            ),
            InputError::TooLong { len, max } => {
                write!(f, "sequence length {len} exceeds the maximum of {max}")
            }
            InputError::Empty { what } => write!(f, "{what} must not be empty"),
            InputError::Invalid { what } => write!(f, "{what}"),
        }
    }
}

impl std::error::Error for InputError {}

/// Reject a batch that is longer than `max`.
pub fn check_len(len: usize, max: usize) -> Result<(), InputError> {
    debug_assert!(max > 0, "a zero maximum would reject every input");
    debug_assert!(
        max <= usize::MAX / 2,
        "max must leave headroom for index arithmetic"
    );
    if len > max {
        return Err(InputError::TooLong { len, max });
    }
    Ok(())
}

/// Reject two parallel batches that are not the same length, or too long.
///
/// `Iterator::zip` truncates to the shorter side, which turns a caller's
/// off-by-one into a quietly shortened result rather than an error.
pub fn check_pair(left: usize, right: usize) -> Result<(), InputError> {
    debug_assert!(
        left <= usize::MAX / 2,
        "left length must be a plausible batch size"
    );
    debug_assert!(
        right <= usize::MAX / 2,
        "right length must be a plausible batch size"
    );
    if left != right {
        return Err(InputError::LengthMismatch { left, right });
    }
    check_len(left, MAX_BATCH_LEN)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn threshold_matches_f64_max_ln() {
        assert_eq!(OVERFLOW_THRESHOLD, f64::MAX.ln());
        assert!(OVERFLOW_THRESHOLD.exp().is_finite());
    }

    #[test]
    fn x_guard_dampens_only_above_threshold() {
        assert_eq!(x_guard(1.0), 1.0);
        assert_eq!(x_guard(OVERFLOW_THRESHOLD), OVERFLOW_THRESHOLD);
        // The old 709.78 literal dampened this point; the corrected one does not.
        assert_eq!(x_guard(709.7813), 709.7813);
        assert_eq!(x_guard(1000.0), 1000.0_f64.ln());
    }

    #[test]
    fn y_guard_preserves_subnormals() {
        // `.max(1e-300)` would return 1e-300 here, a factor-of-46 error in ln.
        assert_eq!(y_guard(-1e-320), 1e-320);
        assert_eq!(y_guard(-2.0), 2.0);
        assert_eq!(y_guard(0.0), 1e-300);
        assert_eq!(y_guard(3.0), 3.0);
    }

    #[test]
    fn eml_matches_the_defining_identity() {
        assert!((eml(1.0, 1.0) - std::f64::consts::E).abs() < 1e-12);
        assert!((eml(0.0, 1.0) - 1.0).abs() < 1e-12);
        assert!((eml(0.0, std::f64::consts::E) - 0.0).abs() < 1e-12);
    }

    #[test]
    fn check_pair_rejects_mismatch_and_overlong() {
        assert_eq!(check_pair(3, 3), Ok(()));
        assert_eq!(
            check_pair(3, 1),
            Err(InputError::LengthMismatch { left: 3, right: 1 })
        );
        assert_eq!(
            check_len(MAX_BATCH_LEN + 1, MAX_BATCH_LEN),
            Err(InputError::TooLong {
                len: MAX_BATCH_LEN + 1,
                max: MAX_BATCH_LEN
            })
        );
    }

    #[test]
    fn input_error_messages_are_ascii() {
        let msgs = [
            InputError::LengthMismatch { left: 1, right: 2 }.to_string(),
            InputError::TooLong { len: 9, max: 8 }.to_string(),
            InputError::Empty { what: "x_data" }.to_string(),
            InputError::Invalid { what: "bad" }.to_string(),
        ];
        for m in &msgs {
            assert!(m.is_ascii(), "diagnostic must stay ASCII: {m}");
        }
    }
}
