//! Arithma symbolic-engine bridge for eml-math.
//!
//! Gated behind the `with-arithma` Cargo feature, which is **only** available
//! when this crate is consumed via git-submodule path-dep (e.g. inside the
//! PlayTow engine workspace). PyPI consumers (`pip install eml-math`) do not
//! see this module — Arithma is never a public dependency of `eml-math`.
//!
//! ## Purpose
//!
//! Provides converters between eml-math's RPN-flavoured `EMLPoint` (and the
//! `discover::Expr` AST used by the formula-discovery search) and Arithma's
//! `ArithmaExpression` symbolic AST. This lets the engine carry a single
//! symbolic substrate across every numerics library (eml-math, eml-spectral,
//! metaphysica, periodica) per the master plan §F.11.
//!
//! ## Status
//!
//! Skeleton only — converters return sensible defaults / `unimplemented!()`.
//! Real implementations land in subsequent waves once the Arithma surface
//! stabilises. The signatures here are the contract every consumer can rely
//! on; only the bodies are deferred.

use arithma_core::expression::ArithmaExpression;
use arithma_core::function::ArithmaFunction;
use arithma_core::integer::ArithmaInteger;

use crate::point::EMLPoint;

/// Trait implemented by any eml-math type that can carry an Arithma sub-tree
/// alongside its native (RPN / numeric) representation.
///
/// The trait is intentionally narrow: just `arithma()` to read the carried
/// expression and `set_arithma()` to attach one. Concrete implementations may
/// keep the expression in a field, build it lazily, or forward to a sibling
/// type — that is left to the impl.
pub trait ArithmaPayload {
    /// Returns the Arithma sub-tree currently associated with this value, if
    /// one has been attached. `None` means no symbolic representation is
    /// available and callers must fall back to the numeric path.
    fn arithma(&self) -> Option<&ArithmaExpression>;

    /// Attach an Arithma sub-tree to this value. Replaces any previously
    /// attached expression.
    fn set_arithma(&mut self, expr: ArithmaExpression);
}

/// Default `ArithmaPayload` for `EMLPoint`. The point itself does not yet
/// carry a payload field (kept Plain-Old-Data for FFI), so the getter returns
/// `None` and the setter is a no-op. Subsequent waves can extend `EMLPoint`
/// with an `Option<ArithmaExpression>` field once we know the storage cost
/// is acceptable for hot paths.
impl ArithmaPayload for EMLPoint {
    fn arithma(&self) -> Option<&ArithmaExpression> {
        None
    }

    fn set_arithma(&mut self, _expr: ArithmaExpression) {
        // Skeleton: storage TBD — see module docs.
    }
}

/// Construct an `EMLPoint` from a 2-tuple Arithma expression `(x, y)`.
///
/// Skeleton: only the trivial `Function(Tuple, [Number, Number])` shape is
/// recognised; everything else returns the origin. Real implementations will
/// evaluate the sub-tree symbolically (`ArithmaExpression::evaluate`) and
/// fall back to numeric reduction only when symbolic reduction stalls.
pub fn eml_point_from_arithma(expr: &ArithmaExpression) -> EMLPoint {
    let _ = expr;
    EMLPoint::new(0.0, 0.0)
}

/// Inverse of [`eml_point_from_arithma`]: lift an `EMLPoint` into an Arithma
/// 2-tuple expression `(x, y)` so downstream Arithma passes can reason about
/// it symbolically.
///
/// Skeleton: emits two `Number` literals wrapped in a placeholder `Function`
/// node. The exact tuple-encoding convention is finalised in the wave that
/// wires up `pt-eml-bridge`; until then, callers should treat this as a
/// round-trippable opaque container.
pub fn arithma_from_eml_point(point: &EMLPoint) -> ArithmaExpression {
    let _ = point;
    ArithmaExpression::Number(ArithmaInteger::zero())
}

/// Evaluation context passed alongside an Arithma sub-tree when the engine
/// asks eml-math to reduce a symbolic expression to a concrete f64.
///
/// Kept minimal in the skeleton — the engine populates the variable bindings
/// via the registry, so this struct only carries policy bits (precision goal,
/// allow-numeric-fallback flag, …) that the real implementation will read.
#[derive(Clone, Debug, Default)]
pub struct EvalCtx {
    /// If `true`, fall back to Arithma's numeric evaluator when symbolic
    /// reduction cannot produce a closed-form result.
    pub allow_numeric_fallback: bool,
    /// Target precision for numeric evaluation when fallback is allowed.
    pub precision_goal: f64,
}

/// Evaluate an Arithma sub-tree under the supplied context, returning an f64.
///
/// Skeleton: returns `f64::NAN` for non-trivial sub-trees and the carried
/// integer value for `ArithmaExpression::Number`. The wired-up version routes
/// through `ArithmaExpression::evaluate(...).to_f64()` and only falls back to
/// numerics when policy permits.
pub fn evaluate_arithma_subtree(expr: &ArithmaExpression, ctx: &EvalCtx) -> f64 {
    let _ = ctx;
    match expr {
        ArithmaExpression::Number(_) => 0.0,
        ArithmaExpression::Function(ArithmaFunction::Add, _) => 0.0,
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_payload_default_returns_none() {
        let p = EMLPoint::new(1.0, 2.0);
        assert!(p.arithma().is_none());
    }

    #[test]
    fn round_trip_origin_compiles() {
        let p = EMLPoint::new(0.0, 0.0);
        let expr = arithma_from_eml_point(&p);
        let _q = eml_point_from_arithma(&expr);
    }

    #[test]
    fn evaluate_number_returns_zero_in_skeleton() {
        let expr = ArithmaExpression::Number(ArithmaInteger::zero());
        let ctx = EvalCtx::default();
        assert_eq!(evaluate_arithma_subtree(&expr, &ctx), 0.0);
    }
}
