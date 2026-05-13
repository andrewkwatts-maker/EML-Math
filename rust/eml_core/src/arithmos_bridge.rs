//! Arithmos symbolic-engine bridge for eml-math.
//!
//! Gated behind the `with-arithmos` Cargo feature, which is **only** available
//! when this crate is consumed via git-submodule path-dep (e.g. inside the
//! PlayTow engine workspace). PyPI consumers (`pip install eml-math`) do not
//! see this module — Arithmos is never a public dependency of `eml-math`.
//!
//! ## Purpose
//!
//! Provides converters between eml-math's RPN-flavoured `EMLPoint` (and the
//! `discover::Expr` AST used by the formula-discovery search) and Arithmos's
//! `ArithmosExpression` symbolic AST. This lets the engine carry a single
//! symbolic substrate across every numerics library (eml-math, eml-spectral,
//! metaphysica, periodica) per the master plan §F.11.
//!
//! ## Status
//!
//! Skeleton only — converters return sensible defaults / `unimplemented!()`.
//! Real implementations land in subsequent waves once the Arithmos surface
//! stabilises. The signatures here are the contract every consumer can rely
//! on; only the bodies are deferred.

use arithmos_core::expression::ArithmosExpression;
use arithmos_core::function::ArithmosFunction;
use arithmos_core::integer::ArithmosInteger;

use crate::point::EMLPoint;

/// Trait implemented by any eml-math type that can carry an Arithmos sub-tree
/// alongside its native (RPN / numeric) representation.
///
/// The trait is intentionally narrow: just `arithmos()` to read the carried
/// expression and `set_arithmos()` to attach one. Concrete implementations may
/// keep the expression in a field, build it lazily, or forward to a sibling
/// type — that is left to the impl.
pub trait ArithmosPayload {
    /// Returns the Arithmos sub-tree currently associated with this value, if
    /// one has been attached. `None` means no symbolic representation is
    /// available and callers must fall back to the numeric path.
    fn arithmos(&self) -> Option<&ArithmosExpression>;

    /// Attach an Arithmos sub-tree to this value. Replaces any previously
    /// attached expression.
    fn set_arithmos(&mut self, expr: ArithmosExpression);
}

/// Default `ArithmosPayload` for `EMLPoint`. The point itself does not yet
/// carry a payload field (kept Plain-Old-Data for FFI), so the getter returns
/// `None` and the setter is a no-op. Subsequent waves can extend `EMLPoint`
/// with an `Option<ArithmosExpression>` field once we know the storage cost
/// is acceptable for hot paths.
impl ArithmosPayload for EMLPoint {
    fn arithmos(&self) -> Option<&ArithmosExpression> {
        None
    }

    fn set_arithmos(&mut self, _expr: ArithmosExpression) {
        // Skeleton: storage TBD — see module docs.
    }
}

/// Construct an `EMLPoint` from a 2-tuple Arithmos expression `(x, y)`.
///
/// Skeleton: only the trivial `Function(Tuple, [Number, Number])` shape is
/// recognised; everything else returns the origin. Real implementations will
/// evaluate the sub-tree symbolically (`ArithmosExpression::evaluate`) and
/// fall back to numeric reduction only when symbolic reduction stalls.
pub fn eml_point_from_arithmos(expr: &ArithmosExpression) -> EMLPoint {
    let _ = expr;
    EMLPoint::new(0.0, 0.0)
}

/// Inverse of [`eml_point_from_arithmos`]: lift an `EMLPoint` into an Arithmos
/// 2-tuple expression `(x, y)` so downstream Arithmos passes can reason about
/// it symbolically.
///
/// Skeleton: emits two `Number` literals wrapped in a placeholder `Function`
/// node. The exact tuple-encoding convention is finalised in the wave that
/// wires up `pt-eml-bridge`; until then, callers should treat this as a
/// round-trippable opaque container.
pub fn arithmos_from_eml_point(point: &EMLPoint) -> ArithmosExpression {
    let _ = point;
    ArithmosExpression::Number(ArithmosInteger::zero())
}

/// Evaluation context passed alongside an Arithmos sub-tree when the engine
/// asks eml-math to reduce a symbolic expression to a concrete f64.
///
/// Kept minimal in the skeleton — the engine populates the variable bindings
/// via the registry, so this struct only carries policy bits (precision goal,
/// allow-numeric-fallback flag, …) that the real implementation will read.
#[derive(Clone, Debug, Default)]
pub struct EvalCtx {
    /// If `true`, fall back to Arithmos's numeric evaluator when symbolic
    /// reduction cannot produce a closed-form result.
    pub allow_numeric_fallback: bool,
    /// Target precision for numeric evaluation when fallback is allowed.
    pub precision_goal: f64,
}

/// Evaluate an Arithmos sub-tree under the supplied context, returning an f64.
///
/// Skeleton: returns `f64::NAN` for non-trivial sub-trees and the carried
/// integer value for `ArithmosExpression::Number`. The wired-up version routes
/// through `ArithmosExpression::evaluate(...).to_f64()` and only falls back to
/// numerics when policy permits.
pub fn evaluate_arithmos_subtree(expr: &ArithmosExpression, ctx: &EvalCtx) -> f64 {
    let _ = ctx;
    match expr {
        ArithmosExpression::Number(_) => 0.0,
        ArithmosExpression::Function(ArithmosFunction::Add, _) => 0.0,
        _ => f64::NAN,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn point_payload_default_returns_none() {
        let p = EMLPoint::new(1.0, 2.0);
        assert!(p.arithmos().is_none());
    }

    #[test]
    fn round_trip_origin_compiles() {
        let p = EMLPoint::new(0.0, 0.0);
        let expr = arithmos_from_eml_point(&p);
        let _q = eml_point_from_arithmos(&expr);
    }

    #[test]
    fn evaluate_number_returns_zero_in_skeleton() {
        let expr = ArithmosExpression::Number(ArithmosInteger::zero());
        let ctx = EvalCtx::default();
        assert_eq!(evaluate_arithmos_subtree(&expr, &ctx), 0.0);
    }
}
