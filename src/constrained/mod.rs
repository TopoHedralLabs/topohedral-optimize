//! Module implementing a set of constrained optimisation algorithms
//!
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{DifferentiableFn, Evaluator, RealFn, RealVectorFn, Vector, VectorReturns},
    constraints::BoundConstraints,
    ValidationError,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod augmented_lagrangian;
mod common;
mod factory;
//}}}

//{{{ pub use: common exports
pub use common::{Error as ConstrainedError, Options as ConstrainedOptions};
//}}}
//{{{ pub use: augmented_lagrangian exports
pub use augmented_lagrangian::{
    InnerMethod as AugmentedLagrangianInnerMethod, Options as AugmentedLagrangianOptions,
};
//}}}
//{{{ pub use: factory export
pub use factory::Method as ConstrainedMethod;
//}}}

//{{{ fn: minimize
#[trace_fn]
/// Minimizes a function subject to optional bounds and vector constraints.
///
/// Equality constraints are interpreted as `h(x) = 0`; inequality constraints
/// are interpreted as `g(x) <= 0`.
///
/// # Errors
///
/// Returns [`ConstrainedError`] if method options or dimensions are invalid,
/// bounds are paired with an incompatible inner solver, an inner solve fails,
/// or the iteration limit is reached.
pub fn minimize<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: Option<BoundConstraints>,
    eq_constraints: Option<&mut dyn RealVectorFn>,
    ieq_constraints: Option<&mut dyn RealVectorFn>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError> {
    let mut evaluator = Evaluator::new(fcn);
    let mut ret = minimize_impl(
        &mut evaluator,
        bounds,
        eq_constraints,
        ieq_constraints,
        x0,
        method,
    )?;
    ret.num_fun_evals = evaluator.num_func_evals;
    ret.num_grad_evals = evaluator.num_grad_evals;
    Ok(ret)
}

pub(crate) fn minimize_impl<F: RealFn + ?Sized>(
    fcn: &mut F,
    bounds: Option<BoundConstraints>,
    eq_constraints: Option<&mut dyn RealVectorFn>,
    ieq_constraints: Option<&mut dyn RealVectorFn>,
    x0: Vector,
    method: ConstrainedMethod,
) -> Result<VectorReturns, ConstrainedError> {
    method.validate()?;
    let expected = fcn.dimension_domain();
    if x0.len() != expected {
        return Err(ValidationError::DimensionMismatch {
            parameter: "x0",
            expected,
            actual: x0.len(),
        }
        .into());
    }
    if let Some(bounds) = bounds.as_ref() {
        let actual = bounds.dimension_domain();
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                parameter: "bounds",
                expected,
                actual,
            }
            .into());
        }
    }
    if let Some(constraints) = eq_constraints.as_ref() {
        let actual = constraints.dimension_domain();
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                parameter: "eq_constraints",
                expected,
                actual,
            }
            .into());
        }
    }
    if let Some(constraints) = ieq_constraints.as_ref() {
        let actual = constraints.dimension_domain();
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                parameter: "ieq_constraints",
                expected,
                actual,
            }
            .into());
        }
    }
    let minimizer = factory::create(fcn, bounds, eq_constraints, ieq_constraints, x0, method);
    minimizer?.minimize()
}
//}}}
