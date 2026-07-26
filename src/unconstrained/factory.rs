//! Factory for unconstrained optimizer implementations.
//!
//! It constructs conjugate-gradient or quasi-Newton solvers.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{Error, Options};
use super::conjugate_gradient::ConjugateGradient;
use super::conjugate_gradient::Options as ConjugateGradientOptions;
use super::quasi_newton::Options as QuasiNewtonOptions;
use super::quasi_newton::QuasiNewton;
use crate::common::Minimizer;
use crate::{RealFn, ValidationError, Vector, VectorReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
/// Selects an unconstrained optimization algorithm.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Method {
    /// Conjugate-gradient method.
    ConjugateGradient(ConjugateGradientOptions),
    /// Quasi-Newton method.
    QuasiNewton(QuasiNewtonOptions),
}
//}}}
//{{{ impl: Method
impl Method {
    /// Returns mutable access to common stopping options.
    #[trace_fn]
    pub fn uncon_opts_mut(&mut self) -> &mut Options {
        match self {
            Method::ConjugateGradient(cg_opts) => &mut cg_opts.uncon_opts,
            Method::QuasiNewton(qn_opts) => &mut qn_opts.uncon_opts,
        }
    }

    /// Returns common stopping options.
    #[trace_fn]
    pub fn uncon_opts(&self) -> &Options {
        match self {
            Method::ConjugateGradient(cg_opts) => &cg_opts.uncon_opts,
            Method::QuasiNewton(qn_opts) => &qn_opts.uncon_opts,
        }
    }

    /// Validates the selected optimizer's complete configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if any nested option is invalid.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::ConjugateGradient(options) => options.validate(),
            Self::QuasiNewton(options) => options.validate(),
        }
    }
}
//}}}
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F: RealFn + ?Sized + 'a>(
    fcn: &'a mut F,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error, Returns = VectorReturns> + 'a> {
    match method {
        Method::ConjugateGradient(opts) => Box::new(ConjugateGradient::new(fcn, x0, opts)),
        Method::QuasiNewton(opts) => Box::new(QuasiNewton::new(fcn, x0, opts)),
    }
}
//}}}
