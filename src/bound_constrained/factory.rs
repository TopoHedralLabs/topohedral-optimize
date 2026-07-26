//! Factory for constructing bound-constrained optimizer implementations.
//!
//! The selected method owns its algorithm-specific options.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::asa::{ActiveSetAlgorithm, Options as AsaOptions};
use super::bfgsb::{Bfgsb, Options as BfgsbOptions};
use super::common::Error;
use crate::bound_constrained::BoundConstrainedOptions;
use crate::common::Minimizer;
use crate::constraints::BoundConstraints;
use crate::{RealFn, ValidationError, Vector, VectorReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
/// Selects a bound-constrained optimization algorithm.
#[non_exhaustive]
pub enum Method {
    /// Active-set algorithm.
    Asa(AsaOptions),
    /// BFGS-B algorithm.
    Bfgsb(BfgsbOptions),
}

impl Method {
    /// Returns the shared bound-constrained options.
    pub fn bound_opts(&self) -> &BoundConstrainedOptions {
        match self {
            Method::Asa(asa_opts) => &asa_opts.bound_opts,
            Method::Bfgsb(bfgsb_opts) => &bfgsb_opts.bound_opts,
        }
    }

    /// Returns mutable access to the shared bound-constrained options.
    pub fn bound_opts_mut(&mut self) -> &mut BoundConstrainedOptions {
        match self {
            Method::Asa(asa_opts) => &mut asa_opts.bound_opts,
            Method::Bfgsb(bfgsb_opts) => &mut bfgsb_opts.bound_opts,
        }
    }

    /// Validates the selected optimizer's complete configuration.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if any nested option is invalid.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Asa(options) => options.validate(),
            Self::Bfgsb(options) => options.validate(),
        }
    }
}

#[trace_fn]
/// Constructs the selected bound-constrained optimizer.
pub fn create<'a, F: RealFn + ?Sized + 'a>(
    fcn: &'a mut F,
    bounds: BoundConstraints,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error, Returns = VectorReturns> + 'a> {
    match method {
        Method::Asa(opts) => Box::new(ActiveSetAlgorithm::new(fcn, bounds, x0, opts)),
        Method::Bfgsb(opts) => Box::new(Bfgsb::new(fcn, bounds, x0, opts)),
    }
}
