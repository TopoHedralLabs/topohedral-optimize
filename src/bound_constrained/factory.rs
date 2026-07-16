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
use crate::constraints::BoundsConstraints;
use crate::{RealFn, Vector, VectorReturns};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

#[derive(Clone)]
/// Selects a bound-constrained optimization algorithm.
pub enum Method {
    /// Active-set algorithm.
    Asa(AsaOptions),
    /// L-BFGS-B algorithm.
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
}

#[trace_fn]
/// Constructs the selected bound-constrained optimizer.
pub fn create<'a, F: RealFn + ?Sized + 'a>(
    fcn: &'a mut F,
    bounds: BoundsConstraints,
    x0: Vector,
    method: Method,
) -> Box<dyn Minimizer<Error = Error, Returns = VectorReturns> + 'a> {
    match method {
        Method::Asa(opts) => Box::new(ActiveSetAlgorithm::new(fcn, bounds, x0, opts)),
        Method::Bfgsb(opts) => Box::new(Bfgsb::new(fcn, bounds, x0, opts)),
    }
}
