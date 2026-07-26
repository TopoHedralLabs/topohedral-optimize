//! Line-search algorithms for choosing optimization step lengths.
//!
//! Both multidimensional and one-dimensional convenience APIs are provided.
//--------------------------------------------------------------------------------------------------

use crate::RealFn1;

//{{{ crate imports
use super::common::{RealFn, Vector};
use crate::{IterData, ValidationError};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ mod: submodules
mod common;
mod factory;
mod nocedal;
mod thuente;
mod utils;
//}}}
//{{{ pub use: common exports
pub use common::{
    Error as LineSearchError, Options as LineSearchOptions, Returns as LineSearchReturns,
};
//}}}
//{{{ pub use: factory export
pub use factory::Method as LineSearchMethod;
//}}}
//{{{ pub use: thuente exports
pub use thuente::Options as ThuenteOptions;
//}}}
//{{{ pub use: nocedal exports
pub use nocedal::Options as NocedalOptions;
//}}}
//{{{ pub use: utils exports
pub use utils::initial_step;
//}}}
//{{{ pub fn: search
#[trace_fn]
/// Searches along a vector direction from the current iterate.
///
/// # Errors
///
/// Returns [`LineSearchError`] if the method configuration, initial step, or
/// vector dimensions are invalid, or if no acceptable step can be found.
pub fn line_search<F: RealFn + ?Sized>(
    fcn: &mut F,
    iter_data: &IterData,
    dir: &Vector,
    alpha_init: f64,
    method: LineSearchMethod,
) -> Result<IterData, LineSearchError> {
    method.validate()?;
    if !alpha_init.is_finite() || alpha_init <= 0.0 {
        return Err(ValidationError::InvalidFloat {
            parameter: "alpha_init",
            value: alpha_init,
            requirement: "must be finite and greater than zero",
        }
        .into());
    }
    let expected = fcn.dimension_domain();
    for (parameter, actual) in [
        ("iter_data.x", iter_data.x.len()),
        ("iter_data.grad_fx", iter_data.grad_fx.len()),
        ("dir", dir.len()),
    ] {
        if actual != expected {
            return Err(ValidationError::DimensionMismatch {
                parameter,
                expected,
                actual,
            }
            .into());
        }
    }
    //{{{ trace
    trace!(target: "ls", "Running with parameters:");
    trace!(target: "ls","iter_data: {iter_data}");
    trace!(target: "ls","alpha_init: {alpha_init}");
    //}}}
    let IterData {
        x,
        fx,
        grad_fx,
        norm_grad_fx: _,
    } = iter_data;
    let dphi0 = grad_fx.dot(dir);
    let ret = {
        let line_search_fcn = common::LineSearchFcn::new(fcn, x.clone(), dir.clone());
        let mut line_searcher = factory::create(line_search_fcn, method);
        line_searcher.search(*fx, dphi0, alpha_init)?
    };
    let new_x: Vector = (x + ret.alpha * dir).into();
    let new_fx = ret.phi_alpha;
    let new_grad_fx = fcn.derivative(&new_x);
    let new_norm_grad_fx = new_grad_fx.norm();
    Ok(IterData {
        x: new_x,
        fx: new_fx,
        grad_fx: new_grad_fx,
        norm_grad_fx: new_norm_grad_fx,
    })
}
//}}}
//{{{ pub fn: search1d
#[trace_fn]
/// Searches for a step using a scalar line-search function.
///
/// # Errors
///
/// Returns [`LineSearchError`] if the method configuration or initial step is
/// invalid, or if no acceptable step can be found.
pub fn line_search_1d<F: RealFn1 + ?Sized>(
    fcn: &mut F,
    alpha_init: f64,
    method: LineSearchMethod,
) -> Result<LineSearchReturns, LineSearchError> {
    method.validate()?;
    if !alpha_init.is_finite() || alpha_init <= 0.0 {
        return Err(ValidationError::InvalidFloat {
            parameter: "alpha_init",
            value: alpha_init,
            requirement: "must be finite and greater than zero",
        }
        .into());
    }
    let phi0 = fcn.eval(&0.0);
    let dphi0 = fcn.derivative(&0.0);
    let mut line_searcher = factory::create(fcn, method);
    line_searcher.search(phi0, dphi0, alpha_init)
}

//}}}
