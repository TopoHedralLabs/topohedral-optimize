//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::{IterData, RealFn, Vector};
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
mod interp;
mod nocedal;
mod thuente;
mod utils;
//}}}
//{{{ pub use: common exports
pub use common::{
    Error as LineSearchError, LineSearch, LineSearchFcn, Options as LineSearchOptions,
    Returns as LineSearchReturns,
};
//}}}
//{{{ pub use: factory exports
pub use factory::{create, Method as LineSearchMethod};
//}}}
//{{{ pub use: interp exports
pub use interp::Interp;
pub use interp::Options as InterpOptions;
//}}}
//{{{ pub use: thuente exports
pub use thuente::Options as ThuenteOptions;
pub use thuente::Thuente;
//}}}
//{{{ pub use: nocedal exports
pub use nocedal::Nocedal;
pub use nocedal::Options as NocedalOptions;
//}}}
//{{{ pub fn: search
#[trace_fn]
pub fn search<F: RealFn>(
    mut fcn: F,
    iter_data: &IterData,
    dir: &Vector,
    alpha_init: f64,
    method: LineSearchMethod,
) -> Result<IterData, LineSearchError>
{
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
    let line_search_fcn = LineSearchFcn::new(fcn.clone(), x.clone(), dir.clone());
    let mut line_searcher = create(line_search_fcn, method);
    let ret = line_searcher.search(*fx, dphi0, alpha_init)?;
    let new_x: Vector = (x + ret.alpha * dir).into();
    let new_fx = ret.phi_alpha;
    let new_grad_fx = fcn.grad(&new_x);
    let new_norm_grad_fx = new_grad_fx.norm();
    Ok(IterData {
        x: new_x,
        fx: new_fx,
        grad_fx: new_grad_fx,
        norm_grad_fx: new_norm_grad_fx,
    })
}
//}}}
//{{{ pub use: utils exports
pub use utils::initial_step;

//}}}
