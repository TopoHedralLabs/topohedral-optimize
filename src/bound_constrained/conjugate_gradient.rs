//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use approx::RelativeEq;

//{{{ crate imports
use super::common::Options as BoundConstrainedOptions;
use crate::common::CountingRealFn;
use crate::common::Vector;
use crate::{bound_constrained::common::BoundConstrainedMinimizer, constraints::BoundsConstraints};
use crate::{IterData, RealFn};
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::VectorOps;
//}}}
//--------------------------------------------------------------------------------------------------

pub struct Options
{
    bound_opts: BoundConstrainedOptions,
}

pub struct BoundedConjugateGradient<F: RealFn>
{
    fcn: F,
    x_init: Vector,
    norm_grad_fx_init: f64,
    bounds: BoundsConstraints,
    opts: Options,
}

impl<F: RealFn> BoundedConjugateGradient<F>
{
    pub fn new(
        mut fcn: F,
        mut x0: Vector,
        bounds: BoundsConstraints,
        opts: Options,
    ) -> Self
    {
        bounds.clamp(&mut x0);
        let grad_0 = fcn.grad(&x0);
        let projectd_grad_0 = bounds.projected_direction(&x0, &grad_0, 1.0);

        Self {
            fcn,
            x_init: x0,
            norm_grad_fx_init: projectd_grad_0.norm(),
            bounds: bounds,
            opts,
        }
    }
}

impl<F: RealFn> BoundConstrainedMinimizer for BoundedConjugateGradient<F>
{
    fn minimize(&mut self) -> Result<crate::Returns, super::common::Error>
    {
        let n = self.fcn.dimension();
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut projected_gradient = Vector::zeros_vec(n, topohedral_linalg::VecType::Col);

        for i in 0..self.opts.bound_opts.max_iter
        {
            projected_gradient.copy_from(self.bounds.projected_direction(
                &iter_k.x,
                &iter_k.grad_fx,
                1.0,
            ));
        }
        todo!()
    }
}
