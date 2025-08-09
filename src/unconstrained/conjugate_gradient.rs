//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use std::cmp::max;

use crate::line_search::LineSearcher;
//{{{ crate imports 
use crate::RealFn;
use crate::line_search as ls;
use super::common::Options as UnonstrainedOptions;
use super::common::{Error, UnconstrainedMinimizer, Returns};
use topohedral_linalg::VectorOps;
//}}}
//{{{ std imports 
use std::ops::{Add, Mul};
//}}}
//{{{ dep imports 
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

pub enum Direction {
    Steepest, 
    PolakRibiere
}


// #[derive(Copy)]
pub struct Options {
    uncon_opts: UnonstrainedOptions,
    direction: Direction, 
    restart: u64
}

pub struct ConjugateGradient<F: RealFn> {
    fcn: F,
    opts: Options,
    // line_searcher: Box<dyn LineSearcher<F>>,
}

impl<F: RealFn> ConjugateGradient<F> 
    where 
    F::Vector: VectorOps<ScalarType=f64> + Add<Output = F::Vector> + Clone + Default,
    f64: Mul<F::Vector, Output = F::Vector>,
{

    pub fn new(fcn: F, opts: Options) -> Self {
        Self {
            fcn: fcn.clone(), 
            opts: opts,
            // line_searcher: ls::create(F::Vector::default(), F::Vector::default(), fcn.clone(), opts.uncon_opts.ls_method)
        }
    }
}

impl<F: RealFn> UnconstrainedMinimizer for ConjugateGradient<F>
where 
    F::Vector: VectorOps<ScalarType=f64> + Add<Output = F::Vector> + Clone,
    f64: Mul<F::Vector, Output = F::Vector>,
{
    type Vector = F::Vector;

    fn minimize(mut self, x0: Self::Vector) -> Result<Returns<Self::Vector>, Error> {

        let mut xk = x0.clone();
        let mut xk_prev = x0.clone();
        let mut fk = self.fcn.eval(&xk);   
        let mut grad_fk = self.fcn.grad(&xk);
        let mut grad_fk_prev = grad_fk.clone(); 
        let mut direction = grad_fk.clone();


        let max_iter = self.opts.uncon_opts.max_iter;

        let mut line_searcher = ls::create(x0, grad_fk.clone(), 
                                           self.fcn, self.opts.uncon_opts.ls_method);

        for i in 0..max_iter {
            info!(target: "cg", "............................ i = {i}");

            let phi0 = fk;
            let dphi0 = grad_fk.dot(&direction);
            let ls_ret = line_searcher.search(phi0, dphi0)?;



        }

        Err(Error::MaxIterations(0))
    }
}