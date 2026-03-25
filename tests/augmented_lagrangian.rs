#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    QuasiNewtonOptions, UnconstrainedConvergedReason, UnconstrainedMethod, UnconstrainedReturns,
    UnonstrainedOptions, UpdateMethod,
};
use topohedral_optimize::{RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_linalg::dvector::{DVector, VecType};
use topohedral_linalg::VectorOps;
use topohedral_tracing::*;
//}}}

//{{{ fun: init_logger
#[ctor]
fn init_logger()
{
    init().unwrap();
}
//}}}
//{{{ fun: colvec
fn colvec(values: &[f64]) -> Vector
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}
//}}}
//{{{ struct: Quadratic
#[derive(Debug, Clone)]
struct Quadratic
{
    xmin: Vector,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn for Quadratic
{
    fn dimension(&self) -> usize
    {
        self.xmin.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let tmp = x.clone() - self.xmin.clone();
        let mut out = 0.0;
        for i in 0..5
        {
            out += tmp[i].powi(2);
        }
        out
    }

    fn grad(
        &mut self,
        x_in: &Vector,
    ) -> Vector
    {
        let tmp = x_in.clone() - self.xmin.clone();
        let mut out = DVector::<f64>::zeros_cvec(5, VecType::Col);
        for i in 0..5
        {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}
//{{{ struct: Quartic
#[derive(Debug, Clone)]
struct Quartic
{
    xmin: Vector,
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn for Quartic
{
    fn dimension(&self) -> usize
    {
        self.xmin.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let tmp = x.clone() - self.xmin.clone();
        let mut out = 0.0;
        for i in 0..5
        {
            out += tmp[i].powi(4);
        }
        out
    }

    fn grad(
        &mut self,
        x_in: &Vector,
    ) -> Vector
    {
        let tmp = x_in.clone() - self.xmin.clone();
        let mut out = DVector::<f64>::zeros_cvec(5, VecType::Col);
        for i in 0..5
        {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
//}}}
//{{{ struct Rosenbrock
#[derive(Debug, Clone, Copy)]
struct Rosenbrock
{
    a: f64,
    b: f64,
}
//}}}
//{{{ impl: Rosenbrock
impl Rosenbrock
{
    fn new() -> Self
    {
        Self { a: 1.0, b: 100.0 }
    }
}
//}}}
//{{{ impl: RealFn for Rosenbrock
impl RealFn for Rosenbrock
{
    fn dimension(&self) -> usize
    {
        2
    }

    fn eval(
        &mut self,
        xvec: &Vector,
    ) -> f64
    {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn grad(
        &mut self,
        xvec: &Vector,
    ) -> Vector
    {
        let a = self.a;
        let b = self.b;
        let x = xvec[0];
        let y = xvec[1];
        let mut out = DVector::<f64>::zeros_cvec(2, VecType::Col);
        out[0] = -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2));
        out[1] = 2.0 * b * (y - x.powi(2));
        out
    }
}
//}}}
