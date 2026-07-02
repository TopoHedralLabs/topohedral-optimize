#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::line_search::{
    LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    minimize, QuasiNewtonOptions, UnconstrainedConvergedReason, UnconstrainedMethod,
    UnconstrainedReturns, UnonstrainedOptions, UpdateMethod,
};
use topohedral_optimize::{RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_linalg::VectorOps;
use topohedral_linalg::{DVector, VecType};
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
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
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
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
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
        let mut out = DVector::<f64>::zeros_vec(2, VecType::Col);
        out[0] = -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2));
        out[1] = 2.0 * b * (y - x.powi(2));
        out
    }
}
//}}}
//{{{ fun: assert_returns
fn assert_returns(
    ret: &UnconstrainedReturns,
    exp_ret: &UnconstrainedReturns,
    xmin_tol: f64,
    fmin_tol: f64,
)
{
    assert!((ret.xmin.clone() - exp_ret.xmin.clone()).norm() < xmin_tol);
    assert!((ret.fmin - exp_ret.fmin).abs() < fmin_tol);
    assert_eq!(ret.reason, exp_ret.reason);
    assert_eq!(ret.num_iterations, exp_ret.num_iterations);
    assert_eq!(ret.num_fun_evals, exp_ret.num_fun_evals);
    assert_eq!(ret.num_grad_evals, exp_ret.num_grad_evals);
}
//}}}
//{{{ const: THUENTE_BFGS
const THUENTE_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
    },
    ls_method: LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.9,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
    }),
    method: UpdateMethod::BFGS,
    restart: 10,
};
//}}}
//{{{ const: NOCEDAL_BFGS
const NOCEDAL_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
    },
    ls_method: LineSearchMethod::Nocedal(NocedalOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.9,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
        zoom_maxiter: 10,
    }),
    method: UpdateMethod::BFGS,
    restart: 10,
};
//}}}

//{{{ test: quadratic
#[rstest]
//{{{ case: test_thuente_bfgs
#[case::test_quadratic_thuente_bfgs(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 2,
        num_fun_evals: 7,
        num_grad_evals: 10
    }
)]
//}}}
//{{{ case: test_nocedal_bfgs
#[case::test_quadratic_nocedal_bfgs(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 3,
        num_fun_evals: 11,
        num_grad_evals: 15
    }
)]
//}}}
fn test_qudratic(
    #[case] x0: Vector,
    #[case] opts: QuasiNewtonOptions,
    #[case] exp_ret: UnconstrainedReturns,
)
{
    let quad = Quadratic {
        xmin: colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };
    let ret = minimize(quad, x0, UnconstrainedMethod::QuasiNewton(opts)).unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-7, 1e-10);
}
//}}}
//{{{ test: quartic
#[rstest]
//{{{ case: test_nocedal_bfgs
#[case::test_quartic_nocedal_bfgs(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 59,
        num_fun_evals: 73,
        num_grad_evals: 132
    }
)]
//}}}
//{{{ case: test_thuente_bfgs
#[case::test_quartic_thuente_bfgs(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 57,
        num_fun_evals: 63,
        num_grad_evals: 121,
    }
)]
//}}}
fn test_quartic(
    #[case] x0: Vector,
    #[case] mut opts: QuasiNewtonOptions,
    #[case] exp_ret: UnconstrainedReturns,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };
    opts.uncon_opts.grad_rtol = 1e-12;
    opts.uncon_opts.grad_atol = 1e-12;
    opts.uncon_opts.max_iter = 1000;
    let ret = minimize(quart, x0, UnconstrainedMethod::QuasiNewton(opts)).unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 5e-2, 1e-5);
}
//}}}
//{{{ test: rosenbrock
#[rstest]
//{{{ case: test_thuente_bfgs
#[case::test_thuente_bfgs(
    colvec(&[0.0, 3.0]),
    THUENTE_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 26,
        num_fun_evals: 34,
        num_grad_evals: 61
    }
)]
//}}}
//{{{ case: test_nocedal_bfgs
#[case::test_nocedal_bfgs(
    colvec(&[0.0, 3.0]),
    NOCEDAL_BFGS,
    UnconstrainedReturns{
        xmin: colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 27,
        num_fun_evals: 38,
        num_grad_evals: 61
    }
)]
//}}}
fn test_rosenbrock(
    #[case] x0: Vector,
    #[case] mut opts: QuasiNewtonOptions,
    #[case] exp_ret: UnconstrainedReturns,
)
{
    let rosenbrock = Rosenbrock::new();

    opts.uncon_opts.grad_rtol = 1e-6;
    opts.uncon_opts.grad_atol = 1e-10;
    opts.uncon_opts.max_iter = 10000;

    // let mut qn = QuasiNewton::new(rosenbrock, x0, opts);
    let ret = minimize(rosenbrock, x0, UnconstrainedMethod::QuasiNewton(opts)).unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-2, 1e-6);
}
//}}}
