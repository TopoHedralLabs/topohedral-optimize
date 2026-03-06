#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    create, Method, QuasiNewton, QuasiNewtonOptions, UnconstrainedConvergedReason,
    UnconstrainedMinimizer, UnconstrainedReturns, UnonstrainedOptions, UpdateMethod,
};
use topohedral_optimize::RealFn;
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
fn colvec(values: &[f64]) -> DVector<f64>
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}
//}}}
//{{{ struct: Quadratic
#[derive(Debug, Clone)]
struct Quadratic
{
    xmin: DVector<f64>,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn for Quadratic
{
    fn eval(
        &mut self,
        x: &DVector<f64>,
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
        x_in: &DVector<f64>,
    ) -> DVector<f64>
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
    xmin: DVector<f64>,
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn for Quartic
{
    fn eval(
        &mut self,
        x: &DVector<f64>,
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
        x_in: &DVector<f64>,
    ) -> DVector<f64>
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
    fn eval(
        &mut self,
        xvec: &DVector<f64>,
    ) -> f64
    {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn grad(
        &mut self,
        xvec: &DVector<f64>,
    ) -> DVector<f64>
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
//{{{ const: INTERP_BFS
const INTERP_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Interp(InterpOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-8,
                step_max: 1e5,
            },
            scale_factor: 1.5,
            maxiter: 10,
        }),
    },
    method: UpdateMethod::BFGS,
    restart: 10,
};
//}}}
//{{{ const: THUENTE_BFGS
const THUENTE_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Thuente(ThuenteOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
        }),
    },
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
    },
    method: UpdateMethod::BFGS,
    restart: 10,
};
//}}}

//{{{ test: quadratic
#[rstest]
//{{{ case: test_interp_bfgs
#[case::test_quadratic_interp_bfgs(
    colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    INTERP_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1,
        num_fun_evals: 6,
        num_grad_evals: 3
    }
)]
//}}}
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
        num_grad_evals: 9
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
        num_grad_evals: 14
    }
)]
//}}}
fn test_qudratic(
    #[case] x0: DVector<f64>,
    #[case] opts: QuasiNewtonOptions,
    #[case] exp_ret: UnconstrainedReturns,
)
{
    let quad = Quadratic {
        xmin: colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };
    let mut qn = QuasiNewton::new(quad, x0, opts);
    let ret = qn.minimize().unwrap();

    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-7, 1e-10);
}
//}}}
//{{{ test: quartic
#[rstest]
//{{{ case: test_interp_bfgs
#[case::test_quartic_interp_bfgs(
    colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    INTERP_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 19,
        num_fun_evals: 77,
        num_grad_evals: 39
    }
)]
//}}}
//{{{ case: test_nocedal_bfgs
#[case::test_quartic_nocedal_bfgs(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 66,
        num_fun_evals: 92,
        num_grad_evals: 150
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
        num_fun_evals: 72,
        num_grad_evals: 129,
    }
)]
//}}}
fn test_quartic(
    #[case] x0: DVector<f64>,
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
    let mut qn = QuasiNewton::new(quart, x0, opts);
    let ret = qn.minimize().unwrap();

    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 5e-2, 1e-5);
}
//}}}
//{{{ test: rosenbrock
#[rstest]
//{{{ case: test_interp_bfgs
#[case::test_interp_bfgs(
    colvec(&[0.0, 3.0]),
    INTERP_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 19,
        num_fun_evals: 226,
        num_grad_evals: 66
    }
)]
//}}}
//{{{ case: test_thuente_bfgs
#[case::test_thuente_bfgs(
    colvec(&[0.0, 3.0]),
    THUENTE_BFGS,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 24,
        num_fun_evals: 36,
        num_grad_evals: 60
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
        num_iterations: 25,
        num_fun_evals: 53,
        num_grad_evals: 60
    }
)]
//}}}
fn test_rosenbrock(
    #[case] x0: DVector<f64>,
    #[case] mut opts: QuasiNewtonOptions,
    #[case] exp_ret: UnconstrainedReturns,
)
{
    let rosenbrock = Rosenbrock::new();

    opts.uncon_opts.grad_rtol = 1e-6;
    opts.uncon_opts.grad_atol = 1e-10;
    opts.uncon_opts.max_iter = 10000;
    if let LineSearchMethod::Interp(interp_opts) = &mut opts.uncon_opts.ls_method
    {
        interp_opts.scale_factor = 1.2;
    }

    // let mut qn = QuasiNewton::new(rosenbrock, x0, opts);
    let mut qn = create(rosenbrock, x0, Method::QuasiNewton(opts));

    let ret = qn.minimize().unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-2, 1e-6);
}
//}}}
