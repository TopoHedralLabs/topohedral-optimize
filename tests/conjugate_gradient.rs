#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

//{{{ crate imports
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    ConjugateGradient, ConjugateGradientOptions, Direction, UnconstrainedConvergedReason,
    UnconstrainedMinimizer, UnconstrainedReturns, UnonstrainedOptions,
};
use topohedral_optimize::RealFn;
//}}}
//{{{ std imports
use std::ops::Sub;
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_linalg::{scvector::SCVector, VectorOps};
use topohedral_tracing::*;
//}}}

//{{{ fun: init_logger
#[ctor]
fn init_logger()
{
    init().unwrap();
}
//}}}
//{{{ struct: Quadratic
#[derive(Debug, Clone, Copy)]
struct Quadratic
{
    xmin: SCVector<f64, 5>,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn for Quadratic
{
    type Vector = SCVector<f64, 5>;

    fn eval(
        &mut self,
        x: &Self::Vector,
    ) -> f64
    {
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5
        {
            out += tmp[i].powi(2);
        }
        out
    }

    fn grad(
        &mut self,
        x_in: &Self::Vector,
    ) -> Self::Vector
    {
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros();
        for i in 0..5
        {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}
//{{{ struct: Quartic
#[derive(Debug, Clone, Copy)]
struct Quartic
{
    xmin: SCVector<f64, 5>,
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn for Quartic
{
    type Vector = SCVector<f64, 5>;

    fn eval(
        &mut self,
        x: &Self::Vector,
    ) -> f64
    {
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5
        {
            out += tmp[i].powi(4);
        }
        out
    }

    fn grad(
        &mut self,
        x_in: &Self::Vector,
    ) -> Self::Vector
    {
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros();
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
    type Vector = SCVector<f64, 2>;

    fn eval(
        &mut self,
        xvec: &Self::Vector,
    ) -> f64
    {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn grad(
        &mut self,
        xvec: &Self::Vector,
    ) -> Self::Vector
    {
        let a = self.a;
        let b = self.b;
        let x = xvec[0];
        let y = xvec[1];
        let mut out = SCVector::<f64, 2>::zeros();
        out[0] = -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2));
        out[1] = 2.0 * b * (y - x.powi(2));
        out
    }
}
//}}}
//{{{ fun: assert_returns
fn assert_returns<T>(
    ret: &UnconstrainedReturns<T>,
    exp_ret: &UnconstrainedReturns<T>,
    xmin_tol: f64,
    fmin_tol: f64,
) where
    T: VectorOps<ScalarType = f64> + Sub<Output = T> + Clone,
{
    assert!((ret.xmin.clone() - exp_ret.xmin.clone()).norm() < xmin_tol);
    assert!((ret.fmin - exp_ret.fmin).abs() < fmin_tol);
    assert_eq!(ret.reason, exp_ret.reason);
    assert_eq!(ret.num_iterations, exp_ret.num_iterations);
    assert_eq!(ret.num_fun_evals, exp_ret.num_fun_evals);
    assert_eq!(ret.num_grad_evals, exp_ret.num_grad_evals);
}
//}}}
//{{{ const: INTERP_STEEPEST
const INTERP_STEEPEST: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Interp(InterpOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            scale_factor: 1.5,
            maxiter: 10,
        }),
    },
    direction: Direction::Steepest,
    restart: 10,
};
//}}}
//{{{ const: INTPER_FR
const INTERP_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Interp(InterpOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            scale_factor: 1.5,
            maxiter: 10,
        }),
    },
    direction: Direction::FletcherReeves,
    restart: 10,
};
//}}}
//{{{ const: INTERP_PR
const INTERP_PR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Interp(InterpOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            scale_factor: 1.5,
            maxiter: 10,
        }),
    },
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}
//{{{ const: THUENTE_STEEPEST
const THUENTE_STEEPEST: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Thuente(ThuenteOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
        }),
    },
    direction: Direction::Steepest,
    restart: 10,
};
//}}}
//{{{ const: THUENTE_FR
const THUENTE_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Thuente(ThuenteOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
        }),
    },
    direction: Direction::FletcherReeves,
    restart: 10,
};
//}}}
//{{{ const: THUENTE_PR
const THUENTE_PR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        ls_method: LineSearchMethod::Thuente(ThuenteOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
        }),
    },
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}
//{{{ test: quadratic
#[rstest]
//{{{ case: test_quadratic_interp_steepest
#[case::test_quadratic_interp_steepest(
    SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    INTERP_STEEPEST,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 6, 
        num_grad_evals: 3
    }
)]
//}}}
//{{{ case: test_quadratic_thuente_steepest
#[case::test_quadratic_thuente_steepest(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_STEEPEST,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 7, 
        num_grad_evals: 8
    }
)]
//}}}
//{{{ case: test_quadratic_interp_fr
#[case::test_quadratic_interp_fr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    INTERP_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 6, 
        num_grad_evals: 3,
    }
)]
//}}}
//{{{ case: test_quadratic_thuente_fr
#[case::test_quadratic_thuente_fr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 7, 
        num_grad_evals: 8,
    }
)]
//}}}
//{{{ case: test_quadratic_interp_pr
#[case::test_quadratic_interp_pr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    INTERP_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 6, 
        num_grad_evals: 3,
    }
)]
//}}}
//{{{ case: test_quadratic_thuente_pr
#[case::test_quadratic_interp_pr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 7, 
        num_grad_evals: 8,
    }
)]
//}}}
fn test_qudratic(
    #[case] x0: SCVector<f64, 5>,
    #[case] opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns<SCVector<f64, 5>>,
)
{
    let quad = Quadratic {
        xmin: SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };
    let mut cg = ConjugateGradient::new(quad, x0, opts);
    let ret = cg.minimize().unwrap();

    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-7, 1e-10);
}
//}}}
//{{{ test: quartic
#[rstest]
//{{{ case: test_quartic_interp_steepest
#[case::test_quartic_interp_steepest(
    SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    INTERP_STEEPEST,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 14,
        num_fun_evals: 59,
        num_grad_evals: 29
    }
)]
//}}}
//{{{ case: test_quartic_thuente_steepest
#[case::test_quartic_thuente_steepest(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_STEEPEST,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 12,
        num_fun_evals: 40,
        num_grad_evals: 52
    }
)]
//}}}
//{{{ case: test_quartic_interp_fr
#[case::test_quartic_interp_fr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    INTERP_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 16,
        num_fun_evals: 152,
        num_grad_evals: 49,
    }
)]
//}}}
//{{{ case: test_quartic_thuente_fr
#[case::test_quartic_thuente_fr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 13,
        num_fun_evals: 35,
        num_grad_evals: 48,
    }
)]
//}}}
//{{{ case: test_quartic_interp_pr
#[case::test_quartic_interp_pr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    INTERP_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 14,
        num_fun_evals: 177,
        num_grad_evals: 52,
    }
)]
//}}}
//{{{ case: test_quartic_thuente_pr
#[case::test_quartic_thuente_pr(
    SCVector::<f64, 5>::from_col_slice(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 12,
        num_fun_evals: 40,
        num_grad_evals: 52,
    }
)]
//}}}
fn test_quartic(
    #[case] x0: SCVector<f64, 5>,
    #[case] mut opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns<SCVector<f64, 5>>,
)
{
    let quart = Quartic {
        xmin: SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };
    opts.uncon_opts.grad_rtol = 1e-12;
    opts.uncon_opts.grad_atol = 1e-12;
    let mut cg = ConjugateGradient::new(quart, x0, opts);
    let ret = cg.minimize().unwrap();

    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 5e-2, 1e-5);
}
//}}}
//{{{ test: rosenbrock
#[rstest]
//{{{ case: test_rosenbrock_interp_fr
#[case::test_rosenbrock_interp_pr(
    SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]),
    INTERP_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 2>::from_col_slice(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 2538,
        num_fun_evals: 12745,
        num_grad_evals: 5088,
    }
)]
//}}}
//{{{ case: test_rosenbrock_thuente_fr
#[case::test_rosenbrock_thuente_fr(
    SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 2>::from_col_slice(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 119,
        num_fun_evals: 235,
        num_grad_evals: 354,
    }
)]
//}}}
//{{{ case: test_rosenbrock_interp_pr
#[case::test_rosenbrock_interp_pr(
    SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]),
    INTERP_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 2>::from_col_slice(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 16,
        num_fun_evals: 161,
        num_grad_evals: 49,
    }
)]
//}}}
//{{{ case: test_rosenbrock_thuente_pr
#[case::test_rosenbrock_thuente_pr(
    SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 2>::from_col_slice(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 19,
        num_fun_evals: 40,
        num_grad_evals: 59,
    }
)]
//}}}
fn test_rosenbrock(
    #[case] x0: SCVector<f64, 2>,
    #[case] mut opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns<SCVector<f64, 2>>,
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

    let mut cg = ConjugateGradient::new(rosenbrock, x0, opts);

    let ret = cg.minimize().unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-2, 1e-6);
}
//}}}
