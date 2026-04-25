#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(clippy::excessive_precision)]

//{{{ crate imports
use topohedral_optimize::constrained::{
    minimize as constrained_minimize, AugmentedLagrangianOptions, BoundsConstraints,
    ConstrainedMethod, ConstrainedReturns, ConstriainedOptions, NoConstraints,
};
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    ConjugateGradientOptions, Direction, QuasiNewtonOptions, UnconstrainedMethod,
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
//{{{ collection: constants
const UNIT_SCALE: f64 = 1.0;
//}}}
//{{{ fn: reldiff
fn reldiff(
    a: f64,
    b: f64,
    atol: f64,
    rtol: f64,
) -> f64
{
    let scale = b.abs().max(UNIT_SCALE);
    let tol = atol + rtol * scale;
    if tol == 0.0
    {
        return if a == b { 0.0 } else { f64::INFINITY };
    }
    (a - b).abs() / tol
}
//}}}
//{{{ fn: vec_reldiff
fn vec_reldiff(
    a: &Vector,
    b: &Vector,
    atol: f64,
    rtol: f64,
) -> f64
{
    let scale = b.norm().max(UNIT_SCALE);
    let tol = atol + rtol * scale;
    let diff = (a.clone() - b.clone()).norm();
    if tol == 0.0
    {
        return if diff == 0.0 { 0.0 } else { f64::INFINITY };
    }
    diff / tol
}
//}}}
//{{{ fun: assert_answer
fn assert_answer(
    ret: &ConstrainedReturns,
    exp_xmin: &Vector,
    exp_fmin: f64,
    xmin_tol: f64,
    fmin_tol: f64,
)
{
    let xmin_err = vec_reldiff(&ret.xmin, exp_xmin, xmin_tol, xmin_tol);
    let fmin_err = reldiff(ret.fmin, exp_fmin, fmin_tol, fmin_tol);
    println!("xmin_err = {xmin_err:1.4e} fmin_err = {fmin_err:1.4e}");
    assert!(xmin_err <= 1.0);
    assert!(fmin_err <= 1.0);
}
//}}}
//{{{ fun: assert_counts
fn assert_counts(
    ret: &ConstrainedReturns,
    exp_num_fun_evals: usize,
    exp_num_grad_evals: usize,
)
{
    assert_eq!(ret.num_fun_evals, exp_num_fun_evals);
    assert_eq!(ret.num_grad_evals, exp_num_grad_evals);
}
//}}}
//{{{ fun: auglag_method
fn auglag_method(mut unconstrained_method: UnconstrainedMethod) -> ConstrainedMethod
{
    unconstrained_method.uncon_opts_mut().make_counting = false;

    ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            grad_rtol: 1e-6,
            grad_atol: 1e-8,
            constraint_tol: 1e-6,
            max_iter: 1000,
            make_counting: true,
        },
        unconstrained_method,
        1.0,
        0.9,
        2.5,
    ))
}
//}}}
//{{{ const: THUENTE_BFGS
const THUENTE_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: false,
        ls_method: LineSearchMethod::Thuente(ThuenteOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 100,
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
        make_counting: false,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 20,
            zoom_maxiter: 10,
        }),
    },
    method: UpdateMethod::BFGS,
    restart: 10,
};
//}}}
//{{{ const: THUENTE_FR
const THUENTE_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: false,
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
        make_counting: false,
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
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}
//{{{ const: NOCEDAL_FR
const NOCEDAL_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: false,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 20,
            zoom_maxiter: 20,
        }),
    },
    direction: Direction::FletcherReeves,
    restart: 10,
};
//}}}
//{{{ const: NOCEDAL_PR
const NOCEDAL_PR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: false,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 20,
            zoom_maxiter: 20,
        }),
    },
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}

//{{{ collection: quadratic
//{{{ test: unconstrained
#[rstest]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 1.0e-12, 1.0e-24, 12, 14)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 1.0e-12, 1.0e-24, 18, 20)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 2.5423787284960917e-13, 6.4636895991094044e-26, 19, 27)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 1.9428902930940239e-16, 3.7748226909989823e-32, 20, 25)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 2.5423787284960917e-13, 6.4636895991094044e-26, 14, 17)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 0.0, 0.0, 18, 21)]
fn test_quadratic_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quad = Quadratic {
        xmin: colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };

    let ret = constrained_minimize(
        quad,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("ret = {ret:?}");

    assert_answer(
        &ret,
        &colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        0.0,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: bound constrained
#[rstest]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), false, 1e-6, 1e-4,  41, 77)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), false, 1e-6, 1e-4,  46, 64)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), false, 1e-6, 1e-4,  49, 87)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), false, 2e-6, 1e-4,  130, 150)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), false, 1e-6, 1e-4,  38, 66)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), false, 1e-6, 1e-4,  60, 61)]
fn test_quadratic_with_bound_constraints_matches_reference(
    #[case] x0: Vector,
    #[case] mut unconstrained_method: UnconstrainedMethod,
    #[case] use_interp_scale_factor_1_2: bool,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quad = Quadratic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let mut ieq_constraints = BoundsConstraints::new(5);
    ieq_constraints.add_bounds(0, Some(20.0), None);

    if use_interp_scale_factor_1_2
    {
        if let LineSearchMethod::Interp(interp_opts) =
            &mut unconstrained_method.uncon_opts_mut().ls_method
        {
            interp_opts.scale_factor = 1.2;
        }
    }

    let ret = constrained_minimize(
        quad,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("ret = {ret:?}");
    let exp_fmin = 100.0;
    assert_answer(
        &ret,
        &colvec(&[20.0, 10.0, 10.0, 10.0, 10.0]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
//{{{ collection: quartic
//{{{ test: unconstrained
#[rstest]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 6.8e-1, 1.217166e-1, 50, 87)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 7.3232511374922560e-1, 6.9775029981934830e-2, 75, 103)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 4.3181595955113955e-1, 8.7598881523012430e-3, 24, 37)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 4.1150527192591912e-1, 1.0639361825963145e-2, 34, 40)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 4.4101768233767547e-1, 1.3287153804932000e-2, 22, 34)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 6.5889078801448453e-1, 7.9115661470400640e-2, 42, 48)]
fn test_quartic_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] mut unconstrained_method: UnconstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    unconstrained_method.uncon_opts_mut().max_iter = 1000;

    let ret = constrained_minimize(
        quart,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("ret = {ret:?}");
    assert_answer(
        &ret,
        &colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        0.0,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: bound constrained
#[rstest]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), false, 1.0, 1e-6, 1e-6, 40, 77)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), false, 1.0, 1e-2, 1e-3, 45, 64)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), false, 1.0, 1.0e-2, 1.0e-3, 48, 87)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), false, 1.0, 2e-1, 1e-3, 129, 150)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), false, 1.0, 5e-2, 1e-3, 85, 135)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), false, 1.0, 5e-2, 1e-3, 113, 107)]
fn test_quartic_with_bound_constraints_matches_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] use_interp_scale_factor_1_2: bool,
    #[case] initial_penalty: f64,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let mut method = auglag_method(unconstrained_method);
    method.con_opts_mut().constraint_tol = 1e-3;
    method.con_opts_mut().grad_rtol = 1e-4;

    if use_interp_scale_factor_1_2
    {
        if let LineSearchMethod::Interp(interp_opts) =
            &mut method.uncon_method_mut().uncon_opts_mut().ls_method
        {
            interp_opts.scale_factor = 1.2;
        }
    }

    let ConstrainedMethod::AugmentedLagrangian(opts) = &mut method;
    opts.initial_penalty = initial_penalty;

    let mut ieq_constraints = BoundsConstraints::new(5);
    ieq_constraints.add_bounds(0, Some(20.0), None);

    let ret = constrained_minimize(
        quart,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        method,
    )
    .unwrap();

    println!("ret = {ret:?}");
    let exp_fmin = 10000.0;
    assert_answer(
        &ret,
        &colvec(&[20.0, 10.0, 10.0, 10.0, 10.0]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
//{{{ collection: rosenbrock
//{{{ test: unconstrained
#[rstest]
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 9.0284228790531235e-7, 3.3156139251068023e-13, 35, 59)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 1.2741169516945091e-7, 8.0476763718045500e-15, 74, 68)]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 2.3939387271972799e-4, 1.1444703681763757e-8, 405, 641)]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 4.1115183059162496e-5, 3.3755631595392825e-10, 498, 614)]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 5.4859043189398348e-6, 6.5279974297398110e-12, 62, 91)]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 2.8080039179491367e-4, 1.5745787517370056e-8, 68, 75)]
fn test_rosenbrock_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] mut unconstrained_method: UnconstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let rosenbrock = Rosenbrock::new();

    unconstrained_method.uncon_opts_mut().grad_rtol = 1e-6;
    unconstrained_method.uncon_opts_mut().grad_atol = 1e-10;
    unconstrained_method.uncon_opts_mut().max_iter = 10000;
    if let LineSearchMethod::Interp(interp_opts) =
        &mut unconstrained_method.uncon_opts_mut().ls_method
    {
        interp_opts.scale_factor = 1.2;
    }

    let ret = constrained_minimize(
        rosenbrock,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("ret = {ret:?}");
    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, xmin_tol, fmin_tol);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
