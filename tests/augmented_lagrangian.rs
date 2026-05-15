#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(clippy::excessive_precision)]

//{{{ crate imports
use topohedral_optimize::constrained::{
    minimize as constrained_minimize, AugmentedLagrangianOptions, ConstrainedMethod,
    ConstrainedReturns, ConstriainedOptions,
};
use topohedral_optimize::constraints::{BoundsConstraints, NoConstraints};
use topohedral_optimize::line_search::{
    LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
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
#[case::quadratic_thuente_bfgs(UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 7, 10)]
#[case::quadratic_nocedal_bfgs(UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 13, 16)]
#[case::quadratic_thuente_fr(UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 14, 23)]
#[case::quadratic_nocedal_fr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 15, 21)]
#[case::quadratic_thuente_pr(UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 9, 13)]
#[case::quadratic_nocedal_pr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 13, 17)]
fn test_quadratic_without_constraints_matches_unconstrained_reference(
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quad = Quadratic {
        xmin: colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };

    let x0 = colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]);

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
        1e-5,
        1e-7,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: bound constrained
#[rstest]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-6, 1e-4,  115, 164)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-6, 1e-4,  121, 151)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-6, 1e-4,  128, 183)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  2e-6, 1e-4,  127, 156)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-6, 1e-4,  122, 174)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-6, 1e-4,  123, 152)]
fn test_quadratic_with_bound_constraints_matches_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
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
#[case::quartic_thuente_bfgs(UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 19, 37)]
#[case::quartic_nocedal_bfgs(UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 28, 39)]
#[case::quartic_thuente_fr(UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 72, 105)]
#[case::quartic_nocedal_fr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 140, 190)]
#[case::quartic_thuente_pr(UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 100, 139)]
#[case::quartic_nocedal_pr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 221, 274)]
fn test_quartic_without_constraints_matches_unconstrained_reference(
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let x0_in = colvec(&[11.0, 11.0, 11.0, 11.0, 11.0]);
    let ret = constrained_minimize(
        quart,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0_in,
        auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("ret = {ret:?}");
    assert_answer(
        &ret,
        &colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        0.0,
        1e-2,
        1e-7,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: bound constrained
#[rstest]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-2, 1e-2, 1532, 2314)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-2, 1e-2, 2757, 1936)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-2, 1e-2, 1862, 2675)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  1e-2, 1e-2, 4243, 2151)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-2, 1e-2, 1971, 2819)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-2, 1e-2, 3744, 1812)]
fn test_quartic_with_bound_constraints_matches_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
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
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  36, 61)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  39, 41)]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),   98, 159)]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),   151, 200)]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),   210, 302)]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),   562, 487)]
fn test_rosenbrock_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let rosenbrock = Rosenbrock::new();
    let ret = constrained_minimize(
        rosenbrock,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap();
    println!("ret = {ret:?}");
    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, 1e-3, 1e-5);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
