#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![allow(clippy::excessive_precision)]

use topohedral_optimize::bound_constrained::{
    AsaOptions, BfgsbOptions, BoundConstrainedMethod, BoundConstrainedOptions,
};
//{{{ crate imports
use topohedral_optimize::constrained::{
    minimize as constrained_minimize, AugmentedLagrangianInnerMethod, AugmentedLagrangianOptions,
    ConstrainedMethod, ConstrainedReturns, ConstriainedOptions,
};
use topohedral_optimize::constraints::{BoundsConstraints, NoConstraints};
use topohedral_optimize::line_search::{
    LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    ConjugateGradientOptions, Direction, QuasiNewtonOptions, UnconstrainedMethod,
    UnonstrainedOptions, UpdateMethod,
};
use topohedral_optimize::{BaseOptions, RealFn, RealVectorFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_linalg::{DVector, VecType};
use topohedral_linalg::{SubViewableMut, VectorOps};
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
    println!("xmin_err = {xmin_err:.4e} fmin_err = {fmin_err:.4e}");
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
    assert!(
        ret.num_fun_evals <= exp_num_fun_evals,
        "expected at most {} function evaluations, got {}",
        exp_num_fun_evals,
        ret.num_fun_evals
    );
    assert!(
        ret.num_grad_evals <= exp_num_grad_evals,
        "expected at most {} gradient evaluations, got {}",
        exp_num_grad_evals,
        ret.num_grad_evals
    );
}
//}}}
//{{{ fun: uncon_auglag_method
fn uncon_auglag_method(mut unconstrained_method: UnconstrainedMethod) -> ConstrainedMethod
{
    unconstrained_method.uncon_opts_mut().make_counting = false;

    ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            base_opts: UnonstrainedOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-6,
                max_iter: 1000,
                make_counting: true,
            },
            constraint_tol: 1e-6,
        },
        AugmentedLagrangianInnerMethod::Unconstrained(unconstrained_method),
    ))
}
//}}}
//{{{ fun: bcon_auglag_method
fn bcon_auglag_method(mut bcon_method: BoundConstrainedMethod) -> ConstrainedMethod
{
    bcon_method.bound_opts_mut().base_opts.make_counting = false;

    ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            base_opts: UnonstrainedOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-6,
                max_iter: 1000,
                make_counting: true,
            },
            constraint_tol: 1e-6,
        },
        AugmentedLagrangianInnerMethod::BoundConstrained(bcon_method),
    ))
}
//}}}
//{{{ const: THUENTE_OPTS_09
const THUENTE_OPTS_09: ThuenteOptions = ThuenteOptions {
    ls_opts: LineSearchOptions {
        c1: 1e-4,
        c2: 0.9,
        step_min: 1e-8,
        step_max: 1e5,
    },
    maxiter: 100,
};
//}}}
//{{{ const: NOCEDAL_OPTS_04
const NOCEDAL_OPTS_04: NocedalOptions = NocedalOptions {
    ls_opts: LineSearchOptions {
        c1: 1e-4,
        c2: 0.4,
        step_min: 1e-8,
        step_max: 1e5,
    },
    maxiter: 100,
    zoom_maxiter: 10,
};
//}}}
//{{{ const: THUENTE_BFGS
const THUENTE_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: false,
    },
    ls_method: LineSearchMethod::Thuente(THUENTE_OPTS_09),
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
    },
    ls_method: LineSearchMethod::Nocedal(NOCEDAL_OPTS_04),
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
    },
    ls_method: LineSearchMethod::Thuente(THUENTE_OPTS_09),
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
    },
    ls_method: LineSearchMethod::Thuente(THUENTE_OPTS_09),
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
    },
    ls_method: LineSearchMethod::Nocedal(NOCEDAL_OPTS_04),
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
    },
    ls_method: LineSearchMethod::Nocedal(NOCEDAL_OPTS_04),
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}
//{{{ const: ASA
const ASA_NOCEDAL_PR: AsaOptions = AsaOptions {
    bound_opts: BoundConstrainedOptions {
        base_opts: BaseOptions {
            grad_rtol: 1e-8,
            grad_atol: 1e-10,
            max_iter: 100,
            make_counting: false,
        },
        constraint_tol: 1e-6,
    },
    unconstrained_method: UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),
    mu: 0.1,
    rho: 0.5,
    n1: 2,
    n2: 1,
    memory: 8,
    delta: 1e-4,
    eta: 0.5,
    alpha_min: 1e-20,
    alpha_max: 1e20,
};
//}}}
//{{{ const: BFGSB
const BFGSB_NOCEDAL: BfgsbOptions = BfgsbOptions {
    bound_opts: BoundConstrainedOptions {
        base_opts: BaseOptions {
            grad_rtol: 1e-8,
            grad_atol: 1e-10,
            max_iter: 100,
            make_counting: false,
        },
        constraint_tol: 1e-6,
    },
    ls_method: LineSearchMethod::Nocedal(NOCEDAL_OPTS_04),
};
//}}}

//{{{ collection: constraints
#[derive(Debug, Clone)]
struct HyperSphereBound
{
    center: Vector,
    radius: f64,
}
impl RealVectorFn for HyperSphereBound
{
    fn dimension_domain(&self) -> usize
    {
        self.center.len()
    }

    fn dimension_range(&self) -> usize
    {
        1
    }

    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
        let mut value = 0.0;
        for i in 0..x.len()
        {
            value += (x[i] - self.center[i]).powi(2)
        }
        (*val)[0] = value - self.radius.powi(2);
    }

    fn grad(
        &mut self,
        x: &Vector,
        val: &mut topohedral_optimize::Matrix,
    )
    {
        val.col_mut(0).copy_from(2.0 * (x - &self.center));
    }
}
//}}}

//{{{ collection: quadratic
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
//{{{ test: test_quadratic_unconstrained
#[rstest]
#[case::quadratic_thuente_bfgs(UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 12, 14)]
#[case::quadratic_nocedal_bfgs(UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 18, 20)]
#[case::quadratic_thuente_fr(UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 19, 27)]
#[case::quadratic_nocedal_fr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 19, 20)]
#[case::quadratic_thuente_pr(UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 14, 17)]
#[case::quadratic_nocedal_pr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 19, 20)]
fn test_quadratic_unconstrained(
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
        None,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        uncon_auglag_method(unconstrained_method),
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
//{{{ test: test_quadratic_bound_constrained_ucon_inner
#[rstest]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-6, 1e-4,  115, 165)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-6, 1e-4,  122, 154)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-6, 1e-4,  130, 186)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  2e-6, 1e-4,  127, 151)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-6, 1e-4,  122, 174)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-6, 1e-4,  123, 147)]
fn test_quadratic_bound_constrained_ucon_inner(
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
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        uncon_auglag_method(unconstrained_method),
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
//{{{ test: test_quadratic_bound_constrained_bcon_inner
#[rstest]
#[case::quadratic_asa_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Asa(ASA_NOCEDAL_PR),  1e-6, 1e-4,  9, 9)]
#[case::quadratic_bfsgb_nocedal(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Bfgsb(BFGSB_NOCEDAL),  1e-6, 1e-4,  14, 17)]
fn test_quadratic_bound_constrained_bcon_inner(
    #[case] x0: Vector,
    #[case] bound_constrained_method: BoundConstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quad = Quadratic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };
    let mut bound_constraints = BoundsConstraints::new(5);
    bound_constraints.add_bounds(0, Some(20.0), None);

    let ret = constrained_minimize(
        quad,
        Some(bound_constraints),
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        bcon_auglag_method(bound_constrained_method),
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
//{{{ test: test_quadratic_hsphere_constrained_ucon_inner
#[rstest]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-6, 1e-4,  3042, 4272)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-6, 1e-4,  1286, 701)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-6, 1e-4,  4135, 5798)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  2e-6, 1e-4,  5690, 2893)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-6, 1e-4,  164, 240)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-6, 1e-4,  7978, 4006)]
fn test_quadratic_hsphere_constrained_ucon_inner(
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

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[20.0, 20.0, 20.0, 20.0, 20.0]),
        radius: 10.0,
    };

    let ret = constrained_minimize(
        quad,
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        uncon_auglag_method(unconstrained_method),
    )
    .unwrap();

    println!("\n\nret = {ret:?}\n\n");
    let exp_fmin = 152.78640450004207;
    assert_answer(
        &ret,
        &colvec(&[
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
        ]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: test_quadratic_bound_constrained_quadratic_bound_constrained_circle_bound
#[rstest]
#[case::quadratic_asa_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Asa(ASA_NOCEDAL_PR),  1e-6, 1e-4,  303, 219)]
#[case::quadratic_bfsgb_nocedal(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Bfgsb(BFGSB_NOCEDAL),  1e-6, 1e-4,  9904, 5368)]
fn test_quadratic_hsphere_and_bound_constrained_bcon_inner(
    #[case] x0: Vector,
    #[case] bound_constrained_method: BoundConstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quad = Quadratic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let mut bound_constraints = BoundsConstraints::new(5);
    for i in 0..x0.len()
    {
        bound_constraints.add_bounds(i, Some(15.0), Some(20.0));
    }

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[20.0, 20.0, 20.0, 20.0, 20.0]),
        radius: 10.0,
    };

    let ret = constrained_minimize(
        quad,
        Some(bound_constraints),
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        bcon_auglag_method(bound_constrained_method),
    )
    .unwrap();

    println!("\n\nret = {ret:?}\n\n");
    let exp_fmin = 152.78640450004207;
    assert_answer(
        &ret,
        &colvec(&[
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
        ]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: test_quadratic_tight_grad_atol_below_old_hardcoded_floor_converges
// Regression test for the AL inner-tolerance schedule fix: `set_inner_tolerances`
// used to clamp the inner atol to a hardcoded [1e-6, 1e-2], so a user-requested
// `grad_atol` tighter than 1e-6 could never actually be delivered no matter how
// long the outer loop ran. This asks for 1e-8 (tighter than that old floor) on a
// well-conditioned single-bound quadratic and checks the solve both converges and
// actually reaches that tighter accuracy, rather than silently stalling at 1e-6.
#[test]
fn test_quadratic_tight_grad_atol_below_old_hardcoded_floor_converges()
{
    let quad = Quadratic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };
    let mut ieq_constraints = BoundsConstraints::new(5);
    ieq_constraints.add_bounds(0, Some(20.0), None);

    let opts = ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            base_opts: UnonstrainedOptions {
                grad_rtol: 0.0,
                grad_atol: 1e-8,
                max_iter: 1000,
                make_counting: true,
            },
            constraint_tol: 1e-8,
        },
        AugmentedLagrangianInnerMethod::Unconstrained(UnconstrainedMethod::QuasiNewton(
            THUENTE_BFGS,
        )),
    ));

    let ret = constrained_minimize(
        quad,
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
        opts,
    )
    .expect(
        "AL should reach a grad_atol tighter than the old hardcoded 1e-6 inner-tolerance floor",
    );

    println!("ret = {ret:?}");
    assert_answer(
        &ret,
        &colvec(&[20.0, 10.0, 10.0, 10.0, 10.0]),
        100.0,
        1e-6,
        1e-6,
    );
}
//}}}
//}}}
//{{{ collection: quartic
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
//{{{ test: unconstrained
#[rstest]
#[case::quartic_thuente_bfgs(UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 40, 67)]
#[case::quartic_nocedal_bfgs(UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 55, 71)]
#[case::quartic_thuente_fr(UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 255, 321)]
#[case::quartic_nocedal_fr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 137, 154)]
#[case::quartic_thuente_pr(UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 304, 375)]
#[case::quartic_nocedal_pr(UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 158, 175)]
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
        None,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0_in,
        uncon_auglag_method(unconstrained_method),
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
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-2, 1e-2, 7478, 8470)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-2, 1e-2, 3950, 2378)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-2, 1e-2, 82704, 90724)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  1e-2, 1e-2, 6622, 2340)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-2, 1e-2, 6780, 7826)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-2, 1e-2, 6137, 2159)]
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

    let mut method = uncon_auglag_method(unconstrained_method);
    method.con_opts_mut().constraint_tol = 1e-3;
    method.con_opts_mut().base_opts.grad_rtol = 1e-4;

    let mut ieq_constraints = BoundsConstraints::new(5);
    ieq_constraints.add_bounds(0, Some(20.0), None);

    let ret = constrained_minimize(
        quart,
        None,
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
//{{{ test: bound constrained with bound-constrained inner
#[rstest]
#[case::quartic_asa_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Asa(ASA_NOCEDAL_PR),  1e-2, 1e-2,  500, 450)]
#[case::quartic_bfsgb_nocedal(colvec(&[20.0, 11.0, 11.0, 11.0, 11.0]), BoundConstrainedMethod::Bfgsb(BFGSB_NOCEDAL),  1e-2, 1e-2,  20, 20)]
fn test_quartic_with_bound_constraints_and_bcon_inner_matches_reference(
    #[case] x0: Vector,
    #[case] bound_constrained_method: BoundConstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let mut bound_constraints = BoundsConstraints::new(5);
    bound_constraints.add_bounds(0, Some(20.0), None);

    let mut method = bcon_auglag_method(bound_constrained_method);
    method.con_opts_mut().constraint_tol = 1e-3;
    method.con_opts_mut().base_opts.grad_rtol = 1e-4;

    let ret = constrained_minimize(
        quart,
        Some(bound_constraints),
        None::<NoConstraints>,
        None::<NoConstraints>,
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
//{{{ test: hsphere constrained
#[rstest]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  1e-2, 1e-2, 125, 200)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  1e-2, 1e-2, 200, 205)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),  1e-2, 1e-2, 250, 315)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),  1e-2, 1e-2, 475, 200)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),  1e-2, 1e-2, 135, 190)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),  1e-2, 1e-2, 9000, 2500)]
fn test_quartic_hsphere_constrained_ucon_inner_matches_reference(
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

    let mut method = uncon_auglag_method(unconstrained_method);
    method.con_opts_mut().constraint_tol = 1e-3;
    method.con_opts_mut().base_opts.grad_rtol = 1e-4;

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[20.0, 20.0, 20.0, 20.0, 20.0]),
        radius: 10.0,
    };

    let ret = constrained_minimize(
        quart,
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        method,
    )
    .unwrap();

    println!("\n\nret = {ret:?}\n\n");
    let exp_fmin = 4668.737080010094;
    assert_answer(
        &ret,
        &colvec(&[
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
        ]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: hsphere and bound constrained with bound-constrained inner
#[rstest]
#[case::quartic_asa_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Asa(ASA_NOCEDAL_PR),  1e-2, 1e-2,  725, 260)]
#[case::quartic_bfsgb_nocedal(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), BoundConstrainedMethod::Bfgsb(BFGSB_NOCEDAL),  1e-2, 1e-2,  300, 185)]
fn test_quartic_hsphere_and_bound_constrained_bcon_inner_matches_reference(
    #[case] x0: Vector,
    #[case] bound_constrained_method: BoundConstrainedMethod,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: f64,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let mut bound_constraints = BoundsConstraints::new(5);
    for i in 0..x0.len()
    {
        bound_constraints.add_bounds(i, Some(15.0), Some(20.0));
    }

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[20.0, 20.0, 20.0, 20.0, 20.0]),
        radius: 10.0,
    };

    let ret = constrained_minimize(
        quart,
        Some(bound_constraints),
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        bcon_auglag_method(bound_constrained_method),
    )
    .unwrap();

    println!("\n\nret = {ret:?}\n\n");
    let exp_fmin = 4668.737080010094;
    assert_answer(
        &ret,
        &colvec(&[
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
            15.52786404500042,
        ]),
        exp_fmin,
        xmin_tol,
        fmin_tol,
    );
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
//{{{ collection: rosenbrock
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
//{{{ test: test_rosenbrock_uncon
#[rstest]
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  52, 84)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  74, 83)]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),   215, 337)]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),   235, 220)]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),   319, 473)]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),   199, 165)]
fn test_rosenbrock_uncon(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let rosenbrock = Rosenbrock::new();
    let ret = constrained_minimize(
        rosenbrock,
        None,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        uncon_auglag_method(unconstrained_method),
    )
    .unwrap();
    println!("ret = {ret:?}");
    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, 1e-3, 1e-5);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: test_rosenbrock_bcon_1
#[rstest]
#[case::rosenbrock_thuente_bfgs(colvec(&[-1.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  113, 183)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[5.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  114, 122)]
#[case::rosenbrock_thuente_fr(colvec(&[-2.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),   342, 532)]
#[case::rosenbrock_nocedal_fr(colvec(&[-3.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),   245, 197)]
#[case::rosenbrock_thuente_pr(colvec(&[-3.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),   319, 473)]
#[case::rosenbrock_nocedal_pr(colvec(&[-10.0, -10.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),   199, 165)]
fn test_rosenbrock_bcon_1(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let rosenbrock = Rosenbrock::new();

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[2.0, 2.0]),
        radius: 3.0,
    };

    let ret = constrained_minimize(
        rosenbrock,
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        uncon_auglag_method(unconstrained_method),
    )
    .unwrap();
    println!("ret = {ret:?}");
    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, 1e-3, 1e-5);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//{{{ test: test_rosenbrock_bcon_2
#[rstest]
#[case::rosenbrock_thuente_bfgs(colvec(&[-1.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS),  250, 416)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[5.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS),  348, 383)]
#[case::rosenbrock_thuente_fr(colvec(&[-2.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR),   864, 1430)]
#[case::rosenbrock_nocedal_fr(colvec(&[-3.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR),   878, 828)]
#[case::rosenbrock_thuente_pr(colvec(&[-3.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR),   1386, 2244)]
#[case::rosenbrock_nocedal_pr(colvec(&[-10.0, -10.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR),   935, 772)]
fn test_rosenbrock_bcon_2(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
    #[case] exp_num_fun_evals: usize,
    #[case] exp_num_grad_evals: usize,
)
{
    let rosenbrock = Rosenbrock::new();

    let ieq_constraints = HyperSphereBound {
        center: colvec(&[0.0, 0.0]),
        radius: 0.5,
    };

    let ret = constrained_minimize(
        rosenbrock,
        None,
        None::<NoConstraints>,
        Some(ieq_constraints),
        x0,
        uncon_auglag_method(unconstrained_method),
    )
    .unwrap();
    println!("ret = {ret:?}");
    assert_answer(&ret, &colvec(&[0.455649, 0.205874]), 0.296621, 1e-3, 1e-5);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
