#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::constrained::{
    minimize as constrained_minimize, AugmentedLagrangianOptions, ConstrainedMethod,
    ConstrainedReturns, ConstriainedOptions, NoConstraints,
};
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    ConjugateGradientOptions, Direction, QuasiNewtonOptions, UnconstrainedMethod,
    UnonstrainedOptions, UpdateMethod,
};
use topohedral_optimize::{Matrix, RealFn, RealVectorFn, Vector};
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
//{{{ fun: assert_answer
fn assert_answer(
    ret: &ConstrainedReturns,
    exp_xmin: &Vector,
    exp_fmin: f64,
    xmin_tol: f64,
    fmin_tol: f64,
)
{
    let xmin_err = (ret.xmin.clone() - exp_xmin.clone()).norm();
    let fmin_err = (ret.fmin - exp_fmin).abs();
    assert!(xmin_err <= xmin_tol);
    assert!(fmin_err <= fmin_tol);
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
    let grad_rtol = unconstrained_method.uncon_opts().grad_rtol;
    let grad_atol = unconstrained_method.uncon_opts().grad_atol;
    unconstrained_method.uncon_opts_mut().make_counting = false;

    ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            grad_rtol,
            grad_atol,
            constraint_tol: 1e-10,
            max_iter: 100,
            make_counting: true,
        },
        unconstrained_method,
        1.0,
        4.0,
        10.0,
    ))
}
//}}}
//{{{ fun: minimize_without_constraints
fn minimize_without_constraints<F: RealFn>(
    fcn: F,
    x0: Vector,
    unconstrained_method: UnconstrainedMethod,
) -> ConstrainedReturns
{
    constrained_minimize(
        fcn,
        None::<NoConstraints>,
        None::<NoConstraints>,
        x0,
        auglag_method(unconstrained_method),
    )
    .unwrap()
}
//}}}
//{{{ const: INTERP_BFGS
const INTERP_BFGS: QuasiNewtonOptions = QuasiNewtonOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
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
        make_counting: true,
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
        make_counting: true,
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
//{{{ const: INTERP_STEEPEST
const INTERP_STEEPEST: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
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
    restart: 100,
};
//}}}
//{{{ const: INTERP_FR
const INTERP_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
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
        make_counting: true,
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
        make_counting: true,
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
        make_counting: true,
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
        make_counting: true,
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
//{{{ const: NOCEDAL_STEEPEST
const NOCEDAL_STEEPEST: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
            zoom_maxiter: 10,
        }),
    },
    direction: Direction::Steepest,
    restart: 10,
};
//}}}
//{{{ const: NOCEDAL_FR
const NOCEDAL_FR: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
        make_counting: true,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
            zoom_maxiter: 10,
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
        make_counting: true,
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.4,
                step_min: 1e-8,
                step_max: 1e5,
            },
            maxiter: 10,
            zoom_maxiter: 10,
        }),
    },
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}

//{{{ test: quadratic
#[rstest]
#[case::quadratic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 2.0183860421701553e-9, 4.0738822152273034e-18, 7, 6)]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 3.2163343058062638e-13, 1.0344806366706260e-25, 8, 12)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 1.4273168140722167e-14, 2.0372332877332630e-28, 12, 17)]
#[case::quadratic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST), 2.0183860421701553e-9, 4.0738822152273034e-18, 7, 6)]
#[case::quadratic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST), 2.5423787284960917e-13, 6.4636895991094044e-26, 8, 11)]
#[case::quadratic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST), 0.0, 0.0, 15, 17)]
#[case::quadratic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 1.9739743194793782e-8, 3.8965746139640743e-16, 7, 6)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 2.5423787284960917e-13, 6.4636895991094044e-26, 8, 11)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 1.9428902930940239e-16, 3.7748226909989823e-32, 15, 17)]
#[case::quadratic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 1.9739743194793782e-8, 3.8965746139640743e-16, 7, 6)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 2.5423787284960917e-13, 6.4636895991094044e-26, 8, 11)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 0.0, 0.0, 15, 17)]
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

    let ret = minimize_without_constraints(quad, x0, unconstrained_method);

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
//{{{ test: quartic
#[rstest]
#[case::quartic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 8.0584010354030333e-4, 8.4442936244381566e-14, 84, 50)]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 2.8900209777231074e-3, 6.4556561582294109e-11, 75, 136)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 6.5216746992386868e-3, 4.7944343000457093e-10, 93, 146)]
#[case::quartic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST), 7.3257814731992124e-4, 5.7603097048887079e-14, 72, 42)]
#[case::quartic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST), 5.4074353342156616e-3, 1.7251059227950362e-10, 49, 68)]
#[case::quartic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST), 8.6607940280398355e-3, 1.1252810254593206e-9, 140, 151)]
#[case::quartic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 7.3901008576251178e-3, 1.2250544373935824e-9, 182, 69)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 4.9945855070538609e-3, 2.7237752486994082e-10, 45, 63)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 5.1622986074759368e-3, 3.3766269996150770e-10, 85, 96)]
#[case::quartic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 9.1696602167280188e-3, 1.4194193247262839e-9, 158, 58)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 5.4149501498073202e-3, 1.7362324129435643e-10, 49, 68)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 4.3538638636983207e-3, 7.1866977053238727e-11, 393, 405)]
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

    unconstrained_method.uncon_opts_mut().grad_rtol = 1e-12;
    unconstrained_method.uncon_opts_mut().grad_atol = 1e-12;
    unconstrained_method.uncon_opts_mut().max_iter = 1000;

    let ret = minimize_without_constraints(quart, x0, unconstrained_method);

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
//{{{ test: rosenbrock
#[rstest]
#[case::rosenbrock_interp_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 5.5239410319076359e-8, 6.0930421678103236e-16, 233, 74)]
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 3.0301067050363771e-8, 1.8439337511737664e-16, 39, 68)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 4.6422783067275514e-9, 1.3042327727944209e-16, 60, 72)]
#[case::rosenbrock_interp_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 1.4203156931653554e-5, 4.0283446150389152e-11, 23137, 9250)]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 9.3919627763419816e-6, 1.7614089622164401e-11, 485, 745)]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 1.2058178005261424e-5, 2.9033584551019353e-11, 768, 923)]
#[case::rosenbrock_interp_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 1.3997451454280088e-3, 3.9103581632830066e-7, 162, 52)]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 1.2849890885491134e-8, 1.7765449516843973e-15, 49, 73)]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 1.6327198456942709e-6, 5.3332486404511308e-13, 149, 129)]
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

    let ret = minimize_without_constraints(rosenbrock, x0, unconstrained_method);

    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, xmin_tol, fmin_tol);
    assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
