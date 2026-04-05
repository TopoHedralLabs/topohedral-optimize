#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

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
    unconstrained_method.uncon_opts_mut().make_counting = false;

    ConstrainedMethod::AugmentedLagrangian(AugmentedLagrangianOptions::new(
        ConstriainedOptions {
            grad_rtol: 1e-6,
            grad_atol: 1e-8,
            constraint_tol: 1e-8,
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
//{{{ test: unconstrained
#[rstest]
#[case::quadratic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 2.0183860421701553e-9, 4.0738822152273034e-18, 8, 7)]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 3.2163343058062638e-13, 1.0344806366706260e-25, 9, 13)]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 1.4273168140722167e-14, 2.0372332877332630e-28, 13, 18)]
#[case::quadratic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST), 2.0183860421701553e-9, 4.0738822152273034e-18, 8, 7)]
#[case::quadratic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST), 2.5423787284960917e-13, 6.4636895991094044e-26, 9, 12)]
#[case::quadratic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST), 0.0, 0.0, 16, 18)]
#[case::quadratic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 1.9739743194793782e-8, 3.8965746139640743e-16, 8, 7)]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 2.5423787284960917e-13, 6.4636895991094044e-26, 9, 12)]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 1.9428902930940239e-16, 3.7748226909989823e-32, 16, 18)]
#[case::quadratic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 1.9739743194793782e-8, 3.8965746139640743e-16, 8, 7)]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 2.5423787284960917e-13, 6.4636895991094044e-26, 9, 12)]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 0.0, 0.0, 16, 18)]
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
#[case::quadratic_interp_bfgs(colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 1e-2, 1e-2, 1, 1)]
fn test_quadratic_with_bound_constraints(
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
    // assert_answer(
    //     &ret,
    //     &colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    //     0.0,
    //     xmin_tol,
    //     fmin_tol,
    // );
    // assert_counts(&ret, exp_num_fun_evals, exp_num_grad_evals);
}
//}}}
//}}}
//{{{ test: quartic
#[rstest]
#[case::quartic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 1.1906159454267216e-1, 4.0189885991360184e-5, 45, 31)]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 6.8e-1, 1.2171652215773493e-1, 49, 87)]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 7.3232511374922560e-1, 6.9775029981934830e-2, 74, 103)]
#[case::quartic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST), 1.4479820748386635e-1, 8.7918999650696810e-5, 25, 18)]
#[case::quartic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST), 2.6054392428170864e-1, 2.2916123661796467e-3, 21, 34)]
#[case::quartic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST), 6.3973449456097953e-1, 4.5171976960111730e-2, 28, 34)]
#[case::quartic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 1.2070577926956085e0, 5.4374493955913960e-1, 47, 29)]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 4.3181595955113955e-1, 8.7598881523012430e-3, 23, 37)]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 4.1150527192591912e-1, 1.0639361825963145e-2, 33, 40)]
#[case::quartic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 8.2596675128718233e-1, 2.2340415680397563e-1, 72, 32)]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 4.4101768233767547e-1, 1.3287153804932000e-2, 21, 34)]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 6.5889078801448453e-1, 7.9115661470400640e-2, 41, 48)]
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
//{{{ test: rosenbrock
#[rstest]
#[case::rosenbrock_interp_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS), 5.5239410319076359e-8, 4.2600533772335230e-14, 124, 54)]
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS), 9.0284228790531235e-7, 3.3156139251068023e-13, 34, 59)]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS), 1.2741169516945091e-7, 8.0476763718045500e-15, 73, 68)]
#[case::rosenbrock_interp_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR), 4.7020691820927910e-4, 4.4157413399944545e-8, 13020, 5212)]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR), 2.3939387271972799e-4, 1.1444703681763757e-8, 404, 641)]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR), 4.1115183059162496e-5, 3.3755631595392825e-10, 497, 614)]
#[case::rosenbrock_interp_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR), 1.3997451454280088e-3, 3.9103581632830066e-7, 125, 54)]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR), 5.4859043189398348e-6, 6.5279974297398110e-12, 61, 91)]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR), 2.8080039179491367e-4, 1.5745787517370056e-8, 67, 75)]
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
