#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::constrained::{
    minimize as constrained_minimize, AugmentedLagrangianOptions, ConstrainedMethod,
    ConstrainedReturns, ConstriainedOptions,
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
//{{{ struct: NoConstraints
#[derive(Debug, Clone, Copy)]
struct NoConstraints;
//}}}
//{{{ impl: RealVectorFn for NoConstraints
impl RealVectorFn for NoConstraints
{
    fn dimension_domain(&self) -> usize
    {
        0
    }

    fn dimension_range(&self) -> usize
    {
        0
    }

    fn eval(
        &mut self,
        _x: &Vector,
        _val: &mut Vector,
    )
    {
    }

    fn grad(
        &mut self,
        _x: &Vector,
        _val: &mut Matrix,
    )
    {
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
    assert!((ret.xmin.clone() - exp_xmin.clone()).norm() < xmin_tol);
    assert!((ret.fmin - exp_fmin).abs() < fmin_tol);
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
            make_counting: false,
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
#[case::quadratic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS))]
#[case::quadratic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS))]
#[case::quadratic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS))]
#[case::quadratic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST))]
#[case::quadratic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST))]
#[case::quadratic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST))]
#[case::quadratic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR))]
#[case::quadratic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR))]
#[case::quadratic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR))]
#[case::quadratic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR))]
#[case::quadratic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR))]
#[case::quadratic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR))]
fn test_quadratic_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] unconstrained_method: UnconstrainedMethod,
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
        1e-7,
        1e-10,
    );
}
//}}}
//{{{ test: quartic
#[rstest]
#[case::quartic_interp_bfgs(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS))]
#[case::quartic_thuente_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS))]
#[case::quartic_nocedal_bfgs(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS))]
#[case::quartic_interp_steepest(colvec(&[0.0, 0.0, 0.0, 0.0, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_STEEPEST))]
#[case::quartic_thuente_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_STEEPEST))]
#[case::quartic_nocedal_steepest(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_STEEPEST))]
#[case::quartic_interp_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR))]
#[case::quartic_thuente_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR))]
#[case::quartic_nocedal_fr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR))]
#[case::quartic_interp_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR))]
#[case::quartic_thuente_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR))]
#[case::quartic_nocedal_pr(colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR))]
fn test_quartic_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] mut unconstrained_method: UnconstrainedMethod,
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
        5e-2,
        1e-5,
    );
}
//}}}
//{{{ test: rosenbrock
#[rstest]
#[case::rosenbrock_interp_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(INTERP_BFGS))]
#[case::rosenbrock_thuente_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(THUENTE_BFGS))]
#[case::rosenbrock_nocedal_bfgs(colvec(&[0.0, 3.0]), UnconstrainedMethod::QuasiNewton(NOCEDAL_BFGS))]
#[case::rosenbrock_interp_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_FR))]
#[case::rosenbrock_thuente_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_FR))]
#[case::rosenbrock_nocedal_fr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_FR))]
#[case::rosenbrock_interp_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(INTERP_PR))]
#[case::rosenbrock_thuente_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(THUENTE_PR))]
#[case::rosenbrock_nocedal_pr(colvec(&[0.0, 3.0]), UnconstrainedMethod::ConjugateGradient(NOCEDAL_PR))]
fn test_rosenbrock_without_constraints_matches_unconstrained_reference(
    #[case] x0: Vector,
    #[case] mut unconstrained_method: UnconstrainedMethod,
)
{
    let rosenbrock = Rosenbrock::new();

    unconstrained_method.uncon_opts_mut().grad_rtol = 1e-6;
    unconstrained_method.uncon_opts_mut().grad_atol = 1e-10;
    unconstrained_method.uncon_opts_mut().max_iter = 10000;
    if let LineSearchMethod::Interp(interp_opts) = &mut unconstrained_method.uncon_opts_mut().ls_method
    {
        interp_opts.scale_factor = 1.2;
    }

    let ret = minimize_without_constraints(rosenbrock, x0, unconstrained_method);

    assert_answer(&ret, &colvec(&[1.0, 1.0]), 0.0, 1e-2, 1e-6);
}
//}}}
