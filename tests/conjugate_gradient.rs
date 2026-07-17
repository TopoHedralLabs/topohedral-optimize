//{{{ crate imports
use topohedral_optimize::{
    unconstrained_minimize as minimize, ConjugateGradientDirection as Direction,
    ConjugateGradientOptions, ConvergedReason as UnconstrainedConvergedReason, LineSearchMethod,
    LineSearchOptions, NocedalOptions, ThuenteOptions, UnconstrainedMethod,
    UnconstrainedOptions as UnonstrainedOptions, Vector, VectorReturns as UnconstrainedReturns,
};
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
fn init_logger() {
    init().unwrap();
}
//}}}
//{{{ fun: colvec
fn colvec(values: &[f64]) -> Vector {
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}
//}}}
//{{{ struct: Quadratic
#[derive(Debug)]
struct Quadratic {
    xmin: Vector,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl topohedral_optimize::DifferentiableFn for Quadratic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        self.xmin.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let tmp = x.clone() - self.xmin.clone();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(2);
        }
        out
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let tmp = x_in.clone() - self.xmin.clone();
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
        for i in 0..5 {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}
//{{{ struct: Quartic
#[derive(Debug)]
struct Quartic {
    xmin: Vector,
}
//}}}
//{{{ impl: RealFn for Quartic
impl topohedral_optimize::DifferentiableFn for Quartic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        self.xmin.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let tmp = x.clone() - self.xmin.clone();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn derivative(
        &mut self,
        x_in: &Vector,
    ) -> Vector {
        let tmp = x_in.clone() - self.xmin.clone();
        let mut out = DVector::<f64>::zeros_vec(5, VecType::Col);
        for i in 0..5 {
            out[i] = 4.0 * tmp[i].powi(3);
        }
        out
    }
}
//}}}
//{{{ struct Rosenbrock
#[derive(Debug, Clone, Copy)]
struct Rosenbrock {
    a: f64,
    b: f64,
}
//}}}
//{{{ impl: Rosenbrock
impl Rosenbrock {
    fn new() -> Self {
        Self { a: 1.0, b: 100.0 }
    }
}
//}}}
//{{{ impl: RealFn for Rosenbrock
impl topohedral_optimize::DifferentiableFn for Rosenbrock {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        2
    }

    fn eval(
        &mut self,
        xvec: &Vector,
    ) -> f64 {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn derivative(
        &mut self,
        xvec: &Vector,
    ) -> Vector {
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
) {
    assert!((ret.xmin.clone() - exp_ret.xmin.clone()).norm() < xmin_tol);
    assert!((ret.fmin - exp_ret.fmin).abs() < fmin_tol);
    assert_eq!(ret.reason, exp_ret.reason);
    assert!(
        ret.num_iterations <= exp_ret.num_iterations,
        "expected at most {} iterations, got {}",
        exp_ret.num_iterations,
        ret.num_iterations
    );
    assert_eq!(ret.num_fun_evals, exp_ret.num_fun_evals);
    assert_eq!(ret.num_grad_evals, exp_ret.num_grad_evals);
}
//}}}
//{{{ const: THUENTE_STEEPEST
const THUENTE_STEEPEST: ConjugateGradientOptions = ConjugateGradientOptions {
    uncon_opts: UnonstrainedOptions {
        grad_rtol: 1e-6,
        grad_atol: 1e-8,
        max_iter: 100,
    },
    ls_method: LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.4,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
    }),
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
    },
    ls_method: LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.4,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
    }),
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
    },
    ls_method: LineSearchMethod::Thuente(ThuenteOptions {
        ls_opts: LineSearchOptions {
            c1: 1.0e-4,
            c2: 0.4,
            step_min: 1e-8,
            step_max: 1e5,
        },
        maxiter: 10,
    }),
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
    },
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
    },
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
    },
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
    direction: Direction::PolakRibiere,
    restart: 10,
};
//}}}
//{{{ test: quadratic
#[rstest]
//{{{ case: test_quadratic_thuente_steepest
#[case::test_quadratic_thuente_steepest(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_STEEPEST,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1,
        num_fun_evals: 7,
        num_grad_evals: 9
    }
)]
//}}}
//{{{ case: test_quadratic_thuente_fr
#[case::test_quadratic_thuente_fr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1,
        num_fun_evals: 7,
        num_grad_evals: 9,
    }
)]
//}}}
//{{{ case: test_quadratic_thuente_pr
#[case::test_quadratic_thuente_pr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1,
        num_fun_evals: 7,
        num_grad_evals: 9,
    }
)]
//}}}
//{{{ case: test_quadratic_nocedal_steepest
#[case::test_quadratic_nocedal_steepest(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_STEEPEST,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 2,
        num_fun_evals: 14,
        num_grad_evals: 15
    }
)]
//}}}
//{{{ case: test_quadratic_nocedal_fr
#[case::test_quadratic_nocedal_fr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 2,
        num_fun_evals: 14,
        num_grad_evals: 15,
    }
)]
//}}}
//{{{ case: test_quadratic_nocedal_pr
#[case::test_quadratic_nocedal_pr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 2,
        num_fun_evals: 14,
        num_grad_evals: 15,
    }
)]
//}}}
fn test_qudratic(
    #[case] x0: Vector,
    #[case] opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns,
) {
    let mut quad = Quadratic {
        xmin: colvec(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };

    let ret = minimize(&mut quad, x0, UnconstrainedMethod::ConjugateGradient(opts)).unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-7, 1e-10);
}
//}}}
//{{{ test: quartic
#[rstest]
//{{{ case: test_quartic_thuente_steepest
#[case::test_quartic_thuente_steepest(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_STEEPEST,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 12,
        num_fun_evals: 34,
        num_grad_evals: 46
    }
)]
//}}}
//{{{ case: test_quartic_thuente_fr
#[case::test_quartic_thuente_fr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 10,
        num_fun_evals: 34,
        num_grad_evals: 45,
    }
)]
//}}}
//{{{ case: test_quartic_thuente_pr
#[case::test_quartic_thuente_pr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 12,
        num_fun_evals: 34,
        num_grad_evals: 46,
    }
)]
//}}}
//{{{ case: test_quartic_nocedal_steepest
#[case::test_quartic_nocedal_steepest(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_STEEPEST,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 19,
        num_fun_evals: 88,
        num_grad_evals: 93
    }
)]
//}}}
//{{{ case: test_quartic_nocedal_fr
#[case::test_quartic_nocedal_fr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 15,
        num_fun_evals: 71,
        num_grad_evals: 70,
    }
)]
//}}}
//{{{ case: test_quartic_nocedal_pr
#[case::test_quartic_nocedal_pr(
    colvec(&[100.0, -100.0, 3.0, 1e-6, 0.0]),
    NOCEDAL_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 20,
        num_fun_evals: 96,
        num_grad_evals: 100,
    }
)]
//}}}
fn test_quartic(
    #[case] x0: Vector,
    #[case] mut opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns,
) {
    let mut quart = Quartic {
        xmin: colvec(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };
    opts.uncon_opts.grad_rtol = 1e-12;
    opts.uncon_opts.grad_atol = 1e-12;

    let ret = minimize(&mut quart, x0, UnconstrainedMethod::ConjugateGradient(opts)).unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 5e-2, 1e-5);
}
//}}}
//{{{ test: rosenbrock
#[rstest]
//{{{ case: test_rosenbrock_thuente_fr
#[case::test_rosenbrock_thuente_fr(
    colvec(&[0.0, 3.0]),
    THUENTE_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 31,
        num_fun_evals: 73,
        num_grad_evals: 105,
    }
)]
//}}}
//{{{ case: test_rosenbrock_thuente_pr
#[case::test_rosenbrock_thuente_pr(
    colvec(&[0.0, 3.0]),
    THUENTE_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 19,
        num_fun_evals: 40,
        num_grad_evals: 60,
    }
)]
//}}}
//{{{ case: test_rosenbrock_nocedal_fr
#[case::test_rosenbrock_nocedal_fr(
    colvec(&[0.0, 3.0]),
    NOCEDAL_FR,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 30,
        num_fun_evals: 93,
        num_grad_evals: 89,
    }
)]
//}}}
//{{{ case: test_rosenbrock_nocedal_pr
#[case::test_rosenbrock_nocedal_pr(
    colvec(&[0.0, 3.0]),
    NOCEDAL_PR,
    UnconstrainedReturns{
        xmin:  colvec(&[1.0, 1.0]),
        fmin: 0.0,
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 24,
        num_fun_evals: 92,
        num_grad_evals: 84,
    }
)]
//}}}
fn test_rosenbrock(
    #[case] x0: Vector,
    #[case] mut opts: ConjugateGradientOptions,
    #[case] exp_ret: UnconstrainedReturns,
) {
    let mut rosenbrock = Rosenbrock::new();

    opts.uncon_opts.grad_rtol = 1e-6;
    opts.uncon_opts.grad_atol = 1e-10;
    opts.uncon_opts.max_iter = 10000;
    let ret = minimize(
        &mut rosenbrock,
        x0,
        UnconstrainedMethod::ConjugateGradient(opts),
    )
    .unwrap();
    println!("{ret:?}");
    assert_returns(&ret, &exp_ret, 1e-2, 1e-6);
}
//}}}
