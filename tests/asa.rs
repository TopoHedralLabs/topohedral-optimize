//{{{ crate imports
use topohedral_optimize::{
    bound_constrained_minimize, AsaOptions, BaseOptions, BoundConstrainedMethod,
    BoundConstrainedOptions, BoundsConstraints, LineSearchMethod, LineSearchOptions,
    QuasiNewtonOptions, QuasiNewtonUpdateMethod as UpdateMethod, RealFn, ThuenteOptions,
    UnconstrainedMethod, UnconstrainedOptions as UnonstrainedOptions, Vector, VectorReturns,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use approx::assert_relative_eq;
use ctor::ctor;
use topohedral_linalg::{DVector, ReduceOps, VecType, VectorOps};
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
//{{{ fun: assert_vector_close
fn assert_vector_close(
    actual: &Vector,
    expected: &Vector,
    epsilon: f64,
) {
    assert_eq!(actual.len(), expected.len());
    for (actual_i, expected_i) in actual.iter().zip(expected.iter()) {
        assert_relative_eq!(*actual_i, *expected_i, epsilon = epsilon);
    }
}
//}}}
//{{{ fun: add_uniform_bounds
fn add_uniform_bounds(
    n: usize,
    lower: Option<f64>,
    upper: Option<f64>,
) -> BoundsConstraints {
    let mut bounds = BoundsConstraints::new(n);
    for i in 0..n {
        bounds.add_bounds(i, lower, upper);
    }
    bounds
}
//}}}
//{{{ fun: kkt_residual
fn kkt_residual<F: RealFn>(
    mut fcn: F,
    x: &Vector,
    lower: &[Option<f64>],
    upper: &[Option<f64>],
) -> f64 {
    let grad = fcn.grad(x);
    let mut projected = x.clone() - grad;

    for i in 0..projected.len() {
        if let Some(lower_i) = lower[i] {
            projected[i] = projected[i].max(lower_i);
        }
        if let Some(upper_i) = upper[i] {
            projected[i] = projected[i].min(upper_i);
        }
    }

    (projected - x.clone()).abs_max().unwrap_or(0.0)
}
//}}}
//{{{ fun: asa_options
fn asa_options(max_iter: u64) -> AsaOptions {
    AsaOptions::new(
        BoundConstrainedOptions {
            base_opts: BaseOptions {
                grad_rtol: 1e-8,
                grad_atol: 1e-8,
                max_iter,
                make_counting: true,
            },
            constraint_tol: 1e-8,
        },
        UnconstrainedMethod::QuasiNewton(QuasiNewtonOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-8,
                grad_atol: 1e-8,
                max_iter: 100,
                make_counting: false,
            },
            ls_method: LineSearchMethod::Thuente(ThuenteOptions {
                ls_opts: LineSearchOptions {
                    c1: 1.0e-4,
                    c2: 0.9,
                    step_min: 1e-12,
                    step_max: 1e5,
                },
                maxiter: 50,
            }),
            method: UpdateMethod::BFGS,
            restart: 10,
        }),
    )
}
//}}}
//{{{ fun: solve_asa
fn solve_asa<F: RealFn>(
    fcn: F,
    x0: Vector,
    bounds: BoundsConstraints,
    max_iter: u64,
) -> VectorReturns {
    bound_constrained_minimize(
        fcn,
        bounds,
        x0,
        BoundConstrainedMethod::Asa(asa_options(max_iter)),
    )
    .expect("ASA minimization should succeed")
}
//}}}

//{{{ struct: ShiftedQuadratic
#[derive(Debug, Clone)]
struct ShiftedQuadratic {
    target: Vector,
}
//}}}
//{{{ impl: RealFn for ShiftedQuadratic
impl RealFn for ShiftedQuadratic {
    fn dimension(&self) -> usize {
        self.target.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let diff = x.clone() - self.target.clone();
        diff.dot(&diff)
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        2.0 * (x.clone() - self.target.clone())
    }
}
//}}}
//{{{ struct: Rosenbrock
#[derive(Debug, Clone)]
struct Rosenbrock {
    n: usize,
}
//}}}
//{{{ impl: RealFn for Rosenbrock
impl RealFn for Rosenbrock {
    fn dimension(&self) -> usize {
        self.n
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let mut value = 0.0;
        for i in 0..(self.n - 1) {
            value += 100.0 * (x[i + 1] - x[i].powi(2)).powi(2) + (1.0 - x[i]).powi(2);
        }
        value
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let mut grad = DVector::<f64>::zeros_vec(self.n, VecType::Col);
        for i in 0..(self.n - 1) {
            grad[i] += -400.0 * x[i] * (x[i + 1] - x[i].powi(2)) - 2.0 * (1.0 - x[i]);
            grad[i + 1] += 200.0 * (x[i + 1] - x[i].powi(2));
        }
        grad
    }
}
//}}}
//{{{ struct: DiagonalSpdQuadratic
#[derive(Debug, Clone)]
struct DiagonalSpdQuadratic {
    diagonal: Vector,
    rhs: Vector,
}
//}}}
//{{{ impl: DiagonalSpdQuadratic
impl DiagonalSpdQuadratic {
    fn from_unconstrained_minimum(unconstrained_minimum: Vector) -> Self {
        let mut diagonal = DVector::<f64>::zeros_vec(unconstrained_minimum.len(), VecType::Col);
        let mut rhs = DVector::<f64>::zeros_vec(unconstrained_minimum.len(), VecType::Col);

        for i in 0..unconstrained_minimum.len() {
            diagonal[i] = 1.0 + 0.1 * i as f64;
            rhs[i] = diagonal[i] * unconstrained_minimum[i];
        }

        Self { diagonal, rhs }
    }
}
//}}}
//{{{ impl: RealFn for DiagonalSpdQuadratic
impl RealFn for DiagonalSpdQuadratic {
    fn dimension(&self) -> usize {
        self.diagonal.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let mut value = 0.0;
        for i in 0..x.len() {
            value += 0.5 * self.diagonal[i] * x[i].powi(2) - self.rhs[i] * x[i];
        }
        value
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let mut grad = DVector::<f64>::zeros_vec(x.len(), VecType::Col);
        for i in 0..x.len() {
            grad[i] = self.diagonal[i] * x[i] - self.rhs[i];
        }
        grad
    }
}
//}}}

//{{{ test: simple quadratic with active upper bound
#[test]
fn asa_minimizes_shifted_quadratic_with_active_upper_bound() {
    let n = 5;
    let fcn = ShiftedQuadratic {
        target: colvec(&[0.0, 1.0, 2.0, 3.0, 4.0]),
    };
    let x0 = DVector::<f64>::from_value_vec(1.5, n, VecType::Col);
    let bounds = add_uniform_bounds(n, Some(0.0), Some(3.0));

    let ret = solve_asa(fcn.clone(), x0, bounds, 500);
    let expected_x = colvec(&[0.0, 1.0, 2.0, 3.0, 3.0]);

    assert_vector_close(&ret.xmin, &expected_x, 1e-5);
    assert_relative_eq!(ret.fmin, 1.0, epsilon = 1e-8);
    assert!(kkt_residual(fcn, &ret.xmin, &[Some(0.0); 5], &[Some(3.0); 5]) <= 1e-6);
}
//}}}
//{{{ test: ten dimensional rosenbrock
#[test]
fn asa_minimizes_ten_dimensional_rosenbrock_inside_box() {
    let n = 10;
    let fcn = Rosenbrock { n };
    let x0 = DVector::<f64>::from_value_vec(-1.2, n, VecType::Col);
    let bounds = add_uniform_bounds(n, Some(-2.0), Some(2.0));

    let ret = solve_asa(fcn.clone(), x0, bounds, 2000);
    let expected_x = DVector::<f64>::from_value_vec(1.0, n, VecType::Col);

    assert_vector_close(&ret.xmin, &expected_x, 1e-4);
    assert_relative_eq!(ret.fmin, 0.0, epsilon = 1e-8);
    assert!(kkt_residual(fcn, &ret.xmin, &[Some(-2.0); 10], &[Some(2.0); 10]) <= 1e-6);
}
#[test]
fn asa_minimizes_ten_dimensional_rosenbrock_outside_box() {
    let n = 10;
    let fcn = Rosenbrock { n };
    let x0 = DVector::<f64>::from_value_vec(-1.2, n, VecType::Col);
    let bounds = add_uniform_bounds(n, Some(-2.0), Some(0.99));

    let ret = solve_asa(fcn.clone(), x0, bounds, 2000);
    let expected_f = 5.516346e-02;
    let expected_x = DVector::<f64>::from_col_slice(
        &[
            9.90000e-01,
            9.90000e-01,
            9.90000e-01,
            9.90000e-01,
            9.90000e-01,
            9.87744e-01,
            9.79445e-01,
            9.61150e-01,
            9.24564e-01,
            8.54819e-01,
        ],
        10,
        1,
    );

    assert_vector_close(&ret.xmin, &expected_x, 1e-4);
    assert_relative_eq!(ret.fmin, expected_f, epsilon = 1e-8);
}
//}}}
//{{{ test: nnls style diagonal spd quadratic
#[test]
fn asa_minimizes_nnls_style_spd_quadratic_with_many_active_lower_bounds() {
    let n = 20;
    let mut unconstrained_minimum = DVector::<f64>::zeros_vec(n, VecType::Col);
    for i in 0..n {
        let value = 0.1 * (i + 1) as f64;
        unconstrained_minimum[i] = if i % 2 == 0 { -value } else { value };
    }

    let fcn = DiagonalSpdQuadratic::from_unconstrained_minimum(unconstrained_minimum.clone());
    let mut expected_x = unconstrained_minimum.clone();
    for i in 0..n {
        expected_x[i] = expected_x[i].max(0.0);
    }

    let x0 = DVector::<f64>::from_value_vec(1.0, n, VecType::Col);
    let bounds = add_uniform_bounds(n, Some(0.0), None);
    let ret = solve_asa(fcn.clone(), x0, bounds, 500);
    let active_count = ret.xmin.iter().filter(|xi| **xi <= 1e-8).count();
    let mut expected_fcn = fcn.clone();

    assert_vector_close(&ret.xmin, &expected_x, 1e-5);
    assert_relative_eq!(ret.fmin, expected_fcn.eval(&expected_x), epsilon = 1e-8);
    assert_eq!(active_count, n / 2);
    assert!(kkt_residual(fcn, &ret.xmin, &[Some(0.0); 20], &[None; 20]) <= 1e-6);
}
//}}}
//{{{ test: large-n shifted quadratic with a mix of active/inactive bounds
// Regression test for the ASA mixed-norm stopping-criterion bug: `is_converged`
// used to compare an L2 per-iteration residual against an L∞ initial reference
// (`norm_grad_fx_init`), making the rtol test up to sqrt(n) harder to satisfy as
// n grows. This mirrors `asa_minimizes_shifted_quadratic_with_active_upper_bound`
// but at n = 500, so a reintroduced dimension-dependent rtol would make it
// converge far more slowly (or not at all within a reasonable iteration budget)
// relative to the n = 5 case, rather than asserting a specific iteration count.
#[test]
fn asa_minimizes_large_n_shifted_quadratic_with_mixed_active_bounds() {
    let n = 500;
    let mut target = DVector::<f64>::zeros_vec(n, VecType::Col);
    for i in 0..n {
        target[i] = i as f64;
    }
    let fcn = ShiftedQuadratic { target };
    let x0 = DVector::<f64>::from_value_vec(125.0, n, VecType::Col);
    let upper = 250.0;
    let bounds = add_uniform_bounds(n, Some(0.0), Some(upper));

    let ret = solve_asa(fcn.clone(), x0, bounds, 2000);

    let mut expected_x = DVector::<f64>::zeros_vec(n, VecType::Col);
    for i in 0..n {
        expected_x[i] = (i as f64).min(upper);
    }
    let mut expected_fcn = fcn.clone();

    assert_vector_close(&ret.xmin, &expected_x, 1e-5);
    assert_relative_eq!(ret.fmin, expected_fcn.eval(&expected_x), epsilon = 1e-6);
    let lower_bounds: Vec<Option<f64>> = vec![Some(0.0); n];
    let upper_bounds: Vec<Option<f64>> = vec![Some(upper); n];
    assert!(kkt_residual(fcn, &ret.xmin, &lower_bounds, &upper_bounds) <= 1e-6);
}
//}}}
