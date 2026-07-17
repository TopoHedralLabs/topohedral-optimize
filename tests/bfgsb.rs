use approx::assert_relative_eq;
use ctor::ctor;
use topohedral_linalg::{DMatrix, DVector, MatMul, ReduceOps, VecType, VectorOps};
use topohedral_optimize::DifferentiableFn;
use topohedral_optimize::{
    bound_constrained_minimize, BaseOptions, BfgsbOptions, BoundConstrainedMethod,
    BoundConstrainedOptions, BoundsConstraints, LineSearchMethod, LineSearchOptions, Matrix,
    NocedalOptions, RealFn, Vector, VectorReturns,
};
use topohedral_tracing::*;

#[ctor]
fn init_logger() {
    init().unwrap();
}

fn colvec(values: &[f64]) -> Vector {
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

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

fn bfgsb_options(
    pgtol: f64,
    ftol: f64,
    max_iter: u64,
) -> BfgsbOptions {
    BfgsbOptions {
        bound_opts: BoundConstrainedOptions {
            base_opts: BaseOptions {
                grad_rtol: 0.0,
                grad_atol: pgtol,
                max_iter,
            },
            constraint_tol: ftol,
        },
        ls_method: LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                c1: 1.0e-4,
                c2: 0.9,
                step_min: 1e-20,
                step_max: 1e20,
            },
            maxiter: 50,
            zoom_maxiter: 50,
        }),
    }
}

fn empty_bounds(n: usize) -> BoundsConstraints {
    BoundsConstraints::new(n)
}

fn bounds_from_pairs(pairs: &[(Option<f64>, Option<f64>)]) -> BoundsConstraints {
    let mut bounds = BoundsConstraints::new(pairs.len());
    for (i, (lower, upper)) in pairs.iter().enumerate() {
        if lower.is_some() || upper.is_some() {
            bounds.add_bounds(i, *lower, *upper);
        }
    }
    bounds
}

fn solve_bfgsb<F: RealFn>(
    mut fcn: F,
    x0: Vector,
    bounds: BoundsConstraints,
    pgtol: f64,
    ftol: f64,
    max_iter: u64,
) -> VectorReturns {
    bound_constrained_minimize(
        &mut fcn,
        bounds,
        x0,
        BoundConstrainedMethod::Bfgsb(bfgsb_options(pgtol, ftol, max_iter)),
    )
    .expect("BFGS-B minimization should succeed")
}

fn kkt_residual<F: RealFn>(
    mut fcn: F,
    x: &Vector,
    bounds: &BoundsConstraints,
) -> f64 {
    let grad = fcn.derivative(x);
    let projected = bounds.projected_direction(x, &(-grad), 1.0);
    projected.abs_max().unwrap_or(0.0)
}

#[derive(Clone, Debug)]
struct Quadratic {
    a: Matrix,
    b: Vector,
}

impl topohedral_optimize::DifferentiableFn for Quadratic {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        self.b.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let ax = self.a.matmul(x);
        0.5 * x.dot(&ax) - self.b.dot(x)
    }

    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Vector {
        self.a.matmul(x) - self.b.clone()
    }
}

#[derive(Clone, Debug)]
struct Linear {
    c: Vector,
}

impl topohedral_optimize::DifferentiableFn for Linear {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        self.c.len()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.c.dot(x)
    }

    fn derivative(
        &mut self,
        _x: &Vector,
    ) -> Vector {
        self.c.clone()
    }
}

#[derive(Clone, Debug)]
struct Rosenbrock {
    n: usize,
}

impl topohedral_optimize::DifferentiableFn for Rosenbrock {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
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

    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let mut grad = Vector::zeros_vec(self.n, VecType::Col);
        for i in 0..(self.n - 1) {
            let t = x[i + 1] - x[i].powi(2);
            grad[i] += -400.0 * x[i] * t - 2.0 * (1.0 - x[i]);
            grad[i + 1] += 200.0 * t;
        }
        grad
    }
}

#[derive(Clone, Debug)]
struct FiniteDiff<F: RealFn> {
    fcn: F,
    eps: f64,
}

impl<F: RealFn> topohedral_optimize::DifferentiableFn for FiniteDiff<F> {
    type Input = Vector;
    type Output = f64;
    type Derivative = Vector;
    fn dimension_domain(&self) -> usize {
        self.fcn.dimension_domain()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.fcn.eval(x)
    }

    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let f0 = self.fcn.eval(x);
        let mut grad = Vector::zeros_vec(x.len(), VecType::Col);
        for i in 0..x.len() {
            let mut xp = x.clone();
            xp[i] += self.eps;
            grad[i] = (self.fcn.eval(&xp) - f0) / self.eps;
        }
        grad
    }
}

fn rosenbrock_gradient_matches_finite_difference() {
    let mut fcn = Rosenbrock { n: 5 };
    let x = colvec(&[-1.2, 1.0, 0.5, 0.0, 2.0]);
    let analytic = fcn.derivative(&x);
    let mut finite_diff = FiniteDiff { fcn, eps: 1e-6 };
    let numerical = finite_diff.derivative(&x);
    assert_vector_close(&analytic, &numerical, 1e-3);
}

#[test]
fn unconstrained_quadratic_matches_analytic_minimum() {
    let a = DMatrix::<f64>::from_row_slice(&[3.0, 0.5, 0.5, 2.0], 2, 2);
    let b = colvec(&[1.0, 2.0]);
    let x_star = a.solve(&b).expect("quadratic system should solve");
    let ret = solve_bfgsb(
        Quadratic {
            a: a.clone(),
            b: b.clone(),
        },
        colvec(&[0.0, 0.0]),
        empty_bounds(2),
        1e-5,
        1e-15,
        100,
    );

    assert_vector_close(&ret.xmin, &x_star, 1e-5);
    assert!(kkt_residual(Quadratic { a, b }, &ret.xmin, &empty_bounds(2)) < 1e-4);
    assert!(ret.num_iterations <= 10);
}

#[test]
fn inactive_bounds_recover_unconstrained_quadratic_minimum() {
    let a = DMatrix::<f64>::from_row_slice(&[3.0, 0.5, 0.5, 2.0], 2, 2);
    let b = colvec(&[1.0, 2.0]);
    let x_star = a.solve(&b).expect("quadratic system should solve");
    let ret = solve_bfgsb(
        Quadratic { a, b },
        colvec(&[4.0, -4.0]),
        bounds_from_pairs(&[(Some(-5.0), Some(5.0)), (Some(-5.0), Some(5.0))]),
        1e-5,
        1e-15,
        100,
    );

    assert_vector_close(&ret.xmin, &x_star, 1e-5);
}

#[test]
fn active_lower_bound_quadratic_satisfies_reduced_stationarity() {
    let a = DMatrix::<f64>::from_row_slice(&[3.0, 0.5, 0.5, 2.0], 2, 2);
    let b = colvec(&[1.0, 2.0]);
    let ret = solve_bfgsb(
        Quadratic {
            a: a.clone(),
            b: b.clone(),
        },
        colvec(&[5.0, 5.0]),
        bounds_from_pairs(&[(Some(1.0), None), (None, None)]),
        1e-5,
        1e-15,
        100,
    );

    let x1_star = (b[1] - a[(1, 0)] * 1.0) / a[(1, 1)];
    assert_relative_eq!(ret.xmin[0], 1.0, epsilon = 1e-6);
    assert_relative_eq!(ret.xmin[1], x1_star, epsilon = 1e-5);
}

#[test]
fn linear_objective_converges_to_box_corner() {
    let n = 3;
    let ret = solve_bfgsb(
        Linear {
            c: colvec(&[-1.0, -2.0, -3.0]),
        },
        colvec(&[0.5, 0.5, 0.5]),
        bounds_from_pairs(&vec![(Some(0.0), Some(1.0)); n]),
        1e-6,
        1e-15,
        100,
    );

    assert_vector_close(&ret.xmin, &colvec(&[1.0, 1.0, 1.0]), 1e-6);
}

#[test]
fn rosenbrock_unconstrained_converges_to_global_minimum() {
    rosenbrock_gradient_matches_finite_difference();

    let n = 5;
    let ret = solve_bfgsb(
        Rosenbrock { n },
        DVector::<f64>::from_value_vec(-1.2, n, VecType::Col),
        empty_bounds(n),
        1e-8,
        1e-15,
        2_000,
    );

    assert_vector_close(
        &ret.xmin,
        &DVector::<f64>::from_value_vec(1.0, n, VecType::Col),
        1e-3,
    );
    assert!(ret.fmin < 1e-8);
}

#[test]
fn rosenbrock_with_active_box_bounds_is_feasible_and_kkt_small() {
    let n = 4;
    let bounds = bounds_from_pairs(&vec![(Some(0.0), Some(0.5)); n]);
    let ret = solve_bfgsb(
        Rosenbrock { n },
        DVector::<f64>::from_value_vec(0.3, n, VecType::Col),
        bounds.clone(),
        1e-8,
        1e-15,
        3_000,
    );

    for xi in ret.xmin.iter() {
        assert!(*xi >= -1e-9);
        assert!(*xi <= 0.5 + 1e-9);
    }
    assert!(kkt_residual(Rosenbrock { n }, &ret.xmin, &bounds) < 1e-5);
}

#[test]
fn high_dimensional_quadratic_with_mixed_bounds_is_feasible() {
    let n = 50;
    let mut a = Matrix::identity(n, n);
    let mut b = Vector::zeros_vec(n, VecType::Col);
    for i in 0..n {
        a[(i, i)] = 10.0 + i as f64;
        b[i] = if i % 3 == 0 {
            1.0
        } else {
            -1.0 + 0.04 * i as f64
        };
    }

    let pairs: Vec<(Option<f64>, Option<f64>)> = (0..n)
        .map(|i| {
            if i % 3 == 0 {
                (None, Some(0.0))
            } else {
                (None, None)
            }
        })
        .collect();
    let bounds = bounds_from_pairs(&pairs);
    let ret = solve_bfgsb(
        Quadratic {
            a: a.clone(),
            b: b.clone(),
        },
        Vector::zeros_vec(n, VecType::Col),
        bounds.clone(),
        1e-6,
        1e-15,
        2_000,
    );

    for i in (0..n).step_by(3) {
        assert!(ret.xmin[i] <= 1e-8);
    }
    assert!(kkt_residual(Quadratic { a, b }, &ret.xmin, &bounds) < 1e-5);
}

#[test]
fn finite_difference_wrapped_rosenbrock_converges() {
    let n = 3;
    let ret = solve_bfgsb(
        FiniteDiff {
            fcn: Rosenbrock { n },
            eps: 1e-7,
        },
        colvec(&[-1.2, 1.0, 0.5]),
        empty_bounds(n),
        1e-5,
        1e-15,
        500,
    );

    assert_vector_close(
        &ret.xmin,
        &DVector::<f64>::from_value_vec(1.0, n, VecType::Col),
        1e-2,
    );
}

#[test]
fn kkt_residual_is_small_on_representative_problems() {
    let a = DMatrix::<f64>::from_row_slice(&[3.0, 0.5, 0.5, 2.0], 2, 2);
    let b = colvec(&[1.0, 2.0]);
    let bounds_quad = empty_bounds(2);
    let ret_quad = solve_bfgsb(
        Quadratic {
            a: a.clone(),
            b: b.clone(),
        },
        colvec(&[0.0, 0.0]),
        bounds_quad.clone(),
        1e-8,
        1e-15,
        200,
    );
    assert!(kkt_residual(Quadratic { a, b }, &ret_quad.xmin, &bounds_quad) <= 1e-5);

    let bounds_rosen = empty_bounds(5);
    let ret_rosen = solve_bfgsb(
        Rosenbrock { n: 5 },
        DVector::<f64>::from_value_vec(-1.2, 5, VecType::Col),
        bounds_rosen.clone(),
        1e-8,
        1e-15,
        5_000,
    );
    assert!(kkt_residual(Rosenbrock { n: 5 }, &ret_rosen.xmin, &bounds_rosen) <= 1e-5);

    let bounds_box = bounds_from_pairs(&[(Some(0.0), Some(0.5)); 4]);
    let ret_box = solve_bfgsb(
        Rosenbrock { n: 4 },
        DVector::<f64>::from_value_vec(0.3, 4, VecType::Col),
        bounds_box.clone(),
        1e-8,
        1e-15,
        5_000,
    );
    assert!(kkt_residual(Rosenbrock { n: 4 }, &ret_box.xmin, &bounds_box) <= 1e-5);
}
