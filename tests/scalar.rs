//! Integration tests for the `scalar` module: unbounded minimization via
//! Brent's method and golden-section search, and bounded minimization.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use topohedral_optimize::RealFn1;
use topohedral_optimize::{
    scalar_minimze as minimize, BoundedOptions, Bracket, BrentOptions, GoldenOptions, ScalarError,
    ScalarMethod,
};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ fun: init_logger
#[ctor]
fn init_logger() {
    init().unwrap();
}
//}}}
//{{{ struct: ScalarFunction
struct ScalarFunction<G: Fn(f64) -> f64> {
    f: G,
}
//}}}
//{{{ impl: ScalarFunction
impl<G: Fn(f64) -> f64> ScalarFunction<G> {
    fn new(f: G) -> Self {
        Self { f }
    }
}
//}}}
//{{{ impl: RealFn1 for ScalarFunction
impl<G: Fn(f64) -> f64> RealFn1 for ScalarFunction<G> {
    fn eval(
        &mut self,
        x: f64,
    ) -> f64 {
        (self.f)(x)
    }

    fn diff(
        &mut self,
        _x: f64,
    ) -> f64 {
        unimplemented!("not needed by scalar minimizers")
    }
}
//}}}
//{{{ fun: parabola
fn parabola(x: f64) -> f64 {
    (x - 1.0).powi(2)
}
//}}}
//{{{ fun: quartic
fn quartic(x: f64) -> f64 {
    (x - 2.0).powi(4) + 3.0
}
//}}}
//{{{ enum: ScalarSolver
/// `Brent` and `Golden` are both bracket-based unbounded solvers with an
/// identical `Options` shape (`bracket`, `xtol`, `max_iter`) and error
/// behaviour, so their tests are parameterized over this enum via `rstest`
/// rather than duplicated per-solver.
#[derive(Clone, Copy)]
enum ScalarSolver {
    Brent,
    Golden,
}
//}}}
//{{{ impl: ScalarSolver
impl ScalarSolver {
    fn method(
        self,
        bracket: Bracket,
    ) -> ScalarMethod {
        match self {
            ScalarSolver::Brent => ScalarMethod::Brent(BrentOptions::new(bracket)),
            ScalarSolver::Golden => ScalarMethod::Golden(GoldenOptions::new(bracket)),
        }
    }

    fn method_with(
        self,
        bracket: Bracket,
        xtol: f64,
        max_iter: usize,
    ) -> ScalarMethod {
        match self {
            ScalarSolver::Brent => {
                let mut opts = BrentOptions::new(bracket);
                opts.xtol = xtol;
                opts.max_iter = max_iter;
                ScalarMethod::Brent(opts)
            }
            ScalarSolver::Golden => {
                let mut opts = GoldenOptions::new(bracket);
                opts.xtol = xtol;
                opts.max_iter = max_iter;
                ScalarMethod::Golden(opts)
            }
        }
    }
}
//}}}

//{{{ mod: brent and golden (bracket-based unbounded solvers)
#[rstest]
#[case::brent(ScalarSolver::Brent, 1e-6)]
#[case::golden(ScalarSolver::Golden, 1e-4)]
fn test_parabola_two_point_bracket(
    #[case] solver: ScalarSolver,
    #[case] xmin_tol: f64,
) {
    let f = ScalarFunction::new(parabola);
    let res = minimize(f, solver.method(Bracket::Points(0.0, 1.0))).unwrap();
    assert!((res.xmin - 1.0).abs() < xmin_tol);
}

#[rstest]
#[case::brent(ScalarSolver::Brent, 1e-6)]
#[case::golden(ScalarSolver::Golden, 1e-4)]
fn test_parabola_three_point_bracket(
    #[case] solver: ScalarSolver,
    #[case] xmin_tol: f64,
) {
    let f = ScalarFunction::new(parabola);
    let res = minimize(f, solver.method(Bracket::Triple(-1.0, 0.5, 3.0))).unwrap();
    assert!((res.xmin - 1.0).abs() < xmin_tol);
}

#[rstest]
#[case::brent(ScalarSolver::Brent, 1e-6)]
#[case::golden(ScalarSolver::Golden, 1e-4)]
fn test_parabola_auto_bracket(
    #[case] solver: ScalarSolver,
    #[case] xmin_tol: f64,
) {
    let f = ScalarFunction::new(parabola);
    let res = minimize(f, solver.method(Bracket::Auto)).unwrap();
    assert!((res.xmin - 1.0).abs() < xmin_tol);
}

#[rstest]
#[case::brent(ScalarSolver::Brent, Some(1e-3))]
#[case::golden(ScalarSolver::Golden, None)]
fn test_quartic(
    #[case] solver: ScalarSolver,
    #[case] fmin_tol: Option<f64>,
) {
    let f = ScalarFunction::new(quartic);
    let res = minimize(f, solver.method(Bracket::Points(0.0, 1.0))).unwrap();
    assert!((res.xmin - 2.0).abs() < 1e-2);
    if let Some(tol) = fmin_tol {
        assert!((res.fmin - 3.0).abs() < tol);
    }
}

#[rstest]
#[case::brent(ScalarSolver::Brent)]
#[case::golden(ScalarSolver::Golden)]
fn test_invalid_triple_order(#[case] solver: ScalarSolver) {
    let f = ScalarFunction::new(parabola);
    let result = minimize(f, solver.method(Bracket::Triple(0.0, 0.9, 0.5)));
    assert!(matches!(
        result,
        Err(ScalarError::InvalidBracketOrder(_, _, _))
    ));
}

#[rstest]
#[case::brent(ScalarSolver::Brent)]
#[case::golden(ScalarSolver::Golden)]
fn test_invalid_triple_values(#[case] solver: ScalarSolver) {
    // Monotone function: f(xb) is not below both f(xa) and f(xc).
    let f = ScalarFunction::new(|x: f64| x);
    let result = minimize(f, solver.method(Bracket::Triple(0.0, 0.5, 1.0)));
    assert!(matches!(result, Err(ScalarError::InvalidBracketValues)));
}

#[rstest]
#[case::brent(ScalarSolver::Brent)]
#[case::golden(ScalarSolver::Golden)]
fn test_max_iterations_exceeded(#[case] solver: ScalarSolver) {
    let f = ScalarFunction::new(parabola);
    let result = minimize(f, solver.method_with(Bracket::Points(0.0, 1.0), 1e-14, 2));
    assert!(matches!(result, Err(ScalarError::MaxIterations(2))));
}
//}}}

//{{{ mod: bounded
#[rstest]
#[case::parabola_symmetric_bounds(parabola as fn(f64) -> f64, -4.0, 4.0, 1.0, 1e-4, Some(1e-8))]
#[case::cosine(f64::cos as fn(f64) -> f64, 0.0, 2.0 * std::f64::consts::PI, std::f64::consts::PI, 1e-4, None)]
#[case::clamps_to_lower_bound(parabola as fn(f64) -> f64, 3.0, 4.0, 3.0, 1e-3, None)]
fn test_bounded_xmin(
    #[case] f: fn(f64) -> f64,
    #[case] lower: f64,
    #[case] upper: f64,
    #[case] expected_xmin: f64,
    #[case] xmin_tol: f64,
    #[case] fmin_tol: Option<f64>,
) {
    let fcn = ScalarFunction::new(f);
    let opts = BoundedOptions::new(lower, upper).unwrap();
    let res = minimize(fcn, ScalarMethod::Bounded(opts)).unwrap();
    assert!((res.xmin - expected_xmin).abs() < xmin_tol);
    if let Some(tol) = fmin_tol {
        assert!(res.fmin < tol);
    }
}

#[test]
fn test_bounded_invalid_bounds() {
    let result = BoundedOptions::new(4.0, 3.0);
    assert!(matches!(result, Err(ScalarError::InvalidBounds(lo, hi)) if lo == 4.0 && hi == 3.0));
}

#[test]
fn test_bounded_non_finite_bounds() {
    let result = BoundedOptions::new(f64::NEG_INFINITY, 4.0);
    assert!(matches!(result, Err(ScalarError::NonFiniteBounds(_, _))));
}

#[test]
fn test_bounded_max_iterations_exceeded() {
    let f = ScalarFunction::new(parabola);
    let mut opts = BoundedOptions::new(-4.0, 4.0).unwrap();
    opts.xatol = 1e-12;
    opts.max_iter = 2;
    let result = minimize(f, ScalarMethod::Bounded(opts));
    assert!(matches!(result, Err(ScalarError::MaxIterations(2))));
}
//}}}
