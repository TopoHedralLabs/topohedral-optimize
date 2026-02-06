#![feature(generic_const_exprs)]
#![allow(incomplete_features)]
#![feature(impl_trait_in_assoc_type)]

use approx::assert_relative_eq;
//{{{ crate imports
use topohedral_optimize::line_search::{
    InterpOptions, LineSearchMethod, LineSearchOptions, ThuenteOptions,
};
use topohedral_optimize::unconstrained::{
    ConjugateGradient, ConjugateGradientOptions, Direction, UnconstrainedMinimizer,
    UnonstrainedOptions, UnconstrainedReturns, UnconstrainedConvergedReason
};
use topohedral_optimize::RealFn;
//}}}
//{{{ std imports
use std::ops::Sub;
//}}}
//{{{ dep imports
use ctor::ctor;
use rstest::rstest;
use topohedral_linalg::{scvector::SCVector, VectorOps};
use topohedral_tracing::*;
//}}}

//{{{ fun: init_logger
#[ctor]
fn init_logger() {
    init().unwrap();
}
//}}}

//{{{ struct: Quadratic
#[derive(Debug, Clone, Copy)]
struct Quadratic {
    xmin: SCVector<f64, 5>,
}
//}}}
//{{{ impl: RealFn for Quadratic
impl RealFn for Quadratic {
    type Vector = SCVector<f64, 5>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(2);
        }
        out
    }

    fn grad(&mut self, x_in: &Self::Vector) -> Self::Vector {
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros();
        for i in 0..5 {
            out[i] = 2.0 * tmp[i];
        }
        out
    }
}
//}}}
//{{{ struct: Quartic
#[derive(Debug, Clone, Copy)]
struct Quartic {
    xmin: SCVector<f64, 5>,
}
//}}}
//{{{ impl: RealFn for Quartic
impl RealFn for Quartic {
    type Vector = SCVector<f64, 5>;

    fn eval(&mut self, x: &Self::Vector) -> f64 {
        let tmp: Self::Vector = (x - &self.xmin).into();
        let mut out = 0.0;
        for i in 0..5 {
            out += tmp[i].powi(4);
        }
        out
    }

    fn grad(&mut self, x_in: &Self::Vector) -> Self::Vector {
        let tmp: Self::Vector = (x_in - &self.xmin).into();
        let mut out = Self::Vector::zeros();
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
impl RealFn for Rosenbrock {
    type Vector = SCVector<f64, 2>;

    fn eval(&mut self, xvec: &Self::Vector) -> f64 {
        let x = xvec[0];
        let y = xvec[1];
        (self.a - x).powi(2) + self.b * (y - x.powi(2)).powi(2)
    }

    fn grad(&mut self, xvec: &Self::Vector) -> Self::Vector {
        let a = self.a;
        let b = self.b;
        let x = xvec[0];
        let y = xvec[1];
        let mut out = SCVector::<f64, 2>::zeros();
        out[0] = -2.0 * (a - x) - 4.0 * b * x * (y - x.powi(2));
        out[1] = 2.0 * b * (y - x.powi(2));
        out
    }
}
//}}}

fn assert_returns<T>(ret: &UnconstrainedReturns<T>, exp_ret: &UnconstrainedReturns<T>)
where 
T: VectorOps<ScalarType = f64> + Sub<Output = T> + Clone
{
    assert!((ret.xmin.clone() - exp_ret.xmin.clone()).norm() < 1e-8);
    assert!((ret.fmin - exp_ret.fmin).abs() < 1e-10);
    assert_eq!(ret.reason, exp_ret.reason);
    assert_eq!(ret.num_iterations, exp_ret.num_iterations);
    assert_eq!(ret.num_fun_evals, exp_ret.num_fun_evals);
    assert_eq!(ret.num_grad_evals, exp_ret.num_grad_evals);
}


#[rstest]
//{{{ case: test_quadratic_interp
#[case::test_quadratic_interp(
    SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-8,
                max_iter: 100,
                ls_method: LineSearchMethod::Interp(InterpOptions {
                    ls_opts: LineSearchOptions::default(),
                    scale_factor: 1.5,
                    maxiter: 10,
                }),
            },
            direction: Direction::Steepest,
            restart: 10,
        },
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 1, 
        num_fun_evals: 6, 
        num_grad_evals: 3
    }
)]
//}}}
//{{{ case: test_quadratic_thuente
#[case::test_quadratic_thuente(
    SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]),
    ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-8,
                max_iter: 100,
                ls_method: LineSearchMethod::Thuente(ThuenteOptions {
                    ls_opts: LineSearchOptions::default(),
                    maxiter: 100,
                }),
            },
            direction: Direction::Steepest,
            restart: 10,
        },
    UnconstrainedReturns{
        xmin:  SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]), 
        fmin: 0.0, 
        reason: UnconstrainedConvergedReason::Rtol,
        num_iterations: 3, 
        num_fun_evals: 9, 
        num_grad_evals: 12
    }
)]
//}}}
fn test_qudratic(#[case] x0: SCVector<f64, 5>, 
                 #[case] opts: ConjugateGradientOptions, 
                 #[case] exp_ret: UnconstrainedReturns<SCVector<f64, 5>>)
{
    let quad = Quadratic {
        xmin: SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
    };
    let mut cg = ConjugateGradient::new(quad, x0, opts);
    let ret = cg.minimize().unwrap();

    println!("{ret:?}");
    assert_returns(&ret, &exp_ret);
    // assert!((ret.xmin - exp_ret.xmin).norm() < 1e-8);
    // assert!((ret.fmin - exp_ret.fmin).abs() < 1e-10);
    // assert_eq!(ret.reason, exp_ret.reason);
    // assert_eq!(ret.num_iterations, exp_ret.num_iterations);
    // assert_eq!(ret.num_fun_evals, exp_ret.num_fun_evals);
    // assert_eq!(ret.num_grad_evals, exp_ret.num_grad_evals);
}


//{{{ test: test_quadratic_interp
// #[test]
// fn test_quadratic_interp() {
//     let quad = Quadratic {
//         xmin: SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
//     };

//     let x0 = SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);

//     let mut cg = ConjugateGradient::new(
//         quad,
//         x0,
//         ConjugateGradientOptions {
//             uncon_opts: UnonstrainedOptions {
//                 grad_rtol: 1e-6,
//                 grad_atol: 1e-8,
//                 max_iter: 100,
//                 ls_method: LineSearchMethod::Interp(InterpOptions {
//                     ls_opts: LineSearchOptions::default(),
//                     scale_factor: 1.5,
//                     maxiter: 10,
//                 }),
//             },
//             direction: Direction::Steepest,
//             restart: 10,
//         },
//     );

//     let ret = cg.minimize().unwrap();

//     print!("{ret:?}")
// }
// //}}}
// //{{{ test: test_quadratic_thuente
// #[test]
// fn test_quadratic_thuente() {
//     let quad = Quadratic {
//         xmin: SCVector::<f64, 5>::from_col_slice(&[1000.0, -100.0, 0.0, 567.0, -23.0]),
//     };

//     let x0 = SCVector::<f64, 5>::from_col_slice(&[0.0, 0.0, 0.0, 0.0, 0.0]);

//     let mut cg = ConjugateGradient::new(
//         quad,
//         x0,
//         ConjugateGradientOptions {
//             uncon_opts: UnonstrainedOptions {
//                 grad_rtol: 1e-6,
//                 grad_atol: 1e-8,
//                 max_iter: 100,
//                 ls_method: LineSearchMethod::Thuente(ThuenteOptions {
//                     ls_opts: LineSearchOptions::default(),
//                     maxiter: 100,
//                 }),
//             },
//             direction: Direction::Steepest,
//             restart: 10,
//         },
//     );

//     let ret = cg.minimize().unwrap();

//     print!("{ret:?}")
// }
//}}}
//{{{ test: test_quartic_interp
#[test]
fn test_quartic_interp() {
    let quart = Quartic {
        xmin: SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let scale_factor = 30.0;
    let mut offset_dir = SCVector::<f64, 5>::from_col_slice(&[1e-3, 1.0, 0.5, 3.0, 1.0]);
    offset_dir = offset_dir.normalize();

    let x0 = quart.xmin + scale_factor * offset_dir;

    let mut cg = ConjugateGradient::new(
        quart,
        x0,
        ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-8,
                grad_atol: 1e-10,
                max_iter: 1000,
                ls_method: LineSearchMethod::Interp(InterpOptions {
                    ls_opts: LineSearchOptions::default(),
                    scale_factor: 1.2,
                    maxiter: 10,
                }),
            },
            direction: Direction::PolakRibiere,
            restart: 100,
        },
    );

    let ret = cg.minimize().unwrap();

    print!("{ret:?}")
}
//}}}
//{{{ test: test_quartic_thuente
#[test]
fn test_quartic_thuente() {
    let quart = Quartic {
        xmin: SCVector::<f64, 5>::from_col_slice(&[10.0, 10.0, 10.0, 10.0, 10.0]),
    };

    let scale_factor = 30.0;
    let mut offset_dir = SCVector::<f64, 5>::from_col_slice(&[1e-3, 1.0, 0.5, 3.0, 1.0]);
    offset_dir = offset_dir.normalize();

    let x0 = quart.xmin + scale_factor * offset_dir;

    let mut cg = ConjugateGradient::new(
        quart,
        x0,
        ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-8,
                grad_atol: 1e-10,
                max_iter: 1000,
                ls_method: LineSearchMethod::Thuente(ThuenteOptions {
                    ls_opts: LineSearchOptions {
                        c1: 1e-4,
                        c2: 0.4,
                        step_min: 1e-8,
                        step_max: 1e9,
                    },
                    maxiter: 100,
                }),
            },
            direction: Direction::PolakRibiere,
            restart: 100,
        },
    );

    let ret = cg.minimize().unwrap();

    print!("{ret:?}")
}
//}}}
//{{{ test: test_rosenbrock_interp
#[test]
fn test_rosenbrock_interp() {
    let rosenbrock = Rosenbrock::new();

    let x0 = SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]);

    let mut cg = ConjugateGradient::new(
        rosenbrock,
        x0,
        ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-6,
                grad_atol: 1e-8,
                max_iter: 1000,
                ls_method: LineSearchMethod::Interp(InterpOptions {
                    ls_opts: LineSearchOptions {
                        c1: 1e-4,
                        c2: 0.4,
                        step_min: 1e-8,
                        step_max: 1e9,
                    },
                    scale_factor: 1.5,
                    maxiter: 30,
                }),
            },
            direction: Direction::FletcherReeves,
            restart: 100,
        },
    );

    let ret = cg.minimize().unwrap();

    print!("{ret:?}")
}
//}}}
//{{{ test: test_rosenbrock_thuente
#[test]
fn test_rosenbrock_thuente() {
    let rosenbrock = Rosenbrock::new();

    let x0 = SCVector::<f64, 2>::from_col_slice(&[0.0, 3.0]);

    let mut cg = ConjugateGradient::new(
        rosenbrock,
        x0,
        ConjugateGradientOptions {
            uncon_opts: UnonstrainedOptions {
                grad_rtol: 1e-9,
                grad_atol: 1e-10,
                max_iter: 1000,
                ls_method: LineSearchMethod::Thuente(ThuenteOptions {
                    ls_opts: LineSearchOptions {
                        c1: 1e-4,
                        c2: 0.4,
                        step_min: 1e-8,
                        step_max: 1e9,
                    },
                    maxiter: 100,
                }),
            },
            direction: Direction::PolakRibiere,
            restart: 100,
        },
    );

    let ret = cg.minimize().unwrap();

    print!("{ret:?}")
}
//}}}
