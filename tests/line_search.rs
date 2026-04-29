#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::line_search::{
    search1d, LineSearchMethod, LineSearchOptions, NocedalOptions, ThuenteOptions,
};
use topohedral_optimize::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use approx::assert_relative_eq;
use ctor::ctor;
use topohedral_tracing::*;
//}}}

//{{{ fun: init_logger
#[ctor]
fn init_logger()
{
    init().unwrap();
}
//}}}
//{{{ collection Quadratic1D
#[derive(Debug, Clone)]
struct Quadratic1D
{
    pub root1: f64,
    pub root2: f64,
}
impl RealFn1 for Quadratic1D
{
    fn eval(
        &mut self,
        x: f64,
    ) -> f64
    {
        (x - self.root1) * (x - self.root2)
    }

    fn diff(
        &mut self,
        x: f64,
    ) -> f64
    {
        2.0 * x - (self.root1 + self.root2)
    }
}
//}}}
//{{{ collection: Cubic1D
#[derive(Debug, Clone)]
struct Cubic1D
{
    pub root1: f64,
    pub root2: f64,
    pub root3: f64,
}
impl RealFn1 for Cubic1D
{
    fn eval(
        &mut self,
        x: f64,
    ) -> f64
    {
        (x - self.root1) * (x - self.root2) * (x - self.root3)
    }

    fn diff(
        &mut self,
        x: f64,
    ) -> f64
    {
        let mut out = 0.0;
        out += (x - self.root2) * (x - self.root3);
        out += (x - self.root1) * (x - self.root3);
        out += (x - self.root1) * (x - self.root2);
        out
    }
}
//}}}
//{{{ colleciton: RationalQuad1D
#[derive(Clone, Copy, Debug)]
struct RationalQuad1D
{
    beta: f64,
}
impl RealFn1 for RationalQuad1D
{
    fn eval(
        &mut self,
        x: f64,
    ) -> f64
    {
        let alpha = x;
        -alpha / (alpha.powi(2) + self.beta)
    }

    fn diff(
        &mut self,
        x: f64,
    ) -> f64
    {
        let alpha = x;
        (alpha.powi(2) - self.beta) / (alpha.powi(2) + self.beta).powi(2)
    }
}
//}}}
//{{{ collection: thuente tests
#[test]
fn test_thuente_rational()
{
    let alpha_set = [1e-4, 500.0];
    let expected_vals = [
        (5.46100000e-01, -2.37618140e-01, -3.22193606e-01),
        (5.55500019e+01, -1.79901396e-02, 3.23435359e-04),
    ];

    let fcn1 = RationalQuad1D { beta: 2.0 };

    for (alpha, exp_vals) in alpha_set.iter().zip(expected_vals.iter())
    {
        let out = search1d(
            fcn1,
            *alpha,
            LineSearchMethod::Thuente(ThuenteOptions {
                ls_opts: LineSearchOptions {
                    step_max: 500.0,
                    ..Default::default()
                },
                maxiter: 100,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.alpha, exp_vals.0, epsilon = 1e-6);
        assert_relative_eq!(out.phi_alpha, exp_vals.1, epsilon = 1e-6);
    }
}

#[test]
fn test_thuente_quadratic()
{
    let root1 = 10.0;
    let root2 = 100.0;

    let fcn1 = Quadratic1D { root1, root2 };

    let alpha_set = [1e-4, 10.0];
    let expected_vals = [
        (8.73810000e+00, 1.15163392e+02, -9.25238000e+01),
        (10.0, 0.0, -9.00000000e+01),
    ];

    for (alpha, exp_vals) in alpha_set.iter().zip(expected_vals.iter())
    {
        let out = search1d(
            fcn1.clone(),
            *alpha,
            LineSearchMethod::Thuente(ThuenteOptions {
                ls_opts: LineSearchOptions {
                    step_max: 500.0,
                    ..Default::default()
                },
                maxiter: 100,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.alpha, exp_vals.0, epsilon = 1e-6);
        assert_relative_eq!(out.phi_alpha, exp_vals.1, epsilon = 1e-6);
    }
}
//}}}
//{{{ collection: nocedal tests
#[test]
fn test_nocedal_rational()
{
    let alpha_set = [1e-4, 500.0];
    let expected_vals = [
        (4.096e-01, -0.188949, -0.389899),
        (1.056683e2, -9.461885e-3, 8.9511233e-5),
    ];

    let fcn1 = RationalQuad1D { beta: 2.0 };

    for (alpha, exp_vals) in alpha_set.iter().zip(expected_vals.iter())
    {
        let out = search1d(
            fcn1,
            *alpha,
            LineSearchMethod::Nocedal(NocedalOptions {
                ls_opts: LineSearchOptions {
                    step_max: 500.0,
                    ..Default::default()
                },
                maxiter: 100,
                zoom_maxiter: 100,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.alpha, exp_vals.0, epsilon = 1e-3);
        assert_relative_eq!(out.phi_alpha, exp_vals.1, epsilon = 1e-3);
    }
}

#[test]
fn test_nocedal_quadratic()
{
    let root1 = 10.0;
    let root2 = 100.0;

    let fcn1 = Quadratic1D { root1, root2 };

    let alpha_set = [1e-4, 10.0];
    let expected_vals = [
        (6.553600e0, 3.220537e2, -9.689280e1),
        (10.0, 0.0, -9.00000000e+01),
    ];

    for (alpha, exp_vals) in alpha_set.iter().zip(expected_vals.iter())
    {
        let out = search1d(
            fcn1.clone(),
            *alpha,
            LineSearchMethod::Nocedal(NocedalOptions {
                ls_opts: LineSearchOptions {
                    step_max: 500.0,
                    ..Default::default()
                },
                maxiter: 100,
                zoom_maxiter: 100,
            }),
        )
        .unwrap();
        assert_relative_eq!(out.alpha, exp_vals.0, epsilon = 1e-5);
        assert_relative_eq!(out.phi_alpha, exp_vals.1, epsilon = 1e-3);
    }
}

#[test]
fn test_nocedal_cubic()
{
    let c1 = Cubic1D {
        root1: -1.0,
        root2: 0.0,
        root3: 1.0,
    };

    let out = search1d(
        c1.clone(),
        1.0,
        LineSearchMethod::Nocedal(NocedalOptions {
            ls_opts: LineSearchOptions {
                step_max: 500.0,
                ..Default::default()
            },
            maxiter: 100,
            zoom_maxiter: 100,
        }),
    )
    .unwrap();
    assert_relative_eq!(out.alpha, 5.0e-1, epsilon = 1e-6);
    assert_relative_eq!(out.phi_alpha, -3.75e-1, epsilon = 1e-6);
}
//}}}
