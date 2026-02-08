//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
use crate::common::RealFn1;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::{scvector::SCVector, smatrix::SMatrix, MatMul};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------
const SMALL: f64 = 1e-32;

//{{{ fun: quadmin
/// Forms quadratic interpolation to find the minimum of a function.
///
/// Forms the function:
///
/// phi_q(x) = beta (x - a)^2 + gamma (x - a) + delta
///
/// And finds the minimum of this function ananlytically.
pub fn quadmin(
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
) -> Option<f64>
{
    //{{{ trace
    error!(target: "ls", "--- Entering quadmin ---");
    trace!(target: "ls", "Entering with phi_a = {:1.4e}, phi_b = {:1.4e}, dphi_a = {:1.4e}, b = {:1.4e}", phi_a, phi_b, dphi_a, b);
    //}}}
    let delta = phi_a;
    let gamma = dphi_a;
    let db = b - a;

    //{{{ trace
    trace!(target: "ls", "db = {:1.4e}", db);
    //}}}

    if db * db < SMALL
    {
        //{{{ trace
        trace!(target: "ls", "db * db too small");
        error!(target: "ls", "--- Leaving quadmin ---");
        //}}}
        return None;
    }

    let beta = (phi_b - delta - gamma * db) / (db * db);
    //{{{ trace
    trace!(target: "ls", "beta = {:1.4e}", beta);
    //}}}

    if (2.0 * beta).abs() < SMALL
    {
        //{{{ trace
        trace!(target: "ls", "2 * beta too small");
        error!(target: "ls", "--- Leaving quadmin ---");
        //}}}
        return None;
    }

    let alpha_min = a - gamma / (2.0 * beta);
    //{{{ trace
    error!(target: "ls", "Returning alpha_min = {:1.4e}", alpha_min);
    error!(target: "ls", "--- Leaving quadmin ---");
    //}}}
    Some(alpha_min)
}
//}}}
//{{{ fun: cubicmin2
#[allow(dead_code)]
pub fn cubicmin2(
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    dphi_b: f64,
) -> Option<f64>
{
    //{{{ trace
    error!(target: "ls", "--- Entering cubicmin2 ---");
    trace!(target: "ls", "Entering with a = {:1.4e}, b = {:1.4e}", a, b);
    trace!(target: "ls", "phi_a = {:1.4e}, phi_b = {:1.4e}", phi_a, phi_b);
    trace!(target: "ls", "dphi_a = {:1.4e}, dphi_b = {:1.4e}", dphi_a, dphi_b);
    //}}}

    let d = b - a;
    if d.abs() <= f64::EPSILON
    {
        return None;
    }

    let quad_coeff = (-2.0 * d * dphi_a - d * dphi_b - 3.0 * phi_a + 3.0 * phi_b) / (d * d);
    let cube_coeff = (d * dphi_a + d * dphi_b + 2.0 * phi_a - 2.0 * phi_b) / (d * d * d);

    //{{{ trace
    trace!(target: "ls", "quad_coeff = {:1.4e} cub_coeff = {:1.4e}", quad_coeff, cube_coeff);
    //}}}

    // Solve 3B t^2 + 2A t + ga = 0
    let c2 = 3.0 * cube_coeff;
    let c1 = 2.0 * quad_coeff;
    let c0 = dphi_a;

    let disc = c1 * c1 - 4.0 * c2 * c0;
    if disc < 0.0
    {
        return None;
    } // no real stationary points
    if c2.abs() < f64::EPSILON
    {
        return None;
    } // degenerate cubic

    let sqrt_disc = disc.sqrt();
    let t1 = (-c1 + sqrt_disc) / (2.0 * c2);
    let t2 = (-c1 - sqrt_disc) / (2.0 * c2);

    // pick feasible minimizer in (0,d) with positive second derivative
    let candidates = [t1, t2].into_iter().filter(|t| *t > 0.0 && *t < d);
    for t in candidates
    {
        let p2 = 2.0 * quad_coeff + 6.0 * cube_coeff * t;
        if p2 > 0.0
        {
            return Some(a + t);
        }
    }
    None
}
//}}}
//{{{ fun: cubicmin3
/// Forms the cubic interpolation to find the minimum of a function using 3 points: $a$, $b$ and $c$
/// and 4 pieces of information:
/// - $\phi(a)$ and $\phi^{'}(a)$
/// - $\phi(b)$
/// - $\phi(c)$
///
/// Forms the function:
///
/// phi_cu(x) = beta(x - a)^3 + gamma (x - a)^2 + delta (x - a) + epsilon
///
/// And finds the minimum of this function ananlytically.
pub fn cubicmin3(
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    c: f64,
    phi_c: f64,
) -> Option<f64>
{
    //{{{ trace
    error!(target: "ls", "--- Entering cubicmin3 ---");
    trace!(target: "ls", "Enterin with a = {:1.4e}, b = {:1.4e}, c = {:1.4e}", a, b, c);
    trace!(target: "ls", "phi_a = {:1.4e}, phi_b = {:1.4e}, phi_c = {:1.4e}", phi_a, phi_b, phi_c);
    //}}}
    let db = b - a;
    let dc = c - a;
    let denom = (db * dc).powi(2) * (db - dc);
    //{{{ trace
    trace!(target: "ls", "db = {:1.4e}, dc = {:1.4e}, denom = {:1.4e}", db, dc, denom);
    //}}}

    if denom.abs() < SMALL
    {
        //{{{ trace
        trace!(target: "ls", "Denominator is too small");
        error!(target: "ls", "--- Leaving cubicmin ---");
        //}}}
        return None;
    }
    let mut diff_mat: SMatrix<f64, 2, 2> = SMatrix::<f64, 2, 2>::zeros();
    diff_mat[(0, 0)] = dc.powi(2);
    diff_mat[(0, 1)] = -db.powi(2);
    diff_mat[(1, 0)] = -dc.powi(3);
    diff_mat[(1, 1)] = db.powi(3);

    let mut diff_vec: SCVector<f64, 2> = SCVector::<f64, 2>::zeros();
    diff_vec[0] = phi_b - phi_a - dphi_a * db;
    diff_vec[1] = phi_c - phi_a - dphi_a * dc;

    let coeffs = diff_mat.matmul(&diff_vec);
    let beta = coeffs[0] / denom;
    let gamma = coeffs[1] / denom;
    let radical = (gamma * gamma - 3.0 * beta * dphi_a).sqrt();

    if (3.0 * beta).abs() < SMALL
    {
        //{{{ trace
        error!(target: "ls", "3 * beta too small");
        error!(target: "ls", "--- Leaving cubicmin ---");
        //}}}
        return None;
    }

    let alpha_min = a + (-gamma + radical) / (3.0 * beta);
    //{{{ trace
    error!(target: "ls", "Returning alpha_min = {:1.4e}", alpha_min);
    error!(target: "ls", "--- Leaving cubicmin ---");
    //}}}
    if alpha_min.is_nan()
    {
        //{{{ trace
        error!("Final result is Nan, returning None");
        error!(target: "ls", "--- Leaving cubicmin ---");
        //}}}
        return None;
    }
    Some(alpha_min)
}
//}}}
//{{{ fun: quadcubmin
#[allow(clippy::too_many_arguments)]
pub fn quadcubmin<F: RealFn1>(
    f: &mut F,
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    c: f64,
    phi_c: f64,
) -> Option<(f64, f64)>
{
    //{{{ trace
    info!(target: "ls", "--- entering quadcubmin ---");
    //}}}
    let to_pair = |x: &Option<f64>| -> Option<(f64, f64)> {
        x.as_ref().map(|alpha| (*alpha, f.eval(*alpha)))
    };
    let cubic_min_alpha = cubicmin3(a, phi_a, dphi_a, b, phi_b, c, phi_c);
    let quad_min1_alpha = quadmin(a, phi_a, dphi_a, b, phi_b);
    let quad_min2_alpha = quadmin(a, phi_a, dphi_a, c, phi_c);
    let opt_min_value = [cubic_min_alpha, quad_min1_alpha, quad_min2_alpha]
        .iter()
        .map(to_pair)
        .filter_map(|x| {
            //{{{ trace
            trace!(target: "ls", "alpha-falpha pair: {:?}", x);
            //}}}
            x
        })
        .min_by(|x, y| x.1.partial_cmp(&y.1).unwrap());

    if opt_min_value.is_none()
    {
        //{{{ trace
        info!(target: "ls", "No value found");
        info!(target: "ls", "--- leaving quadcubmin ---");
        //}}}
        return None;
    }

    let (alpha_min, fmin) = opt_min_value.unwrap();
    //{{{ trace
    trace!(target: "ls", "returning alpha ={alpha_min:1.4e}");
    info!(target: "ls", "--- leaving quadcubmin ---");
    //}}}
    Some((alpha_min, fmin))
}
//}}}
//{{{ fun: satisfies_armijo
pub fn satisfies_armijo(
    c1: f64,
    alpha: f64,
    phi0: f64,
    dphi0: f64,
    phi1: f64,
) -> bool
{
    //{{{ trace
    trace!(target: "ls", "armijo: left = {:1.4e} right = {:1.4e}", phi1, phi0 + c1 * alpha * dphi0);
    trace!(target: "ls", "Satisfies Armijo {}", phi1 <= phi0 + c1 * alpha * dphi0);
    //}}}
    phi1 <= phi0 + c1 * alpha * dphi0
}
//}}}
//{{{ fun:  satisfies_curvature
pub fn satisfies_curvature(
    c2: f64,
    dphi0: f64,
    dphi1: f64,
) -> bool
{
    //{{{ trace
    trace!(target: "ls", "curvature: left = {:1.4e} right = {:1.4e}", dphi1, c2 * dphi0);
    trace!(target: "ls", "Satisfies curvature {}", dphi1 >= c2 * dphi0);
    //}}}
    dphi1 >= c2 * dphi0
}
//}}}
//{{{ fun: satisfies_wolfe
pub fn satisfies_wolfe(
    c1: f64,
    c2: f64,
    phi0: f64,
    dphi0: f64,
    alpha: f64,
    phi1: f64,
    dphi1: f64,
) -> Result<(), Error>
{
    //{{{ trace
    trace!(target: "ls", "phi0 = {:1.4e} dphi0 = {:1.4e} phi1 = {:1.4e} dphi1 = {:1.4e} alpha = {:1.4e}", phi0, dphi0, phi1, dphi1, alpha);
    //}}}
    if !satisfies_armijo(c1, alpha, phi0, dphi0, phi1)
    {
        return Err(Error::Armijo);
    }
    if !satisfies_curvature(c2, dphi0, dphi1)
    {
        return Err(Error::Curvature);
    }
    Ok(())
}
//}}}
pub fn initial_step(
    phi1: f64,
    phi0: f64,
    dphi1: f64,
) -> f64
{
    //{{{ trace
    trace!(target: "ls", "--entering initial_step ---");
    trace!(target: "ls", "phi1 = {phi1:1.4e}, phi0 = {phi0:1.4e} dphi1 = {dphi1:1.4e}");
    //}}}
    let stp1: f64 = 1.0;
    let stp2: f64 = 2.02 * (phi1 - phi0) / dphi1;
    //{{{ trace
    trace!(target: "ls", "stp1 = {stp1} stp2 = {stp2}");
    trace!(target: "ls", "--leaving initial_step ---");
    //}}}
    stp1.min(stp2)
}
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;

    const EPS: f64 = 1e-8;

    fn quad(x: f64) -> f64
    {
        (x - 2.0).powi(2) + 1.0
    }

    fn quad_deriv(x: f64) -> f64
    {
        2.0 * (x - 2.0)
    }

    fn cubic(x: f64) -> f64
    {
        x.powi(3) - 3.0 * x
    }

    fn cubic_deriv(x: f64) -> f64
    {
        3.0 * x * x - 3.0
    }

    // ---------------- quadmin ----------------

    #[test]
    fn quadmin_finds_minimum_for_simple_quadratic()
    {
        let a = 0.0;
        let b = 4.0;
        let phi_a = quad(a);
        let dphi_a = quad_deriv(a);
        let phi_b = quad(b);

        let alpha_min = quadmin(a, phi_a, dphi_a, b, phi_b).expect("should find minimum");
        assert!((alpha_min - 2.0).abs() < EPS);
    }

    #[test]
    fn quadmin_returns_none_for_too_close_points()
    {
        let a = 1.0;
        let b = a + 1e-20; // (b - a)^2 < SMALL
        let phi_a = quad(a);
        let dphi_a = quad_deriv(a);
        let phi_b = quad(b);

        assert!(quadmin(a, phi_a, dphi_a, b, phi_b).is_none());
    }

    // ---------------- cubicmin2 ----------------

    #[test]
    fn cubicmin2_finds_minimum_for_simple_cubic()
    {
        let a = -2.0;
        let b = 2.0;
        let phi_a = cubic(a);
        let phi_b = cubic(b);
        let dphi_a = cubic_deriv(a);
        let dphi_b = cubic_deriv(b);

        let alpha_min = cubicmin2(a, phi_a, dphi_a, b, phi_b, dphi_b).expect("should find minimum");

        // For f(x) = x^3 - 3x, local minimum at x = 1
        assert!((alpha_min - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cubicmin2_returns_none_for_degenerate_interval()
    {
        let a = 1.0;
        let b = 1.0;
        let phi_a = cubic(a);
        let phi_b = cubic(b);
        let dphi_a = cubic_deriv(a);
        let dphi_b = cubic_deriv(b);

        assert!(cubicmin2(a, phi_a, dphi_a, b, phi_b, dphi_b).is_none());
    }

    // ---------------- cubicmin3 ----------------

    #[test]
    fn cubicmin3_finds_minimum_for_simple_cubic()
    {
        // Use same cubic f(x) = x^3 - 3x, minimum at x = 1
        let a = 0.0;
        let b = 1.0;
        let c = 2.0;

        let phi_a = cubic(a);
        let phi_b = cubic(b);
        let phi_c = cubic(c);
        let dphi_a = cubic_deriv(a);

        let alpha_min = cubicmin3(a, phi_a, dphi_a, b, phi_b, c, phi_c)
            .expect("should find minimum for simple cubic");

        assert!((alpha_min - 1.0).abs() < 1e-6);
    }

    #[test]
    fn cubicmin3_returns_none_for_singular_configuration()
    {
        // Make db == dc so denom == 0
        let a = 0.0;
        let b = 1.0;
        let c = 1.0;

        let phi_a = cubic(a);
        let phi_b = cubic(b);
        let phi_c = cubic(c);
        let dphi_a = cubic_deriv(a);

        assert!(cubicmin3(a, phi_a, dphi_a, b, phi_b, c, phi_c).is_none());
    }
}
//}}}
