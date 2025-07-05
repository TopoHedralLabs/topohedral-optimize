//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
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
pub fn quadmin(a: f64, phi_a: f64, dphi_a: f64, b: f64, phi_b: f64) -> Option<f64> {
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

    if db * db < SMALL {
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

    if (2.0 * beta).abs() < SMALL {
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
    return Some(alpha_min);
}
//}}}
//{{{ fun: cubicmin
/// Forms the cubic interpolation to find the minimum of a function.
///
/// Forms the function:
///
/// phi_cu(x) = beta(x - a)^3 + gamma (x - a)^2 + delta (x - a) + epsilon
///
/// And finds the minimum of this function ananlytically.
pub fn cubicmin(
    a: f64,
    phi_a: f64,
    dphi_a: f64,
    b: f64,
    phi_b: f64,
    c: f64,
    phi_c: f64,
) -> Option<f64> {
    //{{{ trace
    error!(target: "ls", "--- Entering cubicmin ---");
    trace!(target: "ls", "Enterin with a = {:1.4e}, b = {:1.4e}, c = {:1.4e}", a, b, c);
    trace!(target: "ls", "phi_a = {:1.4e}, phi_b = {:1.4e}, phi_c = {:1.4e}", phi_a, phi_b, phi_c);
    //}}}
    let db = b - a;
    let dc = c - a;
    let denom = (db * dc).powi(2) * (db - dc);
    //{{{ trace
    trace!(target: "ls", "db = {:1.4e}, dc = {:1.4e}, denom = {:1.4e}", db, dc, denom);
    //}}}

    if denom.abs() < SMALL {
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

    if (3.0 * beta).abs() < SMALL {
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
    return Some(alpha_min);
}
//}}}
//{{{ fun: satisfies_armijo
pub fn satisfies_armijo(c1: f64, alpha: f64, phi0: f64, dphi0: f64, phi1: f64) -> bool {
    //{{{ trace
    trace!(target: "ls", "armijo: left = {:1.4e} right = {:1.4e}", phi1, phi0 + c1 * alpha * dphi0);
    trace!(target: "ls", "Satisfies Armijo {}", phi1 <= phi0 + c1 * alpha * dphi0);
    //}}}
    phi1 <= phi0 + c1 * alpha * dphi0
}
//}}}
//{{{ fun:  satisfies_curvature
pub fn satisfies_curvature(c2: f64, dphi0: f64, dphi1: f64) -> bool {
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
    phi1: f64,
    dphi1: f64,
    alpha: f64,
) -> Result<(), Error> {
    //{{{ trace
    trace!(target: "ls", "phi0 = {:1.4e} dphi0 = {:1.4e} phi1 = {:1.4e} dphi1 = {:1.4e} alpha = {:1.4e}", phi0, dphi0, phi1, dphi1, alpha);
    //}}}
    if !satisfies_armijo(c1, alpha, phi0, dphi0, phi1) {
        return Err(Error::Armijo);
    }
    if !satisfies_curvature(c2, dphi0, dphi1) {
        return Err(Error::Curvature);
    }
    Ok(())
}
//}}}
