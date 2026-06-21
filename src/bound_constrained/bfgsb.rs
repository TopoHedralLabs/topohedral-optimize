//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use std::default;

//{{{ crate imports
use crate::constraints::{BoundStatus, BoundsConstraints, CauchyPathPoint};
use crate::{Matrix, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VecType::Col;
use topohedral_linalg::{MatMul, SubViewable, VectorOps};
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

struct CauchyPoint
{
    pub cauchy_point: Vector,
    pub cauchy_curvature: Vector,
    pub free_variables: Vec<usize>,
}

#[trace_fn]
fn cauchy_point(
    bounds: &BoundsConstraints,
    xk: &Vector,
    gk: &Vector,
    bmatk: &Matrix,
) -> CauchyPoint
{
    let mut dir = -gk.clone();
    let bound_statuses = bounds.bound_statuses(xk, Some(&dir));

    if !bound_statuses.contains(&BoundStatus::Free)
    {
        return CauchyPoint {
            cauchy_point: xk.clone(),
            cauchy_curvature: Vector::zeros_vec(xk.len(), Col),
            free_variables: Vec::new(),
        };
    }

    let mut x_cauchy = Vector::zeros_vec(xk.len(), Col);
    for (vi, bound_status) in bound_statuses.iter().copied().enumerate()
    {
        match bound_status
        {
            BoundStatus::Free =>
            {}
            BoundStatus::AtLower =>
            {
                dir[vi] = 0.0;
                x_cauchy[vi] = bounds.get_lower(vi).unwrap();
            }
            BoundStatus::AtUpper =>
            {
                dir[vi] = 0.0;
                x_cauchy[vi] = bounds.get_upper(vi).unwrap();
            }
        }
    }

    // x - xk starting at x = xk
    let mut x_minus_xk = Vector::zeros_vec(dir.len(), Col);
    let mut bmatk_x_minus_xk = x_minus_xk.clone();
    let mut bmatk_d = bmatk.matmul(&dir);

    let mut dm_dalpha = gk.dot(&dir);
    let mut d2m_dalpha2 = (dir.dot(&bmatk.matmul(&dir))).max(f64::EPSILON);
    let mut dalpha_min = -dm_dalpha / d2m_dalpha2;

    let cauchy_path = bounds.cauchy_path(xk, &dir);
    let mut alpha_old = 0.0;
    let mut final_i;
    for i in 0..cauchy_path.len()
    {
        let CauchyPathPoint {
            alpha: alpha_i,
            variable_index: vi_i,
            bound_status: bs_i,
        } = cauchy_path[i];

        let d_alpha = alpha_i - alpha_old;

        if dalpha_min < d_alpha
        {
            final_i = i;
            break;
        }

        x_cauchy[vi_i] = match bs_i
        {
            BoundStatus::AtLower => bounds.get_lower(vi_i).unwrap(),
            BoundStatus::AtUpper => bounds.get_upper(vi_i).unwrap(),
            _ => panic!("Variable not at bound"),
        };

        let gi = gk[vi_i];
        x_minus_xk += d_alpha * dir.clone();
        bmatk_x_minus_xk += d_alpha * bmatk_d.clone();

        dm_dalpha += d_alpha * d2m_dalpha2 + gi.powi(2) + gi * bmatk_x_minus_xk[vi_i];
        d2m_dalpha2 += 2.0 * gi * bmatk_d[vi_i] + gi * bmatk[(vi_i, vi_i)] * gi;

        dir[vi_i] = 0.0;
        bmatk_d += gi * bmatk.col(vi_i).to_dmatrix();

        dalpha_min = -dm_dalpha / d2m_dalpha2;
        alpha_old = alpha_i;
    }

    dalpha_min = dalpha_min.max(0.0);
    let alpha_total = alpha_old + dalpha_min;

    todo!()
}
