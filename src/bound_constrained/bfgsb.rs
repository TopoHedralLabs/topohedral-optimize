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
use topohedral_linalg::{MatMul, VectorOps};
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
    let mut dk = -gk.clone();
    let (active_set, inactive_set) = bounds.active_and_inactive_sets(xk, Some(&dk));

    if inactive_set.is_empty()
    {
        return CauchyPoint {
            cauchy_point: xk.clone(),
            cauchy_curvature: Vector::zeros_vec(xk.len(), Col),
            free_variables: Vec::new(),
        };
    }

    let mut x_cauchy = Vector::zeros_vec(xk.len(), Col);
    for (vi, bound_status) in active_set
    {
        dk[vi] = 0.0;
        match bound_status
        {
            BoundStatus::AtLower =>
            {
                x_cauchy[vi] = bounds.get_lower(vi).unwrap();
            }
            BoundStatus::AtUpper =>
            {
                x_cauchy[vi] = bounds.get_upper(vi).unwrap();
            }
            _ =>
            {
                panic!("Unexpected")
            }
        }
    }

    let z = Vector::zeros_vec(dk.len(), Col);
    let bmatk_z = z.clone();
    let bmatk_d = bmatk.matmul(&dk);

    let mut df_dt = gk.dot(&dk);
    let mut d2f_dt2 = (dk.dot(&bmatk.matmul(&dk))).max(f64::EPSILON);
    let mut dalpha_min = -df_dt / d2f_dt2;

    let cauchy_path = bounds.cauchy_path(xk, &dk);
    let mut alpha_prev = 0.0;
    let mut dir_cur = &dk;

    for i in 1..cauchy_path.len()
    {
        let CauchyPathPoint {
            alpha: alphai,
            point: xi,
            direction: di,
            variable_index: vi,
        } = &cauchy_path[i];

        let d_alpha = alphai - alpha_prev;

        if dalpha_min < d_alpha
        {
            x_cauchy = (xi + d_alpha * di).into();
            break;
        }
        alpha_prev = *alphai;
    }

    todo!()
}
