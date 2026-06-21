//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

use crate::bound_constrained::BoundConstrainedOptions;
use crate::constraints::BoundStatus::{AtLower, AtUpper};
//{{{ crate imports
use crate::constraints::{BoundStatus, BoundsConstraints, CauchyPathPoint};
use crate::line_search::LineSearchMethod;
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

struct QuadraticModel
{
    fk: f64,
    xk: Vector,
    gk: Vector,
    bmatk: Matrix,
}

impl QuadraticModel {}

//{{{ struct: CauchyPoint
struct CauchyPoint
{
    pub cauchy_point: Vector,
    pub cauchy_curvature: Vector,
    pub bound_statuses: Vec<BoundStatus>,
}
//}}}

//{{{ fn: cauchy_point
#[trace_fn]
fn cauchy_point(
    bounds: &BoundsConstraints,
    quadratic_model: &QuadraticModel,
) -> CauchyPoint
{
    let QuadraticModel {
        fk: _,
        xk,
        gk,
        bmatk,
    } = &quadratic_model;

    let mut dir = -gk.clone();
    let mut bound_statuses = bounds.bound_statuses(xk, Some(&dir));

    if !bound_statuses.contains(&BoundStatus::Free)
    {
        return CauchyPoint {
            cauchy_point: xk.clone(),
            cauchy_curvature: Vector::zeros_vec(xk.len(), Col),
            bound_statuses,
        };
    }

    let mut x_cauchy = xk.clone();
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

    let n = dir.len();
    let mut bmatk_x_minus_xk = Vector::zeros_vec(n, Col);
    let mut bmatk_d = bmatk.matmul(&dir);

    let mut dm_dalpha = gk.dot(&dir);
    let mut d2m_dalpha2 = dir.dot(&bmatk_d).max(f64::EPSILON);
    let mut dalpha_min = -dm_dalpha / d2m_dalpha2;

    let cauchy_path = bounds.cauchy_path(xk, &dir);
    let mut alpha_old = 0.0;
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
            break;
        }

        x_cauchy[vi_i] = match bs_i
        {
            BoundStatus::AtLower =>
            {
                bound_statuses[vi_i] = AtLower;
                bounds.get_lower(vi_i).unwrap()
            }
            BoundStatus::AtUpper =>
            {
                bound_statuses[vi_i] = AtUpper;
                bounds.get_upper(vi_i).unwrap()
            }
            _ => panic!("Variable not at bound"),
        };
        let gi = gk[vi_i];
        bmatk_x_minus_xk += d_alpha * bmatk_d.clone();

        dm_dalpha += d_alpha * d2m_dalpha2 + gi.powi(2) + gi * bmatk_x_minus_xk[vi_i];
        d2m_dalpha2 += 2.0 * gi * bmatk_d[vi_i] + gi * bmatk[(vi_i, vi_i)] * gi;
        d2m_dalpha2 = d2m_dalpha2.max(f64::EPSILON);

        dir[vi_i] = 0.0;
        bmatk_d += gi * bmatk.col(vi_i).to_dmatrix();

        dalpha_min = -dm_dalpha / d2m_dalpha2;
        alpha_old = alpha_i;
    }

    dalpha_min = dalpha_min.max(0.0);
    let alpha_total = alpha_old + dalpha_min;
    for vi in 0..n
    {
        if bound_statuses[vi] == BoundStatus::Free
        {
            x_cauchy[vi] = xk[vi] + alpha_total * dir[vi];
        }
    }
    bmatk_x_minus_xk += dalpha_min * bmatk_d;

    CauchyPoint {
        cauchy_point: x_cauchy,
        cauchy_curvature: bmatk_x_minus_xk,
        bound_statuses,
    }
}
//}}}

fn subspace_minimize(
    bounds: &BoundsConstraints,
    quadratic_model: &QuadraticModel,
    cauchy_point: &CauchyPoint,
) -> Vector
{
    let free_variable_indices: Vec<usize> = cauchy_point
        .bound_statuses
        .iter()
        .enumerate()
        .filter_map(|(variable_index, status)| {
            (*status == BoundStatus::Free).then_some(variable_index)
        })
        .collect();

    if free_variable_indices.is_empty()
    {
        return cauchy_point.cauchy_point.clone();
    }

    // let reduced_gradient =

    let out: Vector = cauchy_point.cauchy_point.clone();

    out
}

struct Options
{
    pub bound_opts: BoundConstrainedOptions,
    pub ls_method: LineSearchMethod,
}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;

    use approx::assert_relative_eq;
    use topohedral_linalg::{DMatrix, DVector, VecType};

    fn colvec(values: &[f64]) -> Vector
    {
        DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
    }

    fn quadratic_model(
        fk: f64,
        gk: &Vector,
        bmatk: &Matrix,
        xk: &Vector,
        x: &Vector,
    ) -> f64
    {
        let step = x.clone() - xk.clone();
        let bmatk_step = bmatk.matmul(&step);
        fk + gk.dot(&step) + 0.5 * step.dot(&bmatk_step)
    }

    fn assert_vector_close(
        actual: &Vector,
        expected: &Vector,
    )
    {
        assert_eq!(actual.len(), expected.len());
        for (actual_i, expected_i) in actual.iter().zip(expected.iter())
        {
            assert_relative_eq!(*actual_i, *expected_i, epsilon = 1e-12);
        }
    }

    #[test]
    fn cauchy_point_minimizes_model_before_any_bound_is_reached()
    {
        let fk = 1.0;
        let xk = colvec(&[1.0, -1.0]);
        let gk = colvec(&[2.0, -1.0]);
        let bmatk = DMatrix::<f64>::from_row_slice(&[4.0, 0.0, 0.0, 2.0], 2, 2);
        let qm = QuadraticModel {
            fk,
            xk: xk.clone(),
            gk: gk.clone(),
            bmatk: bmatk.clone(),
        };
        let mut bounds = BoundsConstraints::new(2);
        bounds.add_bounds(0, Some(-10.0), Some(10.0));
        bounds.add_bounds(1, Some(-10.0), Some(10.0));

        let result = cauchy_point(&bounds, &qm);
        let expected_alpha = 5.0 / 18.0;
        let expected = colvec(&[
            xk[0] - expected_alpha * gk[0],
            xk[1] - expected_alpha * gk[1],
        ]);

        assert_vector_close(&result.cauchy_point, &expected);
        assert_vector_close(&result.cauchy_curvature, &colvec(&[-20.0 / 9.0, 5.0 / 9.0]));
        assert_eq!(
            result.bound_statuses,
            vec![BoundStatus::Free, BoundStatus::Free]
        );

        let before: Vector = (&xk - (expected_alpha - 0.01) * &gk).into();
        let after: Vector = (&xk - (expected_alpha + 0.01) * &gk).into();
        let model_at_cauchy = quadratic_model(fk, &gk, &bmatk, &xk, &result.cauchy_point);
        assert!(model_at_cauchy < quadratic_model(fk, &gk, &bmatk, &xk, &before));
        assert!(model_at_cauchy < quadratic_model(fk, &gk, &bmatk, &xk, &after));
    }

    #[test]
    fn cauchy_point_continues_on_new_face_after_hitting_a_bound()
    {
        let fk = 3.0;
        let xk = colvec(&[0.0, 0.0]);
        let gk = colvec(&[-2.0, -1.0]);
        let bmatk = DMatrix::<f64>::from_row_slice(&[2.0, 0.5, 0.5, 1.0], 2, 2);
        let mut bounds = BoundsConstraints::new(2);
        bounds.add_bounds(0, Some(-1.0), Some(0.5));
        bounds.add_bounds(1, Some(-1.0), Some(2.0));

        let qm = QuadraticModel {
            fk,
            xk: xk.clone(),
            gk: gk.clone(),
            bmatk: bmatk.clone(),
        };

        let result = cauchy_point(&bounds, &qm);

        assert_vector_close(&result.cauchy_point, &colvec(&[0.5, 0.75]));
        assert_vector_close(&result.cauchy_curvature, &colvec(&[1.375, 1.0]));
        assert_eq!(
            result.bound_statuses,
            vec![BoundStatus::AtUpper, BoundStatus::Free]
        );

        let before = colvec(&[0.5, 0.74]);
        let after = colvec(&[0.5, 0.76]);
        let model_at_cauchy = quadratic_model(fk, &gk, &bmatk, &xk, &result.cauchy_point);
        assert!(model_at_cauchy < quadratic_model(fk, &gk, &bmatk, &xk, &before));
        assert!(model_at_cauchy < quadratic_model(fk, &gk, &bmatk, &xk, &after));
    }

    #[test]
    fn cauchy_point_stays_at_iterate_when_all_descent_components_are_pinned()
    {
        let fk = 2.0;
        let xk = colvec(&[0.0, 1.0]);
        let gk = colvec(&[1.0, -2.0]);
        let bmatk = DMatrix::<f64>::from_row_slice(&[2.0, 0.25, 0.25, 3.0], 2, 2);
        let qm = QuadraticModel {
            fk,
            xk: xk.clone(),
            gk: gk.clone(),
            bmatk: bmatk.clone(),
        };
        let mut bounds = BoundsConstraints::new(2);
        bounds.add_bounds(0, Some(0.0), Some(2.0));
        bounds.add_bounds(1, Some(-1.0), Some(1.0));

        let result = cauchy_point(&bounds, &qm);

        assert_vector_close(&result.cauchy_point, &xk);
        assert_vector_close(&result.cauchy_curvature, &colvec(&[0.0, 0.0]));
        assert_eq!(
            result.bound_statuses,
            vec![BoundStatus::AtLower, BoundStatus::AtUpper]
        );
        assert_relative_eq!(
            quadratic_model(fk, &gk, &bmatk, &xk, &result.cauchy_point),
            fk,
            epsilon = 1e-12
        );
    }
}
//}}}
