//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::Error;
use crate::bound_constrained::BoundConstrainedOptions;
use crate::common::ConvergedReason;
use crate::constraints::{BoundStatus, BoundsConstraints, CauchyPathPoint};
use crate::line_search::{self as ls, LineSearchError, LineSearchMethod};
use crate::quadratic_model::{QuadraticModel, UpdateType::Direct};
use crate::{IterData, Minimizer, RealFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VecType::Col;
use topohedral_linalg::{MatMul, ReduceOps, SubViewable, SubViewableMut, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

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
        grad_fk: gk,
        hess_k: bmatk,
        ..
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
            BoundStatus::AtLower(value) =>
            {
                dir[vi] = 0.0;
                x_cauchy[vi] = value;
            }
            BoundStatus::AtUpper(value) =>
            {
                dir[vi] = 0.0;
                x_cauchy[vi] = value;
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
    for path_point in &cauchy_path
    {
        let CauchyPathPoint {
            alpha: alpha_i,
            variable_index: vi_i,
            bound_status: bs_i,
        } = *path_point;

        let d_alpha = alpha_i - alpha_old;

        if dalpha_min < d_alpha
        {
            break;
        }

        x_cauchy[vi_i] = match bs_i
        {
            BoundStatus::AtLower(value) =>
            {
                bound_statuses[vi_i] = BoundStatus::AtLower(value);
                value
            }
            BoundStatus::AtUpper(value) =>
            {
                bound_statuses[vi_i] = BoundStatus::AtUpper(value);
                value
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
//{{{ fn: subspace_minimize
#[trace_fn]
fn subspace_minimize(
    bounds: &BoundsConstraints,
    quadratic_model: &QuadraticModel,
    cauchy_point: &CauchyPoint,
) -> Vector
{
    let QuadraticModel {
        fk: _,
        xk,
        grad_fk: gk,
        hess_k: bmatk,
        ..
    } = &quadratic_model;

    let CauchyPoint {
        cauchy_point: x_cauchy,
        cauchy_curvature: bmat_x_minus_xk,
        bound_statuses,
    } = cauchy_point;

    let free_variable_indices: Vec<usize> = bound_statuses
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

    let n = free_variable_indices.len();
    let mut reduced_gradient = Vector::zeros_vec(n, Col);
    for (i, free_idx) in free_variable_indices.iter().enumerate()
    {
        reduced_gradient[i] = -(gk[*free_idx] + bmat_x_minus_xk[*free_idx]);
    }

    let reduced_bmat = bmatk
        .subview_indices(&free_variable_indices, &free_variable_indices)
        .to_dmatrix();

    let Ok(reduced_direction) = reduced_bmat.solve(&reduced_gradient)
    else
    {
        return cauchy_point.cauchy_point.clone();
    };
    let mut full_direction = Vector::zeros_vec(xk.len(), Col);
    full_direction
        .rows_indices_mut(&free_variable_indices)
        .copy_from(&reduced_direction);

    let alpha_min = bounds.max_feasible_step(x_cauchy, &full_direction).min(1.0);
    let mut out: Vector = (x_cauchy + alpha_min * &full_direction).into();
    bounds.clamp(&mut out);
    out
}
//}}}
//{{{ fn: projected_gradient_inf_norm
#[trace_fn]
fn projected_gradient_inf_norm(
    bounds: &BoundsConstraints,
    x: &Vector,
    grad: &Vector,
) -> f64
{
    let negative_grad = -grad.clone();
    bounds
        .projected_direction(x, &negative_grad, 1.0)
        .abs_max()
        .unwrap_or(0.0)
}
//}}}
//{{{ fn: capped_line_search_method
#[trace_fn]
fn capped_line_search_method(
    method: &LineSearchMethod,
    step_max: f64,
) -> LineSearchMethod
{
    let mut method = method.clone();
    if step_max.is_finite()
    {
        match &mut method
        {
            LineSearchMethod::Thuente(opts) =>
            {
                opts.ls_opts.step_max = step_max;
                opts.ls_opts.step_min = opts.ls_opts.step_min.min(step_max);
            }
            LineSearchMethod::Nocedal(opts) =>
            {
                opts.ls_opts.step_max = step_max;
                opts.ls_opts.step_min = opts.ls_opts.step_min.min(step_max);
            }
        }
    }
    method
}
//}}}
//{{{ struct: Options
#[derive(Clone)]
pub struct Options
{
    pub bound_opts: BoundConstrainedOptions,
    pub ls_method: LineSearchMethod,
}
//}}}
//{{{ struct: Bfgsb
pub struct Bfgsb<F: RealFn>
{
    fcn: F,
    bounds: BoundsConstraints,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
    quadratic_model: QuadraticModel,
}
//}}}
//{{{ impl Bfgsb
impl<F: RealFn> Bfgsb<F>
{
    #[trace_fn]
    pub fn new(
        mut fcn: F,
        bounds: BoundsConstraints,
        mut x0: Vector,
        opts: Options,
    ) -> Self
    {
        bounds.clamp(&mut x0);
        let grad_0 = fcn.grad(&x0);
        let projected_grad_0 = projected_gradient_inf_norm(&bounds, &x0, &grad_0);

        let n = x0.len();
        Self {
            fcn,
            bounds,
            x_init: x0,
            norm_grad_fx_init: projected_grad_0,
            opts,
            quadratic_model: QuadraticModel::new(n),
        }
    }

    #[trace_fn]
    fn is_converged(
        &self,
        grad_norm: f64,
    ) -> Option<ConvergedReason>
    {
        let rtol = self.opts.bound_opts.base_opts.grad_rtol;
        if self.norm_grad_fx_init > 0.0 && (grad_norm / self.norm_grad_fx_init) < rtol
        {
            return Some(ConvergedReason::Rtol);
        }
        let atol_converged = grad_norm < self.opts.bound_opts.base_opts.grad_atol;
        if atol_converged
        {
            return Some(ConvergedReason::Atol);
        }
        None
    }
}
//}}}
//{{{ impl Minimizer for Bfgsb
impl<F: RealFn> Minimizer for Bfgsb<F>
{
    type Error = Error;

    #[trace_fn]
    fn minimize(&mut self) -> Result<crate::Returns, Self::Error>
    {
        let mut xk = self.x_init.clone();
        let mut fk = self.fcn.eval(&xk);
        let mut gk = self.fcn.grad(&xk);
        let mut projected_grad_norm = projected_gradient_inf_norm(&self.bounds, &xk, &gk);
        let max_iter = self.opts.bound_opts.base_opts.max_iter;
        let ftol = self.opts.bound_opts.constraint_tol.max(f64::EPSILON);

        for k in 0..max_iter
        {
            if let Some(reason) = self.is_converged(projected_grad_norm)
            {
                return Ok(crate::Returns {
                    fmin: fk,
                    xmin: xk,
                    reason,
                    num_iterations: k as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }

            self.quadratic_model.update_iterate(&xk, fk, &gk);
            let cp = cauchy_point(&self.bounds, &self.quadratic_model);

            let has_free = cp.bound_statuses.contains(&BoundStatus::Free);

            let z = if has_free
            {
                subspace_minimize(&self.bounds, &self.quadratic_model, &cp)
            }
            else
            {
                cp.cauchy_point.clone()
            };

            let mut dir: Vector = (&z - &xk).into();
            let mut gd = gk.dot(&dir);
            let descent_tol = f64::EPSILON * fk.abs().max(1.0);
            if gd >= -descent_tol
            {
                self.quadratic_model.reset();
                let fallback_dir = self.bounds.projected_direction(&xk, &(-gk.clone()), 1.0);
                let fallback_gd = gk.dot(&fallback_dir);
                if fallback_gd < -descent_tol
                {
                    dir = fallback_dir;
                    gd = fallback_gd;
                }
            }

            if gd >= -descent_tol
            {
                if let Some(reason) = self.is_converged(projected_grad_norm)
                {
                    return Ok(crate::Returns {
                        fmin: fk,
                        xmin: xk,
                        reason,
                        num_iterations: k as usize,
                        num_fun_evals: 0,
                        num_grad_evals: 0,
                    });
                }
                return Err(Error::LineSearch(LineSearchError::NoStepFound));
            }

            let max_feasible_alpha = self.bounds.max_feasible_step(&xk, &dir);
            if max_feasible_alpha <= 0.0
            {
                if let Some(reason) = self.is_converged(projected_grad_norm)
                {
                    return Ok(crate::Returns {
                        fmin: fk,
                        xmin: xk,
                        reason,
                        num_iterations: k as usize,
                        num_fun_evals: 0,
                        num_grad_evals: 0,
                    });
                }
                return Err(Error::LineSearch(LineSearchError::NoStepFound));
            }

            let alpha_init = if max_feasible_alpha.is_finite()
            {
                (0.99 * max_feasible_alpha).min(1.0)
            }
            else
            {
                1.0
            }
            .max(1e-30);

            let iter_k = IterData {
                x: xk.clone(),
                fx: fk,
                grad_fx: gk.clone(),
                norm_grad_fx: gk.norm(),
            };

            let search_result = ls::search(
                self.fcn.clone(),
                &iter_k,
                &dir,
                alpha_init,
                capped_line_search_method(&self.opts.ls_method, max_feasible_alpha),
            );

            let Ok(search_result) = search_result
            else
            {
                if !self.quadratic_model.had_first_update
                {
                    return Err(Error::LineSearch(LineSearchError::NoStepFound));
                }
                self.quadratic_model.reset();
                continue;
            };

            let mut xk_new = search_result.x;
            self.bounds.clamp(&mut xk_new);
            let fk_new = self.fcn.eval(&xk_new);
            let gk_new = self.fcn.grad(&xk_new);
            let sk: Vector = (&xk_new - &xk).into();
            let yk: Vector = (&gk_new - &gk).into();
            let rel_red = (fk - fk_new) / fk.abs().max(fk_new.abs()).max(1.0);

            xk = xk_new;
            fk = fk_new;
            gk = gk_new;
            projected_grad_norm = projected_gradient_inf_norm(&self.bounds, &xk, &gk);
            self.quadratic_model.try_update(&sk, &yk, Direct);

            if let Some(reason) = self.is_converged(projected_grad_norm)
            {
                return Ok(crate::Returns {
                    fmin: fk,
                    xmin: xk,
                    reason,
                    num_iterations: (k + 1) as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }

            if rel_red <= ftol
            {
                return Ok(crate::Returns {
                    fmin: fk,
                    xmin: xk,
                    reason: ConvergedReason::Rtol,
                    num_iterations: (k + 1) as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }
        }

        Err(Error::MaxIterations(max_iter as usize))
    }
}
//}}}

//-------------------------------------------------------------------------------------------------
//{{{ mod: tests
#[cfg(test)]
mod tests
{
    use super::*;

    use crate::Matrix;

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

    fn test_quadratic_model(
        fk: f64,
        gk: &Vector,
        bmatk: &Matrix,
        xk: &Vector,
    ) -> QuadraticModel
    {
        let n = xk.len();
        QuadraticModel {
            fk,
            xk: xk.clone(),
            grad_fk: gk.clone(),
            hess_k: bmatk.clone(),
            inv_hess_k: Matrix::identity(n, n),
            had_first_update: false,
        }
    }

    #[test]
    fn cauchy_point_minimizes_model_before_any_bound_is_reached()
    {
        let fk = 1.0;
        let xk = colvec(&[1.0, -1.0]);
        let gk = colvec(&[2.0, -1.0]);
        let bmatk = DMatrix::<f64>::from_row_slice(&[4.0, 0.0, 0.0, 2.0], 2, 2);
        let qm = test_quadratic_model(fk, &gk, &bmatk, &xk);
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

        let qm = test_quadratic_model(fk, &gk, &bmatk, &xk);

        let result = cauchy_point(&bounds, &qm);

        assert_vector_close(&result.cauchy_point, &colvec(&[0.5, 0.75]));
        assert_vector_close(&result.cauchy_curvature, &colvec(&[1.375, 1.0]));
        assert_eq!(
            result.bound_statuses,
            vec![BoundStatus::AtUpper(0.5), BoundStatus::Free]
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
        let qm = test_quadratic_model(fk, &gk, &bmatk, &xk);
        let mut bounds = BoundsConstraints::new(2);
        bounds.add_bounds(0, Some(0.0), Some(2.0));
        bounds.add_bounds(1, Some(-1.0), Some(1.0));

        let result = cauchy_point(&bounds, &qm);

        assert_vector_close(&result.cauchy_point, &xk);
        assert_vector_close(&result.cauchy_curvature, &colvec(&[0.0, 0.0]));
        assert_eq!(
            result.bound_statuses,
            vec![BoundStatus::AtLower(0.0), BoundStatus::AtUpper(1.0)]
        );
        assert_relative_eq!(
            quadratic_model(fk, &gk, &bmatk, &xk, &result.cauchy_point),
            fk,
            epsilon = 1e-12
        );
    }
}
//}}}
