//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, CountingRealFn},
    constrained::{ConstrainedMinimizer, ConstriainedOptions},
    unconstrained::UnconstrainedMethod,
    Matrix, RealFn, RealVectorFn, Vector,
};
//}}}
//{{{ std imports
use std::{
    f64,
    sync::{Arc, Mutex},
};
//}}}
//{{{ dep imports
use topohedral_linalg::{dvector::VecType::Col, ReduceOps, VectorOps};
use topohedral_tracing::{init, trace_fn};
//}}}
//--------------------------------------------------------------------------------------------------
//{{{ struct Options
#[derive(Copy, Clone)]
pub struct Options
{
    constrained_opts: ConstriainedOptions,
    initial_penalty: f64,
    constraint_improvement_factor: f64,
    penalty_growth_factor: f64,
}
//}}}
//{{{ struct: ConstraintData
#[derive(Debug, Clone)]
struct ConstraintData<F: RealVectorFn>
{
    pub function: F,
    pub penalties: Vector,
    pub shifts: Vector,
    pub constraint_values: Vector,
    pub constraint_gradients: Matrix,
}
//}}}
//{{{ impl: ConstraintData
impl<F: RealVectorFn> ConstraintData<F>
{
    pub fn new(
        fcn: F,
        initial_penalty: f64,
    ) -> Self
    {
        let num_constraints = fcn.dimension_range();
        let dimension = fcn.dimension_domain();

        let initial_penalties = Vector::from_value_vec(initial_penalty, num_constraints, Col);
        let zero_vector = Vector::zeros_cvec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: initial_penalties,
            shifts: zero_vector.clone(),
            constraint_values: zero_vector,
            constraint_gradients: zero_matrix,
        }
    }

    fn evaluate_values(
        &mut self,
        x: &Vector,
    )
    {
        self.function.eval(x, &mut self.constraint_values);
    }

    fn evaluate_gradients(
        &mut self,
        x: &Vector,
    )
    {
        self.function.grad(x, &mut self.constraint_gradients);
    }
}
//}}}
//{{{ fun: eq_penalty_value
fn eq_penalty_value<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_range();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let p_i = constaint_data.penalties[i];
        let theta_i = constaint_data.shifts[i];
        let h_i = constaint_data.constraint_values[i];
        constraint_value += p_i * (h_i + theta_i).powi(2);
    }
    constraint_value
}
//}}}
//{{{ fun: eq_penalty_gradient
fn eq_penalty_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let p_i = constaint_data.penalties[i];
        let theta_i = constaint_data.shifts[i];
        let g_i = constaint_data.constraint_values[i];
        let grad_g_i = constaint_data.constraint_gradients.col(i).to_dmatrix();
        constraint_gradient += (p_i * (g_i + theta_i)) * grad_g_i;
    }
    constraint_gradient
}
//}}}
//{{{ fun: ieq_penalty_value
fn ieq_penalty_value<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_domain();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let q_i = constaint_data.penalties[i];
        let phi_i = constaint_data.shifts[i];
        let g_i = constaint_data.constraint_values[i];
        constraint_value += q_i * ((g_i + phi_i).max(0.0)).powi(2);
    }
    constraint_value
}
//}}}
//{{{ fun: ieq_penalty_gradient
fn ieq_penalty_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let q_i = constaint_data.penalties[i];
        let phi_i = constaint_data.shifts[i];
        let g_i = constaint_data.constraint_values[i];
        let grad_g_i = constaint_data.constraint_gradients.col(i).to_dmatrix();
        constraint_gradient += (q_i * (g_i + phi_i).max(0.0)) * grad_g_i;
    }
    constraint_gradient
}
//}}}
//{{{ struct: AugmentedLagrangianFcn
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: F1,
    eq_constraint_data: Option<ConstraintData<F2>>,
    ieq_constraint_data: Option<ConstraintData<F3>>,
}
//}}}
//{{{ impl: AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangianFcn<F1, F2, F3>
{
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        initial_penalty: f64,
    ) -> Self
    {
        let eq_constraint_data = match eq_constraints
        {
            Some(eq_con) => Some(ConstraintData::new(eq_con, initial_penalty)),
            None => None,
        };

        let ieq_constraint_data = match ieq_constraints
        {
            Some(ieq_con) => Some(ConstraintData::new(ieq_con, initial_penalty)),
            None => None,
        };

        Self {
            fcn: fcn,
            eq_constraint_data: eq_constraint_data,
            ieq_constraint_data: ieq_constraint_data,
        }
    }

    fn max_constraint_violation(&self) -> f64
    {
        let mut max_violation = 0.0;

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            let max_eq_violation = eq_constraint_data.constraint_values.abs_max().unwrap();
            max_violation = f64::max(max_violation, max_eq_violation);
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            let values = &ieq_constraint_data.constraint_values;
            let shifts = &ieq_constraint_data.shifts;
            let max_ieq_violation = values
                .iter()
                .zip(shifts.iter())
                .map(|(&gi, &phi_i)| f64::min(-gi, phi_i))
                .reduce(f64::max)
                .unwrap();

            max_violation = f64::max(max_violation, max_ieq_violation);
        }
        max_violation
    }

    fn unimproved_constraint_violations(
        &self,
        new_kmax: f64,
    ) -> (Vec<usize>, Vec<usize>)
    {
        let mut improved_eq_constraints = Vec::<usize>::new();
        let mut improved_ieq_constraints = Vec::<usize>::new();

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            let neq = eq_constraint_data.constraint_values.len();
            improved_eq_constraints.reserve(neq);
            improved_eq_constraints.extend(
                eq_constraint_data
                    .constraint_values
                    .iter()
                    .enumerate()
                    .filter_map(|(i, h_i)| (h_i.abs() > new_kmax).then_some(i)),
            );
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            let nieq = ieq_constraint_data.constraint_values.len();
            let values = &ieq_constraint_data.constraint_values;
            let shifts = &ieq_constraint_data.shifts;
            improved_ieq_constraints.reserve(nieq);
            improved_ieq_constraints.extend(
                values
                    .iter()
                    .zip(shifts)
                    .enumerate()
                    .filter_map(|(i, (g_i, phi_i))| ((-g_i).min(*phi_i) > new_kmax).then_some(i)),
            )
        }
        (improved_eq_constraints, improved_ieq_constraints)
    }
}
//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2, F3>
{
    fn dimension(&self) -> usize
    {
        self.fcn.dimension()
    }

    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let mut aug_lag = self.fcn.eval(x);

        if let Some(eq_constraint_data) = &mut self.eq_constraint_data
        {
            eq_constraint_data.evaluate_values(x);
            aug_lag += eq_penalty_value(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_constraint_data
        {
            ieq_constraint_data.evaluate_values(x);
            aug_lag += ieq_penalty_value(ieq_constraint_data);
        }

        aug_lag
    }

    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let mut grad_aug_lag = self.fcn.grad(x);

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            grad_aug_lag += eq_penalty_gradient(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            grad_aug_lag += ieq_penalty_gradient(ieq_constraint_data);
        }

        grad_aug_lag
    }
}
//}}}
//{{{ struct: AugmentedLagrangian
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    fcn_grad_init: Vector,
    opts: Options,
}
//}}}
//{{{ impl: AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangian<F1, F2, F3>
{
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        x0: Vector,
        opts: Options,
    ) -> Self
    {
        let fcn_shared = arc_real_fn(CountingRealFn::new(AugmentedLagrangianFcn::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            opts.initial_penalty,
        )));
        let n = fcn_shared.dimension();

        Self {
            fcn: fcn_shared,
            x_init: x0,
            fcn_grad_init: Vector::zeros_cvec(n, Col),
            opts: opts,
        }
    }
}
//}}}
//{{{ impl: ConstrainedMinimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> ConstrainedMinimizer
    for AugmentedLagrangian<F1, F2, F3>
{
    fn minimize(&mut self) -> Result<super::common::Returns, super::common::Error>
    {
        let mut Kmax = f64::INFINITY;

        todo!()
    }
}
//}}}
