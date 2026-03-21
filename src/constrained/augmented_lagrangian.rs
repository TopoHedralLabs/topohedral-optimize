//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::CountingRealFn,
    constrained::common::{ConstrainedMinimizer, Options},
    Matrix, RealFn, RealVectorFn, Vector,
};
//}}}
//{{{ std imports
use std::sync::{Arc, Mutex};
//}}}
//{{{ dep imports
use topohedral_linalg::dvector::VecType::Col;
//}}}
//--------------------------------------------------------------------------------------------------

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
    pub fn new(fcn: F) -> Self
    {
        let num_constraints = fcn.dimension_range();
        let dimension = fcn.dimension_domain();
        let zero_vector = Vector::zeros_cvec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: zero_vector.clone(),
            shifts: zero_vector.clone(),
            constraint_values: zero_vector.clone(),
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
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn>
{
    fcn: F1,
    eq_constraint_data: Option<ConstraintData<F2>>,
    ieq_constraint_data: Option<ConstraintData<F2>>,
}
//}}}
//{{{ impl: AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn> AugmentedLagrangianFcn<F1, F2>
{
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F2>,
    ) -> Self
    {
        let eq_constraint_data = match eq_constraints
        {
            Some(eq_con) => Some(ConstraintData::new(eq_con)),
            None => None,
        };

        let ieq_constraint_data = match ieq_constraints
        {
            Some(ieq_con) => Some(ConstraintData::new(ieq_con)),
            None => None,
        };

        Self {
            fcn: fcn,
            eq_constraint_data: eq_constraint_data,
            ieq_constraint_data: ieq_constraint_data,
        }
    }
}
//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2>
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
//{{{ struct: AugmentedLagrangianData
struct AugmentedLagrangianData
{
    max_constraint_violation: f64,
}
//}}}
//{{{ struct: AugmentedLagrangian
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2>>>>,
    x_init: Vector,
    fcn_grad_init: Vector,
    opts: Options,
}
//}}}
//{{{ impl AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn> AugmentedLagrangian<F1, F2> {}
//}}}
//{{{ impl: ConstrainedMinimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn> ConstrainedMinimizer for AugmentedLagrangian<F1, F2>
{
    fn minimize(&mut self) -> Result<super::common::Returns, super::common::Error>
    {
        todo!()
    }
}
//}}}
