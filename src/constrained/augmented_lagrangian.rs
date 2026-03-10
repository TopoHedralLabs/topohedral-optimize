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

#[derive(Debug, Clone)]
struct ConstraintData<F: RealVectorFn>
{
    pub function: F,
    pub penalties: Vector,
    pub multipliers: Vector,
    pub constraint_values: Vector,
    pub constraint_gradients: Matrix,
}

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
            multipliers: zero_vector.clone(),
            constraint_values: zero_vector.clone(),
            constraint_gradients: zero_matrix,
        }
    }

    fn evaluate_values(
        &mut self,
        x: &Vector,
    )
    {
        let ConstraintData {
            function,
            penalties: _,
            multipliers: _,
            constraint_values,
            constraint_gradients: _,
        } = self;

        function.eval(x, constraint_values);
    }

    fn evaluate_gradients(
        &mut self,
        x: &Vector,
    )
    {
        let ConstraintData {
            function,
            penalties: _,
            multipliers: _,
            constraint_values: _,
            constraint_gradients,
        } = self;

        function.grad(x, constraint_gradients);
    }
}

fn evaluate_eq_constraint<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_range();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let rho_i = constaint_data.penalties[i];
        let lambda_i = constaint_data.multipliers[i];
        let g_i = constaint_data.constraint_values[i];
        constraint_value += rho_i * (g_i + (lambda_i / rho_i)).powi(2);
    }
    constraint_value
}

fn evaluate_eq_constraint_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let rho_i = constaint_data.penalties[i];
        let lambda_i = constaint_data.multipliers[i];
        let g_i = constaint_data.constraint_values[i];
        let grad_g_i = constaint_data.constraint_gradients.col(i).to_dmatrix();
        constraint_gradient += (rho_i * (g_i + (lambda_i / rho_i))) * grad_g_i;
    }
    constraint_gradient
}

fn evaluate_ieq_constraint<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> f64
{
    let n = constaint_data.function.dimension_domain();
    let mut constraint_value = 0.0;
    for i in 0..n
    {
        let rho_i = constaint_data.penalties[i];
        let lambda_i = constaint_data.multipliers[i];
        let g_i = constaint_data.constraint_values[i];
        constraint_value += rho_i * (g_i + (lambda_i / rho_i).max(0.0)).powi(2);
    }
    constraint_value
}

fn evaluate_ieq_constraint_gradient<F: RealVectorFn>(constaint_data: &ConstraintData<F>) -> Vector
{
    let dim = constaint_data.function.dimension_domain();
    let n = constaint_data.function.dimension_range();
    let mut constraint_gradient = Vector::zeros_cvec(dim, Col);

    for i in 0..n
    {
        let rho_i = constaint_data.penalties[i];
        let lambda_i = constaint_data.multipliers[i];
        let g_i = constaint_data.constraint_values[i];
        let grad_g_i = constaint_data.constraint_gradients.col(i).to_dmatrix();
        constraint_gradient += (rho_i * (g_i + (lambda_i / rho_i)).max(0.0)) * grad_g_i;
    }
    constraint_gradient
}

#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn>
{
    fcn: F1,
    eq_constraint_data: Option<ConstraintData<F2>>,
    ieq_constraint_data: Option<ConstraintData<F2>>,
}

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

    pub fn evaluate_constraint_values(
        &mut self,
        x: &Vector,
    )
    {
        if let Some(eq_constraints) = &mut self.eq_constraint_data
        {
            eq_constraints.evaluate_values(x);
        }

        if let Some(ieq_constraints) = &mut self.ieq_constraint_data
        {
            ieq_constraints.evaluate_values(x);
        }
    }

    pub fn evaluate_constraint_gradients(
        &mut self,
        x: &Vector,
    )
    {
        if let Some(eq_constraints) = &mut self.eq_constraint_data
        {
            eq_constraints.evaluate_gradients(x);
        }

        if let Some(ieq_constraints) = &mut self.ieq_constraint_data
        {
            ieq_constraints.evaluate_gradients(x);
        }
    }
}

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

        if let Some(eq_constraint_data) = &self.eq_constraint_data
        {
            aug_lag += evaluate_eq_constraint(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            aug_lag += evaluate_ieq_constraint(ieq_constraint_data);
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
            grad_aug_lag += evaluate_eq_constraint_gradient(eq_constraint_data);
        }

        if let Some(ieq_constraint_data) = &self.ieq_constraint_data
        {
            grad_aug_lag += evaluate_ieq_constraint_gradient(ieq_constraint_data);
        }

        grad_aug_lag
    }
}

pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2>>>>,
    x_init: Vector,
    fcn_grad_init: Vector,
    opts: Options,
}
