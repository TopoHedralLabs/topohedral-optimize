//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    common::{arc_real_fn, ConvergedReason, CountingRealFn, IterData, Returns},
    constrained::{ConstrainedError, ConstrainedMinimizer, ConstriainedOptions},
    unconstrained::{minimize, UnconstrainedMethod, UnconstrainedReturns},
    Matrix, RealFn, RealVectorFn, Vector,
};
use core::f64;
//}}}
//{{{ std imports
use std::{
    collections::HashMap,
    fmt::{self, Display, Formatter},
    iter,
    sync::{Arc, Mutex},
};
//}}}
//{{{ dep imports
use topohedral_linalg::{
    dvector::VecType::Col, FloatTransformOps, MatMul, MatrixOps, ReduceOps, Shape, TransformOps,
    VectorOps,
};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: LagrangianType
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LagrangianType
{
    AugmentedLagrangian,
    Lagrangian,
}
//}}}
//{{{ struct Options
#[derive(Clone)]
pub struct Options
{
    pub constrained_opts: ConstriainedOptions,
    pub uncon_method: UnconstrainedMethod,
    pub initial_penalty: f64,
    pub constraint_improvement_factor: f64,
    pub penalty_growth_factor: f64,
}
//}}}
//{{{ impl: Options
impl Options
{
    pub fn new(
        constrained_opts: ConstriainedOptions,
        uncon_method: UnconstrainedMethod,
        initial_penalty: f64,
        constraint_improvement_factor: f64,
        penalty_growth_factor: f64,
    ) -> Self
    {
        Self {
            constrained_opts,
            uncon_method,
            initial_penalty,
            constraint_improvement_factor,
            penalty_growth_factor,
        }
    }

    pub(crate) fn uncon_method_mut(&mut self) -> &mut UnconstrainedMethod
    {
        &mut self.uncon_method
    }

    pub(crate) fn uncon_method(&self) -> &UnconstrainedMethod
    {
        &self.uncon_method
    }
}
//}}}

//{{{ struct: ConstraintData
#[derive(Debug, Clone)]
struct LagrangianPenaltyData<F: RealVectorFn>
{
    pub function: F,
    pub penalties: Vector,
    pub shifts: Vector,
    pub values: Vector,
    pub gradients: Matrix,
}
//}}}
//{{{ impl: ConstraintData
impl<F: RealVectorFn> LagrangianPenaltyData<F>
{
    //{{{ fn: new
    pub fn new(
        fcn: F,
        initial_penalty: f64,
    ) -> Self
    {
        let num_constraints = fcn.dimension_range();
        let dimension = fcn.dimension_domain();
        //{{{ trace
        trace!(target: "aug", "Creating constraint data object with {num_constraints} constraints");
        //}}}

        let initial_penalties = Vector::from_value_vec(initial_penalty, num_constraints, Col);
        let zero_vector = Vector::zeros_cvec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: initial_penalties,
            shifts: zero_vector.clone(),
            values: zero_vector,
            gradients: zero_matrix,
        }
    }
    //}}}
    //{{{ fn: update_values
    fn update_values(
        &mut self,
        x: &Vector,
        is_ieq: bool,
    )
    {
        self.function.eval(x, &mut self.values);
        if is_ieq
        {
            self.values.transform(|fx| f64::max(0.0, fx));
        }
    }
    //}}}
    //{{{ fn: update_gradients
    fn update_gradients(
        &mut self,
        x: &Vector,
        is_ieq: bool,
    )
    {
        self.function.grad(x, &mut self.gradients);
        if is_ieq
        {
            for constraint_index in 0..self.gradients.ncols()
            {
                let gi = self.values[constraint_index];
                if gi >= 0.0
                {
                    for j in 0..self.gradients.nrows()
                    {
                        self.gradients[(j, constraint_index)] = 0.0;
                    }
                }
            }
        }
    }
    //}}}
}
//}}}

//{{{ struct: EqPenalty
#[derive(Debug, Clone)]
struct EqPenalty<F: RealVectorFn>
{
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl EqPenalty
impl<F: RealVectorFn> EqPenalty<F>
{
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn for EqPenalty
impl<F: RealVectorFn> RealFn for EqPenalty<F>
{
    //{{{ fn: dimension
    fn dimension(&self) -> usize
    {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.data.update_values(x, false);
        let constraint_value = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                //{{{ trace
                trace!(target: "aug", "Computing augmented lagrangian");
                //}}}

                let mut shifted_h: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_h.transform(|hi| hi.powi(2));
                0.5 * self.data.penalties.dot(&shifted_h)
            }
            LagrangianType::Lagrangian =>
            {
                //{{{ trace
                trace!(target: "aug", "Computing classical lagrangian");
                //}}}
                let lambda: Vector = (&self.data.penalties * &self.data.shifts).into();
                lambda.dot(&self.data.values)
            }
        };
        //{{{ trace
        trace!(target: "aug", "Evaluated equality constraint value: {constraint_value:1.4e}");
        //}}}
        constraint_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.data.update_gradients(x, false);
        let weighted_constraint_values: Vector = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                (&self.data.penalties * (&self.data.values + &self.data.shifts)).into()
            }
            LagrangianType::Lagrangian => (&self.data.penalties * &self.data.shifts).into(),
        };
        let constraint_gradient = self.data.gradients.matmul(&weighted_constraint_values);
        constraint_gradient
    }
    //}}}
}
//}}}

//{{{ struct: IeqPenalty
#[derive(Debug, Clone)]
struct IeqPenalty<F: RealVectorFn>
{
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl IeqPenalty
impl<F: RealVectorFn> IeqPenalty<F>
{
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn  for IeqPenalty
impl<F: RealVectorFn> RealFn for IeqPenalty<F>
{
    //{{{ fn: dimension
    fn dimension(&self) -> usize
    {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        self.data.update_values(x, true);
        let constraint_value = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                let mut shifted_g: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_g.transform(|value| {
                    if value > 0.0
                    {
                        value.powi(2)
                    }
                    else
                    {
                        0.0
                    }
                });
                0.5 * self.data.penalties.dot(&shifted_g)
            }
            LagrangianType::Lagrangian =>
            {
                let mu: Vector = (&self.data.penalties * &self.data.shifts).into();
                mu.dot(&self.data.values)
            }
        };
        constraint_value
    }
    //}}}
    //{{{ fn: grad
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        self.data.update_gradients(x, true);
        let weighted_constraint_values: Vector = match self.lagrangian_type
        {
            LagrangianType::AugmentedLagrangian =>
            {
                let mut shifted_values: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_values.transform(|value| {
                    if value > 0.0
                    {
                        value
                    }
                    else
                    {
                        0.0
                    }
                });
                (&self.data.penalties * &shifted_values).into()
            }
            LagrangianType::Lagrangian => (&self.data.penalties * &self.data.shifts).into(),
        };
        let constraint_gradient = self.data.gradients.matmul(&weighted_constraint_values);
        constraint_gradient
    }
    //}}}
}
//}}}

//{{{ struct CachedValues
#[derive(Debug, Clone)]
struct CachedValues
{
    fcn_value: f64,
    fcn_grad: Vector,
    eq_constraint_value: f64,
    eq_constriant_grad: Vector,
    ieq_constraint_value: f64,
    ieq_constraint_grad: Vector,
}
//}}}
//{{{ impl: CachedValues
impl CachedValues
{
    //{{{ fn: new
    #[trace_fn]
    fn new(n: usize) -> Self
    {
        let zero_vector = Vector::zeros_cvec(n, Col);
        Self {
            fcn_value: 0.0,
            fcn_grad: zero_vector.clone(),
            eq_constraint_value: 0.0,
            eq_constriant_grad: zero_vector.clone(),
            ieq_constraint_value: 0.0,
            ieq_constraint_grad: zero_vector.clone(),
        }
    }
    //}}}
}
//}}}

//{{{ struct: AugmentedLagrangianFcn
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: F1,
    eq_penalty: Option<EqPenalty<F2>>,
    ieq_penalty: Option<IeqPenalty<F3>>,
    lagrangian_type: LagrangianType,
    cached_values: HashMap<LagrangianType, CachedValues>,
}
//}}}
//{{{ impl: AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangianFcn<F1, F2, F3>
{
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self
    {
        let eq_penalty = match eq_constraints
        {
            Some(eq_con) => Some(EqPenalty::new(eq_con, initial_penalty, lagrangian_type)),
            None => None,
        };

        let ieq_penalty = match ieq_constraints
        {
            Some(ieq_con) => Some(IeqPenalty::new(ieq_con, initial_penalty, lagrangian_type)),
            None => None,
        };

        let n = fcn.dimension();
        let mut cached_values = HashMap::<LagrangianType, CachedValues>::new();
        cached_values.insert(LagrangianType::AugmentedLagrangian, CachedValues::new(n));
        cached_values.insert(LagrangianType::Lagrangian, CachedValues::new(n));

        Self {
            fcn,
            eq_penalty,
            ieq_penalty,
            lagrangian_type,
            cached_values,
        }
    }
    //}}}
    //{{{ fn: set_lagrangian_type
    #[trace_fn]
    fn set_lagrangian_type(
        &mut self,
        lagrangian_type: LagrangianType,
    )
    {
        if let Some(eq_penalty) = &mut self.eq_penalty
        {
            eq_penalty.lagrangian_type = lagrangian_type
        }

        if let Some(ieq_penalty) = &mut self.ieq_penalty
        {
            ieq_penalty.lagrangian_type = lagrangian_type
        }
    }

    //}}}
}

//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2, F3>
{
    //{{{ fn: dimension
    fn dimension(&self) -> usize
    {
        self.fcn.dimension()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64
    {
        let cached_value = self.cached_values.get_mut(&self.lagrangian_type).unwrap();
        let fcn_value = self.fcn.eval(x);
        let mut eq_value = 0.0;
        let mut ieq_value = 0.0;

        cached_value.fcn_value = fcn_value;

        if let Some(eq_constraint_data) = &mut self.eq_penalty
        {
            eq_value = eq_constraint_data.eval(x);
            cached_value.eq_constraint_value = eq_value;
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_penalty
        {
            ieq_value = ieq_constraint_data.eval(x);
            cached_value.ieq_constraint_value = ieq_value;
        }

        let aug_lag_value = fcn_value + eq_value + ieq_value;
        aug_lag_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector
    {
        let n = self.dimension();
        let cached_value = self.cached_values.get_mut(&self.lagrangian_type).unwrap();

        let fcn_grad = self.fcn.grad(x);
        cached_value.fcn_grad = fcn_grad.clone();

        let mut eq_penalty_grad = Vector::zeros_cvec(n, Col);
        let mut ieq_penalty_grad = Vector::zeros_cvec(n, Col);

        if let Some(eq_constraint_data) = &mut self.eq_penalty
        {
            eq_penalty_grad = eq_constraint_data.grad(x);
            cached_value.eq_constriant_grad = eq_penalty_grad.clone();
        }

        if let Some(ieq_constraint_data) = &mut self.ieq_penalty
        {
            ieq_penalty_grad = ieq_constraint_data.grad(x);
            cached_value.ieq_constraint_grad = ieq_penalty_grad.clone();
        }

        let grad_aug_lag = (&fcn_grad + &eq_penalty_grad + &ieq_penalty_grad).into();
        return grad_aug_lag;
    }
    //}}}
}
//}}}

//{{{ struct: AugmentedLagrangian
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn>
{
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    norm_grad_fx_init: f64,
    opts: Options,
}
//}}}
//{{{ impl: AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangian<F1, F2, F3>
{
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        x0: Vector,
        opts: Options,
    ) -> Self
    {
        assert!(!opts.uncon_method.uncon_opts().make_counting);

        //{{{ trace
        trace!("Creating new Augmented Lagrangian function");
        //}}}
        let mut fcn_shared = arc_real_fn(CountingRealFn::new(AugmentedLagrangianFcn::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            opts.initial_penalty,
            LagrangianType::AugmentedLagrangian,
        )));

        let _ = fcn_shared.eval(&x0);
        let norm_grad_f0 = fcn_shared.grad(&x0).norm();
        Self {
            fcn: fcn_shared,
            x_init: x0,
            norm_grad_fx_init: norm_grad_f0,
            opts,
        }
    }
    //}}}
    //{{{ fn: set_innter_rtol
    #[trace_fn]
    fn set_uncon_options(&mut self) {}
    //}}}
    //{{{ fn: is_converged
    #[trace_fn]
    fn is_converged(&self) -> Option<ConvergedReason>
    {
        None
    }
    //}}}
    //{{{ fn: print_status
    fn print_status(&self) {}
    //}}}
    //{{{ fn: update_penalties_shifts
    fn update_lagrangian(
        &mut self,
        uncon_ret: UnconstrainedReturns,
    ) -> IterData
    {
        todo!()
    }
    //}}}
}
//}}}
//{{{ impl: ConstrainedMinimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> ConstrainedMinimizer
    for AugmentedLagrangian<F1, F2, F3>
{
    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, ConstrainedError>
    {
        let n_iter = self.opts.constrained_opts.max_iter;
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k: IterData;

        for k in 1..n_iter
        {
            // iter_prev_k = iter_k;

            // let ret = minimize(
            //     self.fcn.clone(),
            //     iter_prev_k.x,
            //     self.opts.uncon_method.clone(),
            // )?;
        }

        Err(ConstrainedError::MaxIterations(0 as usize))
    }
}
//}}}
