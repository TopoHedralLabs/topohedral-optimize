//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    bound_constrained::{minimize as bcon_minimize, BoundConstrainedMethod},
    common::{self, arc_real_fn, ConvergedReason, CountingRealFn, IterData, Returns},
    constrained::{ConstrainedError, ConstriainedOptions},
    constraints::BoundsConstraints,
    unconstrained::{minimize as uncon_minimize, UnconstrainedMethod},
    Matrix, Minimizer, RealFn, RealVectorFn, Vector,
};
use core::f64;
//}}}
//{{{ std imports
use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};
//}}}
//{{{ dep imports
#[allow(unused_imports)]
use topohedral_linalg::MatrixOps;
use topohedral_linalg::{
    FloatTransformOps, MatMul, ReduceOps, TransformOps, VecType::Col, VectorOps,
};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ collection: constants
const DEFUALT_INITIAL_PENALTY: f64 = 1.0;
const DEFUALT_CONSTRAINT_IMPROVEMENT_FACTOR: f64 = 0.9;
const DEFAULT_PENALTY_GROWTH_FACTOR: f64 = 2.5;
/// Ceiling for ω_k (inner stationarity tolerance) on the very first outer
/// iteration — matches the old `set_inner_tolerances` atol upper clamp.
/// From the second outer iteration on, the previous ω_k becomes the ceiling
/// instead, which is what makes the sequence monotone non-increasing.
const OMEGA_INIT_CEIL: f64 = 1e-2;
//}}}
//{{{ enum: LagrangianType
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LagrangianType {
    AugmentedLagrangian,
    Lagrangian,
}
//}}}

//{{{ struct Options
#[derive(Clone)]
pub struct Options {
    pub constrained_opts: ConstriainedOptions,
    pub inner_method: InnerMethod,
    pub initial_penalty: f64,
    pub constraint_improvement_factor: f64,
    pub penalty_growth_factor: f64,
}
//}}}
//{{{ impl: Options
impl Options {
    #[trace_fn]
    pub fn new(
        constrained_opts: ConstriainedOptions,
        inner_method: InnerMethod,
    ) -> Self {
        Self {
            constrained_opts,
            inner_method,
            initial_penalty: DEFUALT_INITIAL_PENALTY,
            constraint_improvement_factor: DEFUALT_CONSTRAINT_IMPROVEMENT_FACTOR,
            penalty_growth_factor: DEFAULT_PENALTY_GROWTH_FACTOR,
        }
    }

    #[trace_fn]
    pub(crate) fn inner_method_mut(&mut self) -> &mut InnerMethod {
        &mut self.inner_method
    }

    #[trace_fn]
    pub(crate) fn inner_method(&self) -> &InnerMethod {
        &self.inner_method
    }
}
//}}}

//{{{ struct: LagrangianPenaltyData
#[derive(Debug, Clone)]
struct LagrangianPenaltyData<F: RealVectorFn> {
    pub function: F,
    pub penalties: Vector,
    pub shifts: Vector,
    pub values: Vector,
    pub gradients: Matrix,
    pub max_violations: Vec<(f64, bool)>,
}
//}}}
//{{{ impl: LagrangianPenaltyData
impl<F: RealVectorFn> LagrangianPenaltyData<F> {
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F,
        initial_penalty: f64,
    ) -> Self {
        let num_constraints = fcn.dimension_range();
        let dimension = fcn.dimension_domain();
        //{{{ trace
        trace!(target: "aug", "Creating constraint data object with {num_constraints} constraints");
        //}}}

        let initial_penalties = Vector::from_value_vec(initial_penalty, num_constraints, Col);
        let zero_vector = Vector::zeros_vec(num_constraints, Col);
        let zero_matrix = Matrix::zeros(dimension, num_constraints);

        Self {
            function: fcn,
            penalties: initial_penalties,
            shifts: zero_vector.clone(),
            values: zero_vector,
            gradients: zero_matrix,
            max_violations: vec![(0.0, false); num_constraints],
        }
    }
    //}}}
    //{{{ fn: update_values
    #[trace_fn]
    fn update_values(
        &mut self,
        x: &Vector,
    ) {
        self.function.eval(x, &mut self.values);
    }
    //}}}
    //{{{ fn: update_gradients
    #[trace_fn]
    fn update_gradients(
        &mut self,
        x: &Vector,
    ) {
        self.function.grad(x, &mut self.gradients);
    }
    //}}}
    //{{{ fn: update_max_violations
    #[trace_fn]
    fn update_max_violations(
        &mut self,
        improvement_factor: f64,
        is_ineq: bool,
    ) {
        let current_max_violations = &mut self.max_violations;
        let current_values = &self.values;

        //{{{ trace
        trace!(target: "aug", "current_max_violations = {:?}", current_max_violations);
        trace!(target: "aug", "current_values= {}", current_values.clone().transpose());
        //}}}

        for (_i, max_violation, was_violated, constraint_value_i) in current_max_violations
            .iter_mut()
            .zip(current_values.iter())
            .enumerate()
            .map(|(_i, ((max_violation, was_violated), hi))| (_i, max_violation, was_violated, hi))
        {
            let max_violation_val = *max_violation;
            let constraint_val_tmp = if is_ineq {
                constraint_value_i.max(0.0)
            } else {
                constraint_value_i.abs()
            };
            *was_violated = constraint_val_tmp > improvement_factor * max_violation_val;
            if *was_violated {
                *max_violation = constraint_value_i.abs()
            }
            //{{{ trace
            info!(target: "aug", "i = {} was_violated = {} max_violation = {:.4e}",
               _i, was_violated, max_violation);
            //}}}
        }
    }
    //}}}
    //{{{ fn: compute_lagrange_multiplier_estimates
    #[trace_fn]
    fn compute_lagrange_multiplier_estimates(&self) -> Vector {
        (&self.penalties * &self.shifts).into()
    }
    //}}}
}
//}}}

//{{{ struct: EqPenalty
#[derive(Debug, Clone)]
struct EqPenalty<F: RealVectorFn> {
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl EqPenalty
impl<F: RealVectorFn> EqPenalty<F> {
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_penalties_shifts(
        &mut self,
        improvement_factor: f64,
        penalty_increase_factor: f64,
    ) {
        self.data.update_max_violations(improvement_factor, false);

        for (i, (_, was_violated)) in self.data.max_violations.iter().enumerate() {
            if *was_violated {
                self.data.penalties[i] *= penalty_increase_factor;
                self.data.shifts[i] /= penalty_increase_factor;
            } else {
                let hi = self.data.values[i];
                self.data.shifts[i] += hi;
            }
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn for EqPenalty
impl<F: RealVectorFn> RealFn for EqPenalty<F> {
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.data.update_values(x);
        let constraint_value = match self.lagrangian_type {
            LagrangianType::AugmentedLagrangian => {
                //{{{ trace
                trace!(target: "aug", "Computing augmented lagrangian");
                //}}}

                let mut shifted_h: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_h.transform(|hi| hi.powi(2));
                0.5 * self.data.penalties.dot(&shifted_h)
            }
            LagrangianType::Lagrangian => {
                //{{{ trace
                trace!(target: "aug", "Computing classical lagrangian");
                //}}}
                let lambda = self.data.compute_lagrange_multiplier_estimates();
                lambda.dot(&self.data.values)
            }
        };
        //{{{ trace
        trace!(target: "aug", "Evaluated equality constraint value: {constraint_value:.4e}");
        //}}}
        constraint_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        self.data.update_gradients(x);
        let weighted_constraint_values: Vector = match self.lagrangian_type {
            LagrangianType::AugmentedLagrangian => {
                (&self.data.penalties * (&self.data.values + &self.data.shifts)).into()
            }
            LagrangianType::Lagrangian => self.data.compute_lagrange_multiplier_estimates(),
        };
        self.data.gradients.matmul(&weighted_constraint_values)
    }
    //}}}
}
//}}}

//{{{ struct: IeqPenalty
#[derive(Debug, Clone)]
struct IeqPenalty<F: RealVectorFn> {
    data: LagrangianPenaltyData<F>,
    lagrangian_type: LagrangianType,
}
//}}}
//{{{ impl IeqPenalty
impl<F: RealVectorFn> IeqPenalty<F> {
    //{{{ fn: new
    #[trace_fn]
    fn new(
        fcn: F,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self {
        Self {
            data: LagrangianPenaltyData::new(fcn, initial_penalty),
            lagrangian_type,
        }
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_penalties_shifts(
        &mut self,
        improvement_factor: f64,
        penalty_increase_factor: f64,
    ) {
        self.data.update_max_violations(improvement_factor, true);

        for (i, (_, was_violated)) in self.data.max_violations.iter().enumerate() {
            //{{{ trace
            debug!(target: "aug", "i = {i} was_violated = {was_violated}");
            //}}}
            if *was_violated {
                //{{{ trace
                debug!(target: "aug", "i = {i} Constraint not improved, increasing penalties");
                //}}}
                self.data.penalties[i] *= penalty_increase_factor;
                self.data.shifts[i] /= penalty_increase_factor;
            } else {
                let gi = self.data.values[i];
                let old_shift = self.data.shifts[i];
                self.data.shifts[i] = f64::max(0.0, old_shift + gi);

                //{{{ trace
                debug!(target: "aug", "i = {i} Constraint improved, updating shifts: old_shift = {old_shift:.4e} new_shift = {:.4e}",
                            self.data.shifts[i]);
                //}}}
            }
        }
    }
    //}}}
}
//}}}
//{{{ impl RealFn  for IeqPenalty
impl<F: RealVectorFn> RealFn for IeqPenalty<F> {
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize {
        self.data.function.dimension_domain()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        self.data.update_values(x);
        match self.lagrangian_type {
            LagrangianType::AugmentedLagrangian => {
                let mut shifted_g: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_g.transform(|value| value.max(0.0).powi(2));
                0.5 * self.data.penalties.dot(&shifted_g)
            }
            LagrangianType::Lagrangian => {
                let mu = self.data.compute_lagrange_multiplier_estimates();
                mu.dot(&self.data.values)
            }
        }
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        self.data.update_gradients(x);
        let weighted_constraint_values: Vector = match self.lagrangian_type {
            LagrangianType::AugmentedLagrangian => {
                let mut shifted_values: Vector = (&self.data.values + &self.data.shifts).into();
                shifted_values.pos();
                (&self.data.penalties * &shifted_values).into()
            }
            LagrangianType::Lagrangian => self.data.compute_lagrange_multiplier_estimates(),
        };
        self.data.gradients.matmul(&weighted_constraint_values)
    }
    //}}}
}
//}}}

//{{{ struct CachedValues
#[derive(Debug, Clone)]
struct CachedValues {
    fcn_value: f64,
    fcn_grad: Vector,
    eq_constraint_value: f64,
    eq_constriant_grad: Vector,
    ieq_constraint_value: f64,
    ieq_constraint_grad: Vector,
    all_constraint_value: f64,
    all_constraint_grad: Vector,
    auglag_value: f64,
    auglag_grad: Vector,
}
//}}}
//{{{ impl: CachedValues
impl CachedValues {
    //{{{ fn: new
    #[trace_fn]
    fn new(n: usize) -> Self {
        let zero_vector = Vector::zeros_vec(n, Col);
        Self {
            fcn_value: 0.0,
            fcn_grad: zero_vector.clone(),
            eq_constraint_value: 0.0,
            eq_constriant_grad: zero_vector.clone(),
            ieq_constraint_value: 0.0,
            ieq_constraint_grad: zero_vector.clone(),
            all_constraint_value: 0.0,
            all_constraint_grad: zero_vector.clone(),
            auglag_value: 0.0,
            auglag_grad: zero_vector.clone(),
        }
    }
    //}}}
}
//}}}

//{{{ struct: AugmentedLagrangianFcn
#[derive(Debug, Clone)]
pub struct AugmentedLagrangianFcn<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> {
    fcn: F1,
    eq_penalty: Option<EqPenalty<F2>>,
    ieq_penalty: Option<IeqPenalty<F3>>,
    lagrangian_type: LagrangianType,
    cached_values: HashMap<LagrangianType, CachedValues>,
}
//}}}
//{{{ impl: AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangianFcn<F1, F2, F3> {
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        initial_penalty: f64,
        lagrangian_type: LagrangianType,
    ) -> Self {
        let eq_penalty =
            eq_constraints.map(|eq_con| EqPenalty::new(eq_con, initial_penalty, lagrangian_type));

        let ieq_penalty = ieq_constraints
            .map(|ieq_con| IeqPenalty::new(ieq_con, initial_penalty, lagrangian_type));

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
    ) {
        if let Some(eq_penalty) = &mut self.eq_penalty {
            eq_penalty.lagrangian_type = lagrangian_type
        }

        if let Some(ieq_penalty) = &mut self.ieq_penalty {
            ieq_penalty.lagrangian_type = lagrangian_type
        }
    }
    //}}}
    //{{{ fn: update_max_eq_violations
    #[trace_fn]
    fn update_max_eq_violations(
        &self,
        max_eq_violations: &mut Vector,
    ) {
        let eq_penalty = self.eq_penalty.as_ref().unwrap();

        for (max_violation, hi) in max_eq_violations
            .iter_mut()
            .zip(eq_penalty.data.values.iter())
        {
            *max_violation = max_violation.max(hi.abs());
        }
    }
    //}}}
    //{{{ fn: update_max_ieq_violations
    #[trace_fn]
    fn update_max_ieq_violations(
        &self,
        max_ieq_violations: &mut Vector,
    ) {
        let ieq_penalty = self.ieq_penalty.as_ref().unwrap();

        for (max_violation, gi, shift_i) in max_ieq_violations
            .iter_mut()
            .zip(ieq_penalty.data.values.iter())
            .zip(ieq_penalty.data.shifts.iter())
            .map(|((max_violation, gi), phi_i)| (max_violation, gi, phi_i))
        {
            *max_violation = max_violation.max(((-gi).min(*shift_i)).abs());
        }
    }
    //}}}
    //{{{ fn: has_eq_penalty
    #[trace_fn]
    fn has_eq_penalty(&self) -> bool {
        self.eq_penalty.is_some()
    }
    //}}}
    //{{{ fn: has_ieq_penalty
    #[trace_fn]
    fn has_ieq_penalty(&self) -> bool {
        self.ieq_penalty.is_some()
    }
    //}}}
    //{{{ fn: get_cached_values
    #[trace_fn]
    fn get_cached_values(&self) -> &CachedValues {
        self.cached_values.get(&self.lagrangian_type).unwrap()
    }
    //}}}
    //{{{ fn: get_cached_values_mut
    #[trace_fn]
    fn get_cached_values_mut(&mut self) -> &mut CachedValues {
        self.cached_values.get_mut(&self.lagrangian_type).unwrap()
    }
    //}}}
    //{{{ fn: is_constrained
    #[trace_fn]
    fn is_constrained(&self) -> bool {
        self.has_eq_penalty() || self.has_ieq_penalty()
    }
    //}}}
}

//}}}
//{{{ impl: RealFn for AugmentedLagrangianFcn
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> RealFn for AugmentedLagrangianFcn<F1, F2, F3> {
    //{{{ fn: dimension
    #[trace_fn]
    fn dimension(&self) -> usize {
        self.fcn.dimension()
    }
    //}}}
    //{{{ fn: eval
    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> f64 {
        let fcn_value = self.fcn.eval(x);

        //{{{ trace
        info!(target: "aug", "Evaluated objective: {fcn_value:.4e}");
        //}}}

        let eq_value = if let Some(eq_penalty) = &mut self.eq_penalty {
            let eq_value = eq_penalty.eval(x);
            //{{{ trace
            debug!(target: "aug", "Evaluated EQ penalty: {eq_value:.4e}");
            //}}}
            eq_value
        } else {
            0.0
        };

        let ieq_value = if let Some(ieq_penalty) = &mut self.ieq_penalty {
            let ieq_value = ieq_penalty.eval(x);
            //{{{ trace
            debug!(target: "aug", "Evaluated IEQ penalty: {ieq_value:.4e}");
            //}}}
            ieq_value
        } else {
            0.0
        };

        let auglag_value = fcn_value + eq_value + ieq_value;

        {
            let cached_value = self.get_cached_values_mut();
            cached_value.fcn_value = fcn_value;
            cached_value.eq_constraint_value = eq_value;
            cached_value.ieq_constraint_value = ieq_value;
            cached_value.all_constraint_value = eq_value + ieq_value;
            cached_value.auglag_value = auglag_value;
        }

        //{{{ trace
        debug!(target: "aug", "Augmented Lagrangian value: {auglag_value:.4e}");
        //}}}
        auglag_value
    }
    //}}}
    //{{{ fn: grad
    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
    ) -> Vector {
        let n = self.dimension();

        let fcn_grad = self.fcn.grad(x);

        //{{{ trace
        trace!(target: "aug", "Evaluated Objective Gradient = {}",
                    fcn_grad.clone().transpose());
        //}}}

        let eq_penalty_grad = if let Some(eq_constraint_data) = &mut self.eq_penalty {
            let eq_penalty_grad = eq_constraint_data.grad(x);
            //{{{ trace
            trace!(target: "aug", "Evaluated EQ Gradient = {}",
                    eq_penalty_grad.clone().transpose());
            //}}}
            eq_penalty_grad
        } else {
            Vector::zeros_vec(n, Col)
        };

        let ieq_penalty_grad = if let Some(ieq_constraint_data) = &mut self.ieq_penalty {
            let ieq_penalty_grad = ieq_constraint_data.grad(x);
            //{{{ trace
            trace!(target: "aug", "Evaluated IEQ Gradient = {}",
                    ieq_penalty_grad.clone().transpose());
            //}}}
            ieq_penalty_grad
        } else {
            Vector::zeros_vec(n, Col)
        };

        let grad_aug_lag: Vector = (&fcn_grad + &eq_penalty_grad + &ieq_penalty_grad).into();

        {
            let cached_value = self.get_cached_values_mut();
            cached_value.fcn_grad = fcn_grad;
            cached_value.eq_constriant_grad = eq_penalty_grad.clone();
            cached_value.ieq_constraint_grad = ieq_penalty_grad.clone();
            cached_value.all_constraint_grad = (&eq_penalty_grad + &ieq_penalty_grad).into();
            cached_value.auglag_grad = grad_aug_lag.clone();
        }

        //{{{ trace
        trace!(target: "aug", "Evaluated Augmented Lagrangian Gradient = {}",
                    grad_aug_lag.clone().transpose());
        //}}}
        grad_aug_lag
    }
    //}}}
}
//}}}

//{{{ struct: AugmentedLagrangian
pub struct AugmentedLagrangian<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> {
    fcn: Arc<Mutex<CountingRealFn<AugmentedLagrangianFcn<F1, F2, F3>>>>,
    x_init: Vector,
    bounds: Option<BoundsConstraints>,
    opts: Options,
    omega_k: f64,
    /// Scheduling-only ramp used by `compute_omega_k`; grows every outer
    /// iteration unconditionally. Deliberately decoupled from the real
    /// per-constraint penalties (`EqPenalty`/`IeqPenalty`), which only grow
    /// when a constraint fails to improve and can otherwise stay fixed
    /// forever — seeding ω_k's floor off the real penalty would then let it
    /// get stuck loose indefinitely.
    mu_k: f64,
}
//}}}
//{{{ impl: AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> AugmentedLagrangian<F1, F2, F3> {
    //{{{ fn: new
    #[trace_fn]
    pub fn new(
        fcn: F1,
        bounds: Option<BoundsConstraints>,
        eq_constraints: Option<F2>,
        ieq_constraints: Option<F3>,
        x0: Vector,
        opts: Options,
    ) -> Self {
        //{{{ trace
        trace!(target: "aug", "Creating new Augmented Lagrangian function");
        //}}}

        let mut fcn_shared = arc_real_fn(CountingRealFn::new(AugmentedLagrangianFcn::new(
            fcn,
            eq_constraints,
            ieq_constraints,
            opts.initial_penalty,
            LagrangianType::AugmentedLagrangian,
        )));

        let _ = fcn_shared.eval(&x0);
        let _ = fcn_shared.grad(&x0);

        let user_omega = opts.constrained_opts.base_opts.grad_atol;
        let mu0 = opts.initial_penalty;
        Self {
            fcn: fcn_shared,
            x_init: x0,
            bounds,
            opts,
            omega_k: OMEGA_INIT_CEIL.max(user_omega),
            mu_k: mu0,
        }
    }
    //}}}
    //{{{ fn: is_converged
    #[trace_fn]
    fn is_converged(
        &self,
        iter_k: &IterData,
    ) -> Option<ConvergedReason> {
        //{{{ trace
        info!(target: "aug", "Checking convergence");
        //}}}
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {
            let mut classical_auglag = fcn.clone();
            classical_auglag.set_lagrangian_type(LagrangianType::Lagrangian);
            let _ = classical_auglag.eval(&iter_k.x);
            let _ = classical_auglag.grad(&iter_k.x);

            let norm_eq = if let Some(eq_penalty) = &classical_auglag.eq_penalty
            {
                eq_penalty.data.values.abs_max().unwrap()
            }
            else {
                0.0
            };
            let norm_ieq = if let Some(ieq_penalty) = &classical_auglag.ieq_penalty
            {
                ieq_penalty.data.values.posed().abs_max().unwrap()
            }
            else {
                0.0
            };

            let cached_values = classical_auglag.get_cached_values();
            let norm_grad_f = cached_values.fcn_grad.abs_max().unwrap();
            let norm_grad_penalty = cached_values.all_constraint_grad.abs_max().unwrap();
            let stationarity_gradient = if let Some(bounds) = &self.bounds
            {
                bounds.projected_direction(&iter_k.x, &(-cached_values.auglag_grad.clone()), 1.0)
            }
            else
            {
                cached_values.auglag_grad.clone()
            };
            let _norm_grad_auglag = cached_values.auglag_grad.abs_max().unwrap();
            let norm_projected_grad_auglag = stationarity_gradient.abs_max().unwrap_or(0.0);

            let residual_stationarity = norm_projected_grad_auglag;
            let residual_stationarity_scaled =
                norm_projected_grad_auglag / 1.0f64.max(norm_grad_f).max(norm_grad_penalty);
            let residual_primal = norm_eq.max(norm_ieq);
            //{{{ trace
            info!(target: "aug", "||∇P|| = {norm_grad_penalty:.4e} ||∇F|| = {norm_grad_f:.4} ||h|| = {norm_eq:.4e} ||g|| = {norm_ieq:.4e}");
            info!(target: "aug", "||∇L|| = {_norm_grad_auglag:.4e})");
            info!(target: "aug", "||∇L_proj|| = {norm_projected_grad_auglag:.4e})");
            info!(target: "aug", "||∇L_proj|| / max(1, ||∇F||, ||∇P||) = {residual_stationarity_scaled:.4e}");
            //}}}
            let ctol = self.opts.constrained_opts.constraint_tol;
            let rtol = self.opts.constrained_opts.base_opts.grad_rtol;
            let atol = self.opts.constrained_opts.base_opts.grad_atol;
            let constraints_satsifed = residual_primal < ctol;
            let stationarity_rtol_satisfied = residual_stationarity_scaled < rtol;
            let stationarity_atol_satisfied = residual_stationarity< atol;

            if constraints_satsifed && stationarity_rtol_satisfied
            {
                Some(ConvergedReason::Rtol)
            }
            else if constraints_satsifed && stationarity_atol_satisfied
            {
                Some(ConvergedReason::Atol)
            }
            else
            {
                None
            }
        })
    }
    //}}}
    //{{{ fn: print_status
    fn print_status(
        &self,
        _k: u64,
        _iter_k: &IterData,
    ) {
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {

            info!(target: "aug", "******************************************************************************************** k = {_k}");
            trace!(target: "aug", "Current solution: {}", _iter_k.x.clone().transpose());
            trace!(target: "aug", "Current gradient: {}", _iter_k.grad_fx.clone().transpose());

            if let Some(_eq_penalty) = &fcn.eq_penalty
            {
                trace!(target: "aug", "Current EQ penalties: {}", _eq_penalty.data.penalties.clone().transpose());
                trace!(target: "aug", "Current EQ shifts: {}", _eq_penalty.data.shifts.clone().transpose());
            }

            if let Some(_ieq_penalty) = &fcn.ieq_penalty
            {
                trace!(target: "aug", "Current IEQ penalties: {}", _ieq_penalty.data.penalties.clone().transpose());
                trace!(target: "aug", "Current IEQ shifts: {}", _ieq_penalty.data.shifts.clone().transpose());
            }


            info!(target: "aug", "******************************************************************************************** k = {_k}");
        })
    }
    //}}}
    //{{{ fn: update_penalties_shifts
    #[trace_fn]
    fn update_lagrangian(
        &mut self,
        constraint_improvement_factor: f64,
        penalty_increase_factor: f64,
        uncon_ret: Returns,
    ) -> IterData {
        self.fcn.lock().unwrap().with_inner_mut(|fcn| {
            if let Some(eq_penalty) = &mut fcn.eq_penalty {
                //{{{ trace
                info!(target: "aug", "Updating penalties and shifts for EQ");
                //}}}
                eq_penalty.update_penalties_shifts(
                    constraint_improvement_factor,
                    penalty_increase_factor,
                );
            }

            if let Some(ieq_penalty) = &mut fcn.ieq_penalty {
                //{{{ trace
                info!(target: "aug", "Updating penalties and shifts for IEQ");
                //}}}
                ieq_penalty.update_penalties_shifts(
                    constraint_improvement_factor,
                    penalty_increase_factor,
                );
            }
        });
        IterData::new(self.fcn.clone(), &uncon_ret.xmin)
    }
    //}}}
    /// Computes this outer iteration's inner stationarity tolerance ω_k and
    /// advances the stored ceiling for the next call.
    ///
    /// `ω_k` follows the classical LANCELOT/Birgin–Martínez ramp
    /// `0.1 * (1/μ_k)^1.5`, clamped to `[user_grad_atol, previous ω_k]`. That
    /// makes the sequence monotone non-increasing (never loosens between
    /// outer iterations), reach tolerances tighter than the old hardcoded
    /// 1e-6 floor when the user asks for them via `grad_atol`, and — because
    /// `μ_k` (see the `mu_k` field) grows by a fixed factor every outer
    /// iteration unconditionally — land on `user_grad_atol` within a bounded
    /// number of iterations regardless of problem-specific behavior.
    ///
    /// Earlier versions scaled this off `max(‖c‖∞, 1/μ_k)`, tying it to the
    /// live constraint residual. Two failure modes ruled that out:
    /// - Seeding the floor off the *real* per-constraint penalty stalled
    ///   forever, since `update_max_violations` only grows a constraint's
    ///   penalty when that constraint fails to improve — one that's already
    ///   comfortably satisfied can leave its penalty fixed indefinitely.
    /// - Seeding it off the live residual instead (even via a scheduling-only
    ///   `μ_k`) let `ω_k` keep chasing the residual down long after outer
    ///   stationarity was already satisfied and only feasibility was still
    ///   converging (which happens on its own schedule via the multiplier
    ///   update, not by solving the inner problem tighter) — for
    ///   ill-conditioned inner subproblems (e.g. quartics, or gradients with
    ///   O(10) magnitude requiring near-cancellation to resolve an O(1e-8)
    ///   residual) this pushed past the inner solver's achievable precision
    ///   and the line search failed outright. A pure μ_k ramp, independent
    ///   of the live residual, avoids both.
    #[trace_fn]
    fn compute_omega_k(&mut self) -> f64 {
        let user_omega = self.opts.constrained_opts.base_opts.grad_atol;
        let prev_omega_k = self.omega_k;
        let mu_k = self.mu_k;
        self.mu_k *= self.opts.penalty_growth_factor;

        let omega_target = 0.1 * (1.0 / mu_k).powf(1.5);
        let omega_k = omega_target.max(user_omega).min(prev_omega_k);

        self.omega_k = omega_k;
        //{{{ trace
        info!(target: "aug", "Inner ω_k = {omega_k:.4e} (prev = {prev_omega_k:.4e}, user floor = {user_omega:.4e})");
        //}}}
        omega_k
    }

    #[trace_fn]
    fn set_inner_tolerances(&mut self) -> InnerMethod {
        let inner_method = self.opts.inner_method.clone();
        let is_constrained = self
            .fcn
            .lock()
            .unwrap()
            .with_inner_mut(|fcn| fcn.is_constrained());

        match inner_method {
            InnerMethod::Unconstrained(mut uncon_method) => {
                if !is_constrained {
                    uncon_method.uncon_opts_mut().grad_rtol =
                        self.opts.constrained_opts.base_opts.grad_rtol;
                    uncon_method.uncon_opts_mut().grad_atol =
                        self.opts.constrained_opts.base_opts.grad_atol;
                } else {
                    let omega_k = self.compute_omega_k();
                    uncon_method.uncon_opts_mut().grad_rtol = 0.0;
                    uncon_method.uncon_opts_mut().grad_atol = omega_k;
                }
                InnerMethod::Unconstrained(uncon_method)
            }
            InnerMethod::BoundConstrained(mut bcon_method) => {
                if !is_constrained {
                    bcon_method.bound_opts_mut().base_opts.grad_rtol =
                        self.opts.constrained_opts.base_opts.grad_rtol;
                    bcon_method.bound_opts_mut().base_opts.grad_atol =
                        self.opts.constrained_opts.base_opts.grad_atol;
                } else {
                    let omega_k = self.compute_omega_k();
                    bcon_method.bound_opts_mut().base_opts.grad_rtol = 0.0;
                    bcon_method.bound_opts_mut().base_opts.grad_atol = omega_k;
                }
                InnerMethod::BoundConstrained(bcon_method)
            }
        }
    }
}
//}}}
//{{{ enum: InnerMethod
#[derive(Clone)]
pub enum InnerMethod {
    Unconstrained(UnconstrainedMethod),
    BoundConstrained(BoundConstrainedMethod),
}
//}}}
//{{{ fn: inner_minimize
fn inner_minimize<F: RealFn>(
    fcn: F,
    bounds: Option<BoundsConstraints>,
    x0: Vector,
    inner_method: InnerMethod,
) -> Result<common::Returns, ConstrainedError> {
    match inner_method {
        InnerMethod::Unconstrained(uncon_method) => Ok(uncon_minimize(fcn, x0, uncon_method)?),
        InnerMethod::BoundConstrained(bcon_method) => {
            Ok(bcon_minimize(fcn, bounds.unwrap(), x0, bcon_method)?)
        }
    }
}
//}}}
//{{{ impl: Minimizer for AugmentedLagrangian
impl<F1: RealFn, F2: RealVectorFn, F3: RealVectorFn> Minimizer for AugmentedLagrangian<F1, F2, F3> {
    type Error = ConstrainedError;
    type Returns = Returns;

    #[trace_fn]
    fn minimize(&mut self) -> Result<Returns, Self::Error> {
        let alpha = self.opts.constraint_improvement_factor;
        let beta = self.opts.penalty_growth_factor;
        let n_iter = self.opts.constrained_opts.base_opts.max_iter;
        let mut iter_k = IterData::new(self.fcn.clone(), &self.x_init);
        let mut iter_prev_k = iter_k.clone();

        for k in 1..n_iter {
            self.print_status(k, &iter_k);
            let uncon_method = self.set_inner_tolerances();
            let _omega_k = self.omega_k;
            let ret = match inner_minimize(
                self.fcn.clone(),
                self.bounds.clone(),
                iter_prev_k.x.clone(),
                uncon_method,
            ) {
                Ok(ret) => ret,
                // A line-search failure here (typically "no decreasing step found")
                // usually means the inner iterate is already stationary to numerical
                // precision for the requested ω_k — the augmented Lagrangian's
                // penalty-gradient term can dominate the objective's own gradient
                // near the optimum, so resolving a further decrease runs into
                // floating-point cancellation before ω_k reaches its target. Treat
                // the unmoved iterate as this step's result (the outer `is_converged`
                // check still gates whether that's actually good enough) instead of
                // aborting the whole outer solve; if the inner solver is genuinely
                // stuck rather than just precision-limited, this degrades to
                // `MaxIterations` once the outer loop stops making progress, rather
                // than masking a real failure as success.
                Err(_err) => {
                    //{{{ trace
                    info!(target: "aug", "Inner solve failed at ω_k = {_omega_k:.4e}: {_err:?} — treating {k}'s starting iterate as this step's result");
                    //}}}
                    let fmin = self
                        .fcn
                        .lock()
                        .unwrap()
                        .inner_mut()
                        .fcn
                        .eval(&iter_prev_k.x);
                    Returns {
                        xmin: iter_prev_k.x.clone(),
                        fmin,
                        reason: ConvergedReason::Atol,
                        num_iterations: 0,
                        num_fun_evals: 0,
                        num_grad_evals: 0,
                    }
                }
            };
            iter_k = self.update_lagrangian(alpha, beta, ret);
            iter_prev_k.copy_from(&iter_k);
            if let Some(reason) = self.is_converged(&iter_k) {
                //{{{ trace
                info!(target: "cg", "*********************************************");
                info!(target: "cg", "Converging with reason {reason:?}");
                info!(target: "cg","Convergence measures:");
                info!(target: "cg", "||∇L(k)|| = {:.4e}", iter_k.norm_grad_fx);
                info!(target: "cg", "*********************************************");
                //}}}
                let fmin = self.fcn.lock().unwrap().inner_mut().fcn.eval(&iter_k.x);
                let xmin = iter_k.x;
                return Ok(Returns {
                    fmin,
                    xmin,
                    reason,
                    num_iterations: k as usize,
                    num_fun_evals: 0,
                    num_grad_evals: 0,
                });
            }
        }

        Err(ConstrainedError::MaxIterations(0_usize))
    }
}
//}}}
