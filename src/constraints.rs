//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{
    constraints::BoundStatus::{AtLower, AtUpper},
    Matrix, RealVectorFn, Vector,
};
//}}}
//{{{ std imports
use std::collections::HashMap;
//}}}
//{{{ dep imports
use topohedral_linalg::{Shape, TransformOps, VecType, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: CauchyPathPoint
#[derive(Debug, Clone)]
pub struct CauchyPathPoint
{
    pub alpha: f64,
    pub variable_index: usize,
    pub bound_status: BoundStatus,
}
//}}}
//{{{ struct: NoConstraints
#[derive(Debug, Clone, Copy)]
pub struct NoConstraints;
//}}}
//{{{ impl: RealVectorFn for NoConstraints
impl RealVectorFn for NoConstraints
{
    #[trace_fn]
    fn dimension_domain(&self) -> usize
    {
        0
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize
    {
        0
    }

    #[trace_fn]
    fn eval(
        &mut self,
        _x: &Vector,
        _val: &mut Vector,
    )
    {
    }

    #[trace_fn]
    fn grad(
        &mut self,
        _x: &Vector,
        _val: &mut Matrix,
    )
    {
    }
}
//}}}
//{{{ struct: BoundsConstraints
#[derive(Debug, Clone)]
pub struct BoundsConstraints
{
    num_variables: usize,
    bounds: HashMap<usize, (Option<f64>, Option<f64>)>,
}
//}}}
//{{{ enum: BoundStatus
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundStatus
{
    Free,
    AtLower,
    AtUpper,
}
//}}}
//{{{ struct: BoundSignature
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoundSignature(Box<[(usize, BoundStatus)]>);
//}}}
//{{{ impl: BoundsConstraints
impl BoundsConstraints
{
    //{{{ fn: new
    #[trace_fn]
    pub fn new(num_variables: usize) -> Self
    {
        Self {
            num_variables,
            bounds: HashMap::<usize, (Option<f64>, Option<f64>)>::new(),
        }
    }
    //}}}
    #[trace_fn]
    pub fn is_empty(&self) -> bool
    {
        self.bounds.is_empty()
    }
    //{{{ fn: add_bounds
    #[trace_fn]
    pub fn add_bounds(
        &mut self,
        variable_index: usize,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    )
    {
        assert!(variable_index < self.dimension_domain());
        assert!(!self.bounds.contains_key(&variable_index));
        self.bounds
            .insert(variable_index, (lower_bound, upper_bound));
    }
    //}}}
    //{{{ fn: get_lower
    #[trace_fn]
    pub fn get_lower(
        &self,
        idx: usize,
    ) -> Option<f64>
    {
        let op_bounds_i = self.bounds.get(&idx);
        if let Some(bounds_i) = op_bounds_i
        {
            return bounds_i.0;
        }
        None
    }
    //}}}
    //{{{ fn: get_higher
    #[trace_fn]
    pub fn get_upper(
        &self,
        idx: usize,
    ) -> Option<f64>
    {
        let op_bounds_i = self.bounds.get(&idx);
        if let Some(bounds_i) = op_bounds_i
        {
            return bounds_i.1;
        }
        None
    }
    //}}}
    //{{{ fn: num_ieq_constraints
    #[trace_fn]
    pub fn num_ieq_constraints(&self) -> usize
    {
        let mut num_constraints = 0;
        for (lower_bound, upper_bound) in self.bounds.values()
        {
            if lower_bound.is_some()
            {
                num_constraints += 1;
            }
            if upper_bound.is_some()
            {
                num_constraints += 1;
            }
        }
        num_constraints
    }
    //}}}
    //{{{ fn: clamp
    #[trace_fn]
    pub fn clamp(
        &self,
        x: &mut Vector,
    )
    {
        for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
        {
            if let Some(low_bound) = opt_low_bound
            {
                let xi = x[*variable_index];
                (*x)[*variable_index] = xi.max(*low_bound);
            }
            if let Some(high_bound) = opt_high_bound
            {
                let xi = x[*variable_index];
                (*x)[*variable_index] = xi.min(*high_bound);
            }
        }
    }
    //}}}
    //{{{ fn: projected_direction
    #[trace_fn]
    pub fn projected_direction(
        &self,
        location: &Vector,
        direction: &Vector,
        alpha: f64,
    ) -> Vector
    {
        let mut new_location: Vector = (location + alpha * direction).into();
        self.clamp(&mut new_location);
        new_location -= location.clone();
        return new_location;
    }
    //}}}
    //{{{ fn: max_feasible_step
    #[trace_fn]
    pub fn max_feasible_step(
        &self,
        location: &Vector,
        direction: &Vector,
    ) -> f64
    {
        self.cauchy_path(location, direction)
            .first()
            .map(|point| point.alpha)
            .unwrap_or(f64::INFINITY)
            .max(0.0)
    }
    //}}}
    //{{{ fn: cauchy_path
    #[trace_fn]
    pub fn cauchy_path(
        &self,
        location: &Vector,
        direction: &Vector,
    ) -> Vec<CauchyPathPoint>
    {
        let mut x_start = location.clone();
        self.clamp(&mut x_start);

        let mut breakpoints = Vec::<CauchyPathPoint>::with_capacity(self.bounds.len());

        for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
        {
            let vi = *variable_index;
            if let Some(low_bound) = opt_low_bound
            {
                let gi = direction[vi];
                if gi < 0.0
                {
                    let xi = x_start[vi];
                    breakpoints.push(CauchyPathPoint {
                        alpha: (low_bound - xi) / gi,
                        variable_index: vi,
                        bound_status: AtLower,
                    });
                }
            }
            if let Some(high_bound) = opt_high_bound
            {
                let gi = direction[vi];
                if gi > 0.0
                {
                    let xi = x_start[vi];
                    breakpoints.push(CauchyPathPoint {
                        alpha: (high_bound - xi) / gi,
                        variable_index: vi,
                        bound_status: AtUpper,
                    });
                }
            }
        }
        breakpoints.sort_by(|a, b| {
            a.alpha
                .partial_cmp(&b.alpha)
                .unwrap()
                .then_with(|| a.variable_index.cmp(&b.variable_index))
        });
        breakpoints
    }
    //}}}
    //{{{ fn: bound_statuses
    #[trace_fn]
    pub fn bound_statuses(
        &self,
        location: &Vector,
        direction: Option<&Vector>,
    ) -> Vec<BoundStatus>
    {
        assert_eq!(location.len(), self.dimension_domain());
        let mut statuses = vec![BoundStatus::Free; self.num_variables];

        if let Some(direction) = direction
        {
            assert_eq!(direction.len(), self.dimension_domain());

            let mut x_clamped = location.clone();
            self.clamp(&mut x_clamped);

            for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
            {
                if let Some(low_bound) = opt_low_bound
                {
                    let gi = direction[*variable_index];
                    let xi = x_clamped[*variable_index];
                    if gi < 0.0 && xi == *low_bound
                    {
                        statuses[*variable_index] = AtLower;
                        continue;
                    }
                }
                if let Some(high_bound) = opt_high_bound
                {
                    let gi = direction[*variable_index];
                    let xi = x_clamped[*variable_index];
                    if gi > 0.0 && xi == *high_bound
                    {
                        statuses[*variable_index] = AtUpper;
                    }
                }
            }
        }
        else
        {
            for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
            {
                let xi = location[*variable_index];
                if opt_low_bound.is_some_and(|low_bound| xi <= low_bound)
                {
                    statuses[*variable_index] = AtLower;
                    continue;
                }
                if opt_high_bound.is_some_and(|high_bound| xi >= high_bound)
                {
                    statuses[*variable_index] = AtUpper;
                }
            }
        }

        statuses
    }
    //}}}
    //{{{ fn active_signature
    #[trace_fn]
    pub fn active_signature(
        &self,
        x: &Vector,
    ) -> BoundSignature
    {
        let mut sig = Vec::<(usize, BoundStatus)>::with_capacity(self.bounds.len());

        for (&idx, (lower, upper)) in &self.bounds
        {
            let xi = x[idx];

            if lower.is_some_and(|lower| xi <= lower)
            {
                sig.push((idx, BoundStatus::AtLower));
            }
            else if upper.is_some_and(|upper| xi >= upper)
            {
                sig.push((idx, BoundStatus::AtUpper));
            }
        }

        sig.sort_unstable_by_key(|(idx, _)| *idx);
        BoundSignature(sig.into_boxed_slice())
    }
    //}}}
    //{{{ fn: mask_gradient_in_place
    #[trace_fn]
    pub fn mask_gradient_in_place(
        &self,
        x: &Vector,
        grad_f: &mut Vector,
    )
    {
        for (&idx, (lower, upper)) in &self.bounds
        {
            let xi = x[idx];

            if lower.is_some_and(|lower| xi <= lower) || upper.is_some_and(|upper| xi >= upper)
            {
                (*grad_f)[idx] = 0.0;
            }
        }
    }
    //}}}
    //{{{ fn: masked_gradient
    #[trace_fn]
    pub fn masked_gradient(
        &self,
        x: &Vector,
        grad: &Vector,
    ) -> Vector
    {
        let mut out = grad.clone();
        self.mask_gradient_in_place(x, &mut out);
        return out;
    }

    //}}}
    //{{{ fn: minimum_distance
    #[trace_fn]
    pub fn minimum_distance(
        &self,
        x: &Vector,
    ) -> f64
    {
        let mut min_dist = f64::MAX;
        for (idx, (opt_lower, opt_upper)) in self.bounds.iter()
        {
            let xi = x[*idx];
            let mut lower_dist = f64::MAX;
            let mut upper_dist = f64::MAX;

            if let Some(lower) = opt_lower
            {
                lower_dist = xi - lower
            }

            if let Some(upper) = opt_upper
            {
                upper_dist = upper - xi;
            }

            min_dist = min_dist.min(f64::min(lower_dist, upper_dist));
        }
        min_dist
    }
    //}}}
    //{{{ fn: all_distances
    #[trace_fn]
    pub fn all_distances(
        &self,
        x: &Vector,
    ) -> Vector
    {
        let mut distances = Vector::from_value_vec(f64::INFINITY, x.len(), VecType::Col);
        for (idx, (opt_lower, opt_upper)) in self.bounds.iter()
        {
            let xi = x[*idx];
            let mut lower_dist = f64::MAX;
            let mut upper_dist = f64::MAX;

            if let Some(lower) = opt_lower
            {
                lower_dist = xi - lower
            }

            if let Some(upper) = opt_upper
            {
                upper_dist = upper - xi;
            }

            distances[*idx] = lower_dist.min(upper_dist);
        }
        distances
    }
    //}}}
}
//}}}
//{{{ impl: RealVectorFn for BoundsConstraints
impl RealVectorFn for BoundsConstraints
{
    #[trace_fn]
    fn dimension_domain(&self) -> usize
    {
        self.num_variables
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize
    {
        self.num_ieq_constraints()
    }

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
        assert_eq!(x.len(), self.dimension_domain());
        assert_eq!(val.len(), self.dimension_range());

        let mut constraint_index = 0;
        for (variable_index, (opt_lower, opt_upper)) in self.bounds.iter()
        {
            let xi = x[*variable_index];
            if let Some(lower) = opt_lower
            {
                (*val)[constraint_index] = lower - xi;
                constraint_index += 1;
            }

            if let Some(upper) = opt_upper
            {
                (*val)[constraint_index] = xi - upper;
                constraint_index += 1;
            }
        }
    }

    #[trace_fn]
    fn grad(
        &mut self,
        x: &Vector,
        val: &mut crate::Matrix,
    )
    {
        assert_eq!(x.len(), self.dimension_domain());
        assert_eq!(val.ncols(), self.dimension_range());
        assert_eq!(val.nrows(), self.dimension_domain());

        val.fill(0.0);
        let mut constraint_index = 0;
        for (variable_index, (opt_lower, opt_upper)) in self.bounds.iter()
        {
            if opt_lower.is_some()
            {
                (*val)[(*variable_index, constraint_index)] = -1.0;
                constraint_index += 1;
            }

            if opt_upper.is_some()
            {
                (*val)[(*variable_index, constraint_index)] = 1.0;
                constraint_index += 1;
            }
        }
    }
}
//}}}
