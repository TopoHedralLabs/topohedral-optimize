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
use std::ops::IndexMut;
//}}}
//{{{ dep imports
use topohedral_linalg::{Shape, TransformOps, VecType, VectorOps};
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: NoConstraints
#[derive(Debug, Clone, Copy)]
pub struct NoConstraints;
//}}}
//{{{ impl: RealVectorFn for NoConstraints
impl RealVectorFn for NoConstraints
{
    fn dimension_domain(&self) -> usize
    {
        0
    }

    fn dimension_range(&self) -> usize
    {
        0
    }

    fn eval(
        &mut self,
        _x: &Vector,
        _val: &mut Vector,
    )
    {
    }

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
    pub fn new(num_variables: usize) -> Self
    {
        Self {
            num_variables,
            bounds: HashMap::<usize, (Option<f64>, Option<f64>)>::new(),
        }
    }
    //}}}
    //{{{ fn: add_bounds
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
    //{{{ fn: num_ieq_constraints
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
    //{{{ fn: cauchy_path
    pub fn cauchy_path(
        &self,
        location: &Vector,
        direction: &Vector,
    ) -> Vec<(f64, usize)>
    {
        let mut out = Vec::<(f64, usize)>::with_capacity(self.bounds.len());

        let mut x_clamped = location.clone();
        self.clamp(&mut x_clamped);

        for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
        {
            if let Some(low_bound) = opt_low_bound
            {
                let gi = direction[*variable_index];
                if gi < 0.0
                {
                    let xi = x_clamped[*variable_index];
                    let alphai = (low_bound - xi) / gi;
                    out.push((alphai, *variable_index));
                }
            }
            if let Some(high_bound) = opt_high_bound
            {
                let gi = direction[*variable_index];
                if gi > 0.0
                {
                    let xi = x_clamped[*variable_index];
                    let alphai = (high_bound - xi) / gi;
                    out.push((alphai, *variable_index));
                }
            }
        }
        out.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
        out
    }
    //}}}
    //{{{ fn: active_set
    pub fn active_and_inactive_sets(
        &self,
        location: &Vector,
        direction: Option<&Vector>,
    ) -> (Vec<(usize, BoundStatus)>, Vec<usize>)
    {
        let mut active_set = Vec::<(usize, BoundStatus)>::with_capacity(self.num_variables);

        let mut inactive_set = Vec::<usize>::with_capacity(self.num_variables);
        for variable_index in 0..self.num_variables
        {
            if !self.bounds.contains_key(&variable_index)
            {
                inactive_set.push(variable_index);
            }
        }

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
                        active_set.push((*variable_index, AtLower));
                        continue;
                    }
                }
                if let Some(high_bound) = opt_high_bound
                {
                    let gi = direction[*variable_index];
                    let xi = x_clamped[*variable_index];
                    if gi > 0.0 && xi == *high_bound
                    {
                        active_set.push((*variable_index, AtUpper));
                        continue;
                    }
                }

                inactive_set.push(*variable_index)
            }
        }
        else
        {
            for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter()
            {
                let xi = location[*variable_index];
                if opt_low_bound.is_some_and(|low_bound| xi <= low_bound)
                {
                    active_set.push((*variable_index, AtLower));
                    continue;
                }
                if opt_high_bound.is_some_and(|high_bound| xi >= high_bound)
                {
                    active_set.push((*variable_index, AtUpper));
                    continue;
                }

                inactive_set.push(*variable_index)
            }
        }

        active_set.sort_by_key(|(idx, _)| *idx);
        inactive_set.sort();
        (active_set, inactive_set)
    }
    //}}}
    //{{{ fn active_signature
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
    pub fn mask_gradient_in_place(
        &self,
        x: &Vector,
        grad_f: &mut Vector,
    )
    {
        for (&idx, (lower, upper)) in &self.bounds
        {
            let xi = x[idx];

            if lower.is_some_and(|lower| xi <= lower)
            {
                (*grad_f)[idx] = 0.0;
            }
            else if upper.is_some_and(|upper| xi >= upper)
            {
                (*grad_f)[idx] = 0.0;
            }
        }
    }
    //}}}
    //{{{ fn: masked_gradient
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
    fn dimension_domain(&self) -> usize
    {
        self.num_variables
    }

    fn dimension_range(&self) -> usize
    {
        self.num_ieq_constraints()
    }

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
