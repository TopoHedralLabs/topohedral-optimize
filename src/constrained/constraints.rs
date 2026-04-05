//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{Matrix, RealVectorFn, Vector};
//}}}
//{{{ std imports
use std::collections::HashMap;
//}}}
//{{{ dep imports
use topohedral_linalg::{Shape, TransformOps, VectorOps};
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
//{{{ impl: BoundsConstraints
impl BoundsConstraints
{
    pub fn new(num_variables: usize) -> Self
    {
        Self {
            num_variables,
            bounds: HashMap::<usize, (Option<f64>, Option<f64>)>::new(),
        }
    }

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

    fn num_eq_constraints(&self) -> usize
    {
        0
    }

    fn num_ieq_constraints(&self) -> usize
    {
        let mut num_constraints = 0;
        for (_, (lower_bound, upper_bound)) in &self.bounds
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
            if let Some(_) = opt_lower
            {
                (*val)[(*variable_index, constraint_index)] = -1.0;
                constraint_index += 1;
            }

            if let Some(_) = opt_upper
            {
                (*val)[(*variable_index, constraint_index)] = 1.0;
                constraint_index += 1;
            }
        }
    }
}
//}}}
