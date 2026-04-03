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
    bounds: HashMap<usize, (Option<f64>, Option<f64>)>,
}
//}}}
//{{{ impl: BoundsConstraints
impl BoundsConstraints
{
    pub fn add_bounds(
        &mut self,
        variable_index: usize,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    )
    {
        self.bounds
            .insert(variable_index, (lower_bound, upper_bound));
    }

    fn num_eq_constraints(&self) -> usize
    {
        0
    }

    fn nun_ieq_constraints(&self) -> usize
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
        todo!()
    }

    fn dimension_range(&self) -> usize
    {
        todo!()
    }

    fn eval(
        &mut self,
        x: &Vector,
        val: &mut Vector,
    )
    {
    }

    fn grad(
        &mut self,
        x: &Vector,
        val: &mut crate::Matrix,
    )
    {
        todo!()
    }
}
//}}}

//{{{ struct: LinearConstraints
pub struct LinearConstraints
{
    eq_constraints: HashMap<usize, (f64, Vector)>,
    ieq_constraints: HashMap<usize, (Option<f64>, Option<f64>, Vector)>,
}
//}}}

//{{{ impl: LinearConstraints
impl LinearConstraints
{
    fn num_eq_constraints(&self) -> usize
    {
        0
    }

    fn nun_ieq_constraints(&self) -> usize
    {
        let mut num_constraints = 0;
        for (_, (lower_bound, upper_bound, _)) in &self.ieq_constraints
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
