//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
use std::ops::{Index, IndexMut, Add, Sub, Mul, Div, Neg};
//}}}
//{{{ dep imports 
use topohedral_linalg::VectorOps;
//}}}
//--------------------------------------------------------------------------------------------------

// pub trait VectorTrait:
//     VectorOps<ScalarType = f64> + 
//     Index<usize> +
//     IndexMut<usize> +
//     Add<Self, Output = Self> +
//     Sub<Self, Output = Self> +
//     Mul<f64, Output = Self> +
//     Div<f64, Output = Self> +
//     Neg<Output = Self>
//     + Clone
// { }

/// 1D real-valued function trait
pub trait RealFn1 {

    fn eval(&mut self, x: f64) -> f64;
    fn diff(&mut self, x: f64) -> f64;
}


pub trait RealFn {

    type Vector;

    fn eval(&mut self, x: &Self::Vector) -> f64;
    fn grad(&mut self, x: &Self::Vector) -> Self::Vector;

}
