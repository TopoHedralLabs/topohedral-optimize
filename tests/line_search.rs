

//{{{ crate imports
use topohedral_optimize::{line_search::LineSearchFcn, RealFn, RealFn1, CountingRealFn};
//}}}
//{{{ std imports
use std::{rc::Rc, sync::Mutex};
use std::sync::Arc;
use std::cell::RefCell;
//}}}
//{{{ dep imports
use topohedral_linalg::{
    MatMul,
    dvector::{DVector, VecType},
    dmatrix::{ DMatrix},
    scvector::SCVector,
    smatrix::{SMatrix},
    GreaterThan, VectorOps
};
use approx::assert_relative_eq;
//}}}

#[derive(Debug, Clone)]
struct QuadraticDynamic {
    n: usize,
    center: DVector<f64>,
    coeffs: DMatrix<f64>,
}