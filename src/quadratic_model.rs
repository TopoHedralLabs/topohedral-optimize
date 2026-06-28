//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::common::{Matrix, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::{
    MatMul, ReduceOps, Shape, SubViewable, SubViewableMut, TransformOps, VecType::Col, VectorOps,
};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

pub struct QuadraticModel
{
    pub xk: Vector,
    pub fk: f64,
    pub grad_fk: Vector,
    pub hess_k: Matrix,
    pub inv_hess_k: Matrix,
    pub had_first_update: bool,
}

#[derive(PartialEq, Eq, Copy, Clone)]
pub enum UpdateType
{
    Direct,
    Inverse,
    Both,
}

impl QuadraticModel
{
    #[trace_fn]
    pub fn new(n: usize) -> Self
    {
        QuadraticModel {
            xk: Vector::zeros_vec(n, Col),
            fk: 0.0,
            grad_fk: Vector::zeros_vec(n, Col),
            hess_k: Matrix::identity(n, n),
            inv_hess_k: Matrix::identity(n, n),
            had_first_update: false,
        }
    }

    #[trace_fn]
    pub fn reset(&mut self)
    {
        let n = self.xk.len();
        self.hess_k.fill(0.0);
        self.inv_hess_k.fill(0.0);
        for i in 0..n
        {
            self.hess_k[(i, i)] = 1.0;
            self.inv_hess_k[(i, i)] = 1.0;
        }
    }

    #[trace_fn]
    pub fn try_update(
        &mut self,
        delta_x: &Vector,
        delta_grad_fx: &Vector,
        update_type: UpdateType,
    ) -> bool
    {
        let n = self.xk.len();
        let sk = delta_x;
        let yk = delta_x;
        let sk_dot_yk = sk.dot(yk);
        let yk_dot_yk = yk.dot(yk);

        if !self.had_first_update
        {
            self.first_update(yk_dot_yk, sk_dot_yk, update_type);
        }

        let curvature = sk.dot(&(self.hess_k.matmul(sk)));
        if sk_dot_yk <= f64::EPSILON * yk_dot_yk.max(1.0)
        {
            return false;
        }

        let mut out = true;

        if update_type == UpdateType::Direct || update_type == UpdateType::Both
        {
            out = out && self.try_update_direct(delta_x, delta_grad_fx);
        }
        if update_type == UpdateType::Inverse || update_type == UpdateType::Both
        {
            out = out && self.try_update_inverse(delta_x, delta_grad_fx);
        }
        out
    }

    fn first_update(
        &mut self,
        yk_dot_yk: f64,
        sk_dot_yk: f64,
        update_type: UpdateType,
    )
    {
        let n = self.xk.len();
        let scale = yk_dot_yk / sk_dot_yk;
        if update_type == UpdateType::Direct || update_type == UpdateType::Both
        {
            self.hess_k.fill(0.0);
            for i in 0..n
            {
                self.hess_k[(i, i)] = scale;
            }
        }
        if update_type == UpdateType::Inverse || update_type == UpdateType::Both
        {
            let inv_scale = 1.0 / scale;
            self.inv_hess_k.fill(0.0);
            for i in 0..n
            {
                self.inv_hess_k[(i, i)] = inv_scale;
            }
        }
        self.had_first_update = true;
    }

    fn try_update_direct(
        &mut self,
        delta_x: &Vector,
        delta_grad_fx: &Vector,
    ) -> bool
    {
        let sk = delta_x;
        let yk = delta_x;

        false
    }

    fn try_update_inverse(
        &mut self,
        delta_x: &Vector,
        delta_grad_fx: &Vector,
    ) -> bool
    {
        false
    }
}
