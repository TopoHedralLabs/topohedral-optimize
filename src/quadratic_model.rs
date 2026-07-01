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
use topohedral_linalg::{MatMul, MatrixOps, OuterProduct, TransformOps, VecType::Col, VectorOps};
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
        self.had_first_update = false;
    }

    #[trace_fn]
    pub fn update_iterate(
        &mut self,
        xk: &Vector,
        fk: f64,
        grad_fk: &Vector,
    )
    {
        self.xk.copy_from(xk);
        self.fk = fk;
        self.grad_fk.copy_from(grad_fk);
    }

    #[trace_fn]
    pub fn try_update(
        &mut self,
        delta_x: &Vector,
        delta_grad_fx: &Vector,
        update_type: UpdateType,
    ) -> bool
    {
        let sk = delta_x;
        let yk = delta_grad_fx;
        let sk_dot_yk = sk.dot(yk);
        let yk_dot_yk = yk.dot(yk);

        if sk_dot_yk <= f64::EPSILON * yk_dot_yk.max(1.0)
        {
            return false;
        }

        if !self.had_first_update
        {
            self.first_update(yk_dot_yk, sk_dot_yk, update_type);
        }

        let hessk_mult_sk = self.hess_k.matmul(sk);
        let curvature = sk.dot(&hessk_mult_sk);
        if curvature < 0.0
        {
            return false;
        }

        let mut out = true;

        if update_type == UpdateType::Direct || update_type == UpdateType::Both
        {
            out = out && self.try_update_direct(delta_x, delta_grad_fx, &hessk_mult_sk);
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
        hess_mult_delta_x: &Vector,
    ) -> bool
    {
        let n = self.xk.len();
        let sk = delta_x;
        let yk = delta_grad_fx;
        let b_sk = hess_mult_delta_x;
        let sk_b_sk = sk.dot(b_sk);
        let yk_dot_sk = yk.dot(sk);

        if sk_b_sk <= 0.0 || yk_dot_sk <= 0.0
        {
            return false;
        }

        for i in 0..n
        {
            for j in 0..n
            {
                self.hess_k[(i, j)] += (yk[i] * yk[j] / yk_dot_sk) - (b_sk[i] * b_sk[j] / sk_b_sk);
            }
        }
        true
    }

    fn try_update_inverse(
        &mut self,
        delta_x: &Vector,
        delta_grad_fx: &Vector,
    ) -> bool
    {
        let n = self.xk.len();
        let sk = delta_x;
        let yk = delta_grad_fx;
        let yk_dot_sk = yk.dot(sk);
        if yk_dot_sk <= 0.0
        {
            return false;
        }
        let rho_k = 1.0 / (sk.dot(yk));
        let identity = Matrix::identity(n, n);
        let mat1: Matrix = (&identity - rho_k * sk.outer(yk)).into();
        let mat2 = mat1.transpose();
        let mat3: Matrix = (rho_k * sk.outer(sk)).into();
        let new_inv_hess = mat1.matmul(&self.inv_hess_k).matmul(mat2) + mat3;
        self.inv_hess_k.copy_from(new_inv_hess);
        true
    }
}
