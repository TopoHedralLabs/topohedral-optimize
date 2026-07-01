use approx::assert_relative_eq;
use topohedral_linalg::{DMatrix, DVector, MatMul, Shape, VecType, VectorOps};
use topohedral_optimize::quadratic_model::{QuadraticModel, UpdateType};
use topohedral_optimize::{Matrix, Vector};

fn colvec(values: &[f64]) -> Vector
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}

fn basis(
    n: usize,
    index: usize,
) -> Vector
{
    let mut out = Vector::zeros_vec(n, VecType::Col);
    out[index] = 1.0;
    out
}

fn quadratic_value(
    hess: &Matrix,
    linear: &Vector,
    constant: f64,
    x: &Vector,
) -> f64
{
    0.5 * x.dot(&hess.matmul(x)) + linear.dot(x) + constant
}

fn quadratic_grad(
    hess: &Matrix,
    linear: &Vector,
    x: &Vector,
) -> Vector
{
    hess.matmul(x) + linear.clone()
}

fn implied_linear_term(model: &QuadraticModel) -> Vector
{
    model.grad_fk.clone() - model.hess_k.matmul(&model.xk)
}

fn implied_constant_term(model: &QuadraticModel) -> f64
{
    model.fk - model.grad_fk.dot(&model.xk)
        + 0.5 * model.xk.dot(&model.hess_k.matmul(&model.xk))
}

fn assert_vector_close(
    actual: &Vector,
    expected: &Vector,
    epsilon: f64,
)
{
    assert_eq!(actual.len(), expected.len());
    for (actual_i, expected_i) in actual.iter().zip(expected.iter())
    {
        assert_relative_eq!(*actual_i, *expected_i, epsilon = epsilon);
    }
}

fn assert_matrix_close(
    actual: &Matrix,
    expected: &Matrix,
    epsilon: f64,
)
{
    assert_eq!(actual.nrows(), expected.nrows());
    assert_eq!(actual.ncols(), expected.ncols());
    for row in 0..actual.nrows()
    {
        for col in 0..actual.ncols()
        {
            assert_relative_eq!(actual[(row, col)], expected[(row, col)], epsilon = epsilon);
        }
    }
}

#[test]
fn coordinate_updates_recover_quadratic_coefficients()
{
    let hess = DMatrix::<f64>::from_row_slice(
        &[
            4.0, 0.0, 0.0, //
            0.0, 7.0, 0.0, //
            0.0, 0.0, 11.0,
        ],
        3,
        3,
    );
    let linear = colvec(&[-3.0, 2.0, 5.0]);
    let constant = -1.25;
    let xk = colvec(&[0.25, -1.0, 2.0]);

    let mut model = QuadraticModel::new(3);
    model.xk.copy_from(&xk);
    model.fk = quadratic_value(&hess, &linear, constant, &xk);
    model.grad_fk.copy_from(&quadratic_grad(&hess, &linear, &xk));

    for i in 0..3
    {
        let delta_x = basis(3, i);
        let delta_grad = quadratic_grad(&hess, &linear, &(xk.clone() + delta_x.clone()))
            - quadratic_grad(&hess, &linear, &xk);

        assert!(model.try_update(&delta_x, &delta_grad, UpdateType::Both));
    }

    assert_matrix_close(&model.hess_k, &hess, 1e-12);
    assert_vector_close(&implied_linear_term(&model), &linear, 1e-12);
    assert_relative_eq!(
        implied_constant_term(&model),
        constant,
        epsilon = 1e-12
    );

    let x = colvec(&[-0.5, 1.5, 0.75]);
    let model_value = model.fk
        + model.grad_fk.dot(&(x.clone() - model.xk.clone()))
        + 0.5
            * (x.clone() - model.xk.clone())
                .dot(&model.hess_k.matmul(&(x.clone() - model.xk.clone())));
    assert_relative_eq!(
        model_value,
        quadratic_value(&hess, &linear, constant, &x),
        epsilon = 1e-12
    );
}

#[test]
fn hessian_and_inverse_hessian_stay_inverse()
{
    let hess = DMatrix::<f64>::from_row_slice(
        &[
            5.0, 1.0, 0.5, //
            1.0, 4.0, -0.25, //
            0.5, -0.25, 3.0,
        ],
        3,
        3,
    );
    let steps = [
        colvec(&[1.0, 0.5, -0.25]),
        colvec(&[-0.2, 1.0, 0.4]),
        colvec(&[0.3, -0.6, 1.0]),
        colvec(&[0.7, 0.2, 0.5]),
    ];

    let mut model = QuadraticModel::new(3);
    for delta_x in steps
    {
        let delta_grad = hess.matmul(&delta_x);
        assert!(model.try_update(&delta_x, &delta_grad, UpdateType::Both));
    }

    let identity = Matrix::identity(3, 3);
    assert_matrix_close(&model.hess_k.matmul(&model.inv_hess_k), &identity, 1e-10);
    assert_matrix_close(&model.inv_hess_k.matmul(&model.hess_k), &identity, 1e-10);
}
