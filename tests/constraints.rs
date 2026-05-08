#![feature(generic_const_exprs)]
#![allow(incomplete_features)]

//{{{ crate imports
use topohedral_optimize::constraints::{BoundsConstraints, NoConstraints};
use topohedral_optimize::{Matrix, RealVectorFn, Vector};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use approx::assert_relative_eq;
use topohedral_linalg::dvector::{DVector, VecType};
use topohedral_linalg::{Shape, TransformOps, VectorOps};
//}}}

//{{{ fun: colvec
fn colvec(values: &[f64]) -> Vector
{
    DVector::<f64>::from_slice_vec(values, values.len(), VecType::Col)
}
//}}}
//{{{ fun: assert_bound_columns_match
fn assert_bound_columns_match(
    x: &Vector,
    values: &Vector,
    gradient: &Matrix,
    lower_bounds: &[Option<f64>],
    upper_bounds: &[Option<f64>],
)
{
    let mut seen_lower = vec![0usize; x.len()];
    let mut seen_upper = vec![0usize; x.len()];

    for constraint_index in 0..values.len()
    {
        let mut nonzero_entries = Vec::new();
        for variable_index in 0..x.len()
        {
            let entry = gradient[(variable_index, constraint_index)];
            if entry.abs() > 1e-12
            {
                nonzero_entries.push((variable_index, entry));
            }
        }

        assert_eq!(nonzero_entries.len(), 1);
        let (variable_index, entry) = nonzero_entries[0];

        if entry < 0.0
        {
            seen_lower[variable_index] += 1;
            let lower = lower_bounds[variable_index]
                .expect("gradient column with -1.0 must correspond to a lower bound");
            assert_relative_eq!(entry, -1.0, epsilon = 1e-12);
            assert_relative_eq!(
                values[constraint_index],
                lower - x[variable_index],
                epsilon = 1e-12
            );
        }
        else
        {
            seen_upper[variable_index] += 1;
            let upper = upper_bounds[variable_index]
                .expect("gradient column with +1.0 must correspond to an upper bound");
            assert_relative_eq!(entry, 1.0, epsilon = 1e-12);
            assert_relative_eq!(
                values[constraint_index],
                x[variable_index] - upper,
                epsilon = 1e-12
            );
        }
    }

    for variable_index in 0..x.len()
    {
        assert_eq!(
            seen_lower[variable_index],
            usize::from(lower_bounds[variable_index].is_some())
        );
        assert_eq!(
            seen_upper[variable_index],
            usize::from(upper_bounds[variable_index].is_some())
        );
    }
}
//}}}

//{{{ test: no constraints
#[test]
fn test_no_constraints_is_empty_and_noop()
{
    let mut constraints = NoConstraints;

    assert_eq!(constraints.dimension_domain(), 0);
    assert_eq!(constraints.dimension_range(), 0);

    let x = colvec(&[]);
    let mut values = DVector::<f64>::zeros_cvec(0, VecType::Col);
    let mut gradient = Matrix::zeros(0, 0);

    constraints.eval(&x, &mut values);
    constraints.grad(&x, &mut gradient);

    assert_eq!(values.len(), 0);
    assert_eq!(gradient.nrows(), 0);
    assert_eq!(gradient.ncols(), 0);
}
//}}}
//{{{ test: mixed bounds
#[test]
fn test_bounds_constraints_mixed_bounds_eval_and_grad_match_public_contract()
{
    let mut constraints = BoundsConstraints::new(4);
    constraints.add_bounds(0, Some(-1.0), Some(2.0));
    constraints.add_bounds(2, Some(0.5), None);
    constraints.add_bounds(3, None, Some(4.5));

    assert_eq!(constraints.dimension_domain(), 4);
    assert_eq!(constraints.dimension_range(), 4);

    let x = colvec(&[1.5, -3.0, -0.25, 5.0]);
    let mut values = DVector::<f64>::zeros_cvec(constraints.dimension_range(), VecType::Col);
    let mut gradient = Matrix::zeros(
        constraints.dimension_domain(),
        constraints.dimension_range(),
    );
    gradient.fill(7.0);

    constraints.eval(&x, &mut values);
    constraints.grad(&x, &mut gradient);

    assert_bound_columns_match(
        &x,
        &values,
        &gradient,
        &[Some(-1.0), None, Some(0.5), None],
        &[Some(2.0), None, None, Some(4.5)],
    );
}
//}}}
//{{{ test: zero stale matrix entries
#[test]
fn test_bounds_constraints_grad_clears_stale_matrix_entries()
{
    let mut constraints = BoundsConstraints::new(3);
    constraints.add_bounds(1, Some(-2.0), Some(3.0));

    let x = colvec(&[10.0, 1.5, -4.0]);
    let mut gradient = Matrix::zeros(3, constraints.dimension_range());
    gradient.fill(-9.0);

    constraints.grad(&x, &mut gradient);

    for row in 0..gradient.nrows()
    {
        for col in 0..gradient.ncols()
        {
            let expected = match (row, col)
            {
                (1, 0) => -1.0,
                (1, 1) => 1.0,
                _ => 0.0,
            };
            assert_relative_eq!(gradient[(row, col)], expected, epsilon = 1e-12);
        }
    }
}
//}}}
