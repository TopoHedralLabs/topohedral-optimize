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
use topohedral_linalg::{DVector, VecType};
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
    let mut values = DVector::<f64>::zeros_vec(0, VecType::Col);
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
    let mut values = DVector::<f64>::zeros_vec(constraints.dimension_range(), VecType::Col);
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
//{{{ test: cauchy path upper bound
#[test]
fn test_cauchy_path_single_upper_bound_hit()
{
    let mut constraints = BoundsConstraints::new(1);
    constraints.add_bounds(0, Some(0.0), Some(1.0));

    let x = colvec(&[0.5]);
    let d = colvec(&[1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert_eq!(path.len(), 1);
    assert_eq!(path[0].1, 0);
    assert_relative_eq!(path[0].0, 0.5, epsilon = 1e-12);
}
//}}}
//{{{ test: cauchy path lower bound
#[test]
fn test_cauchy_path_single_lower_bound_hit()
{
    let mut constraints = BoundsConstraints::new(1);
    constraints.add_bounds(0, Some(0.0), Some(1.0));

    let x = colvec(&[0.5]);
    let d = colvec(&[-1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert_eq!(path.len(), 1);
    assert_eq!(path[0].1, 0);
    assert_relative_eq!(path[0].0, 0.5, epsilon = 1e-12);
}
//}}}
//{{{ test: cauchy path no hit
#[test]
fn test_cauchy_path_direction_away_from_only_bound_returns_empty()
{
    // Only a lower bound; direction is positive (moving away from it).
    let mut constraints = BoundsConstraints::new(1);
    constraints.add_bounds(0, Some(0.0), None);

    let x = colvec(&[0.5]);
    let d = colvec(&[1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert!(path.is_empty());
}
//}}}
//{{{ test: cauchy path multiple variables sorted
#[test]
fn test_cauchy_path_multiple_variables_sorted_by_t()
{
    // Three variables all with bounds [0, 2], direction [1, 1, 1].
    // var 0: hits upper at t = (2 - 0.5) / 1 = 1.5
    // var 1: hits upper at t = (2 - 0.0) / 1 = 2.0
    // var 2: hits upper at t = (2 - 1.5) / 1 = 0.5
    // Sorted: [(0.5, 2), (1.5, 0), (2.0, 1)]
    let mut constraints = BoundsConstraints::new(3);
    constraints.add_bounds(0, Some(0.0), Some(2.0));
    constraints.add_bounds(1, Some(0.0), Some(2.0));
    constraints.add_bounds(2, Some(0.0), Some(2.0));

    let x = colvec(&[0.5, 0.0, 1.5]);
    let d = colvec(&[1.0, 1.0, 1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert_eq!(path.len(), 3);
    assert_relative_eq!(path[0].0, 0.5, epsilon = 1e-12);
    assert_eq!(path[0].1, 2);
    assert_relative_eq!(path[1].0, 1.5, epsilon = 1e-12);
    assert_eq!(path[1].1, 0);
    assert_relative_eq!(path[2].0, 2.0, epsilon = 1e-12);
    assert_eq!(path[2].1, 1);
}
//}}}
//{{{ test: cauchy path infeasible start clamped
#[test]
fn test_cauchy_path_infeasible_start_uses_clamped_location()
{
    // x = [1.5] is outside upper bound 1.0; clamped to [1.0].
    // d = [-1.0], lower = 0.0 → t = (0.0 - 1.0) / (-1.0) = 1.0
    let mut constraints = BoundsConstraints::new(1);
    constraints.add_bounds(0, Some(0.0), Some(1.0));

    let x = colvec(&[1.5]);
    let d = colvec(&[-1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert_eq!(path.len(), 1);
    assert_eq!(path[0].1, 0);
    assert_relative_eq!(path[0].0, 1.0, epsilon = 1e-12);
}
//}}}
//{{{ test: cauchy path already at bound
#[test]
fn test_cauchy_path_at_lower_bound_direction_into_bound_returns_t_zero()
{
    // x is at the lower bound; d pushes into it → t = 0.
    let mut constraints = BoundsConstraints::new(1);
    constraints.add_bounds(0, Some(0.0), Some(1.0));

    let x = colvec(&[0.0]);
    let d = colvec(&[-1.0]);
    let path = constraints.cauchy_path(&x, &d);

    assert_eq!(path.len(), 1);
    assert_eq!(path[0].1, 0);
    assert_relative_eq!(path[0].0, 0.0, epsilon = 1e-12);
}
//}}}
//{{{ fun: project_at
fn project_at(
    constraints: &BoundsConstraints,
    x: &Vector,
    d: &Vector,
    t: f64,
) -> Vector
{
    let vals: Vec<f64> = (0..x.len()).map(|i| x[i] + t * d[i]).collect();
    let mut p = colvec(&vals);
    constraints.clamp(&mut p);
    p
}
//}}}
//{{{ test: cauchy path geometric kinks
#[test]
fn test_cauchy_path_geometric_projected_path_kinks_at_breakpoints()
{
    // 3 variables, all bounded [0, 2].
    // x = [0.5, 0.0, 1.5], d = [1, 1, 1].
    //
    // Breakpoints from cauchy_path:
    //   t = 0.5  → var 2 hits upper bound (2.0)
    //   t = 1.5  → var 0 hits upper bound (2.0)
    //   t = 2.0  → var 1 hits upper bound (2.0)
    //
    // Between breakpoints the path is linear in each free variable; at each
    // breakpoint one component "kinks" onto its bound and stays there.
    let mut constraints = BoundsConstraints::new(3);
    constraints.add_bounds(0, Some(0.0), Some(2.0));
    constraints.add_bounds(1, Some(0.0), Some(2.0));
    constraints.add_bounds(2, Some(0.0), Some(2.0));

    let x = colvec(&[0.5, 0.0, 1.5]);
    let d = colvec(&[1.0, 1.0, 1.0]);

    // Verify the breakpoint sequence first.
    let path = constraints.cauchy_path(&x, &d);
    assert_eq!(path.len(), 3);
    assert_relative_eq!(path[0].0, 0.5, epsilon = 1e-12);
    assert_eq!(path[0].1, 2);
    assert_relative_eq!(path[1].0, 1.5, epsilon = 1e-12);
    assert_eq!(path[1].1, 0);
    assert_relative_eq!(path[2].0, 2.0, epsilon = 1e-12);
    assert_eq!(path[2].1, 1);

    // Segment 0: t in [0, 0.5) — all three components move freely.
    let p = project_at(&constraints, &x, &d, 0.0);
    assert_relative_eq!(p[0], 0.5, epsilon = 1e-12);
    assert_relative_eq!(p[1], 0.0, epsilon = 1e-12);
    assert_relative_eq!(p[2], 1.5, epsilon = 1e-12);

    let p = project_at(&constraints, &x, &d, 0.25);
    assert_relative_eq!(p[0], 0.75, epsilon = 1e-12);
    assert_relative_eq!(p[1], 0.25, epsilon = 1e-12);
    assert_relative_eq!(p[2], 1.75, epsilon = 1e-12);

    // At t = 0.5: var 2 exactly on upper bound (kink point).
    let p = project_at(&constraints, &x, &d, 0.5);
    assert_relative_eq!(p[0], 1.0, epsilon = 1e-12);
    assert_relative_eq!(p[1], 0.5, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12);

    // Segment 1: t in (0.5, 1.5) — var 2 pinned at 2.0, vars 0 and 1 still free.
    let p = project_at(&constraints, &x, &d, 1.0);
    assert_relative_eq!(p[0], 1.5, epsilon = 1e-12);
    assert_relative_eq!(p[1], 1.0, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12); // pinned

    // At t = 1.5: var 0 exactly on upper bound (kink point), var 2 still pinned.
    let p = project_at(&constraints, &x, &d, 1.5);
    assert_relative_eq!(p[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(p[1], 1.5, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12); // pinned

    // Segment 2: t in (1.5, 2.0) — vars 0 and 2 pinned, only var 1 moves.
    let p = project_at(&constraints, &x, &d, 1.75);
    assert_relative_eq!(p[0], 2.0, epsilon = 1e-12); // pinned
    assert_relative_eq!(p[1], 1.75, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12); // pinned

    // At t = 2.0: all variables pinned at their upper bounds.
    let p = project_at(&constraints, &x, &d, 2.0);
    assert_relative_eq!(p[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(p[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12);

    // Beyond last breakpoint: path stays constant.
    let p = project_at(&constraints, &x, &d, 3.0);
    assert_relative_eq!(p[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(p[1], 2.0, epsilon = 1e-12);
    assert_relative_eq!(p[2], 2.0, epsilon = 1e-12);
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
