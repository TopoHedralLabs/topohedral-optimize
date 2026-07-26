use topohedral_optimize::{BaseOptions, BoundConstraints, LineSearchOptions, ValidationError};

#[test]
fn common_options_reject_non_finite_and_empty_iteration_limits() {
    assert!(matches!(
        BaseOptions::new(f64::NAN, 1e-8, 100).validate(),
        Err(ValidationError::InvalidFloat {
            parameter: "grad_rtol",
            ..
        })
    ));
    assert!(matches!(
        BaseOptions::new(1e-6, 1e-8, 0).validate(),
        Err(ValidationError::InvalidInteger {
            parameter: "max_iter",
            ..
        })
    ));
}

#[test]
fn line_search_options_enforce_wolfe_and_step_invariants() {
    assert!(LineSearchOptions::default().validate().is_ok());
    assert!(matches!(
        LineSearchOptions::new(0.9, 0.1, 1e-8, 1e5).validate(),
        Err(ValidationError::InvalidFloat {
            parameter: "c1",
            ..
        })
    ));
    assert!(matches!(
        LineSearchOptions::new(1e-4, 0.9, 2.0, 1.0).validate(),
        Err(ValidationError::InvalidFloat {
            parameter: "step_min",
            ..
        })
    ));
}

#[test]
fn bounds_validate_indices_duplicates_and_values() {
    let mut bounds = BoundConstraints::new(2);

    assert!(matches!(
        bounds.add_bounds(2, Some(0.0), None),
        Err(ValidationError::BoundIndexOutOfRange { index: 2, .. })
    ));
    assert!(matches!(
        bounds.add_bounds(0, None, None),
        Err(ValidationError::EmptyBound { index: 0 })
    ));
    assert!(matches!(
        bounds.add_bounds(0, Some(2.0), Some(1.0)),
        Err(ValidationError::InvalidBounds { index: 0, .. })
    ));

    bounds.add_bounds(0, Some(0.0), Some(1.0)).unwrap();
    assert!(matches!(
        bounds.add_bounds(0, Some(-1.0), None),
        Err(ValidationError::DuplicateBound { index: 0 })
    ));
}
