#![cfg(feature = "serde")]

use topohedral_optimize::{
    BaseOptions, BoundConstrainedOptions, BoundConstraints, BoundedOptions, LineSearchMethod,
    NocedalOptions, QuasiNewtonOptions, QuasiNewtonUpdateMethod,
};

#[test]
fn nested_method_configuration_round_trips() {
    let options = QuasiNewtonOptions::new(
        BaseOptions::new(1e-7, 1e-9, 250),
        LineSearchMethod::Nocedal(NocedalOptions::default()),
        QuasiNewtonUpdateMethod::Dfp,
    )
    .with_restart(25);

    let json = serde_json::to_string(&options).unwrap();
    let decoded: QuasiNewtonOptions = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, options);
}

#[test]
fn constraints_round_trip_in_deterministic_index_order() {
    let mut bounds = BoundConstraints::new(4);
    bounds.add_bounds(3, None, Some(7.0)).unwrap();
    bounds.add_bounds(1, Some(-2.0), Some(2.0)).unwrap();

    let json = serde_json::to_string(&bounds).unwrap();
    let decoded: BoundConstraints = serde_json::from_str(&json).unwrap();
    assert_eq!(decoded, bounds);

    let first_index = json.find("\"1\"").unwrap();
    let second_index = json.find("\"3\"").unwrap();
    assert!(first_index < second_index);
}

#[test]
fn representative_scalar_and_bound_options_round_trip() {
    let bounded = BoundedOptions::new(-4.0, 9.0)
        .unwrap()
        .with_x_abs_tolerance(1e-9);
    let bound_common =
        BoundConstrainedOptions::new(BaseOptions::default()).with_constraint_tolerance(1e-7);

    let bounded_json = serde_json::to_string(&bounded).unwrap();
    let common_json = serde_json::to_string(&bound_common).unwrap();

    assert_eq!(
        serde_json::from_str::<BoundedOptions>(&bounded_json).unwrap(),
        bounded
    );
    assert_eq!(
        serde_json::from_str::<BoundConstrainedOptions>(&common_json).unwrap(),
        bound_common
    );
}
