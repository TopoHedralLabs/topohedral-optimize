# Developer Notes

The crate is organized around a small shared interface and separate optimizer
families:

- `common` defines function traits, vectors, iteration data, stopping options,
  and result types;
- `scalar`, `line_search`, `unconstrained`, `bound_constrained`, and `constrained`
  contain the algorithm families and their factories;
- `constraints` contains sparse bound representation and projection helpers;
- `quadratic_model` provides curvature updates used by quasi-Newton methods.

Public APIs are re-exported from `src/lib.rs`, so new user-facing types should be
documented there and tested through the corresponding top-level entry point.
