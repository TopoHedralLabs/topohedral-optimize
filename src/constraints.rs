//! Types and operations for representing optimization constraints.
//!
//! Bounds are stored sparsely and can be used to project points and directions.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::{DifferentiableFn, Matrix, ValidationError, Vector};
//}}}
//{{{ std imports
use std::collections::BTreeMap;
//}}}
//{{{ dep imports
use topohedral_linalg::{VecType, VectorOps};
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ struct: CauchyPathPoint
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
/// One breakpoint along a projected Cauchy path.
pub struct CauchyPathPoint {
    /// Step length at which the breakpoint occurs.
    pub alpha: f64,
    /// Variable reaching a bound.
    pub variable_index: usize,
    /// Bound reached by the variable.
    pub bound_status: BoundStatus,
}
//}}}
//{{{ struct: NoConstraints
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash)]
/// Empty vector-valued constraint function.
pub struct NoConstraints;
//}}}
//{{{ impl: RealVectorFn for NoConstraints
impl crate::DifferentiableFn for NoConstraints {
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;
    #[trace_fn]
    fn dimension_domain(&self) -> usize {
        0
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize {
        0
    }

    #[trace_fn]
    fn eval(
        &mut self,
        _x: &Vector,
    ) -> Vector {
        Vector::zeros_vec(0, VecType::Col)
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        _x: &Vector,
    ) -> Matrix {
        Matrix::zeros(0, 0)
    }
}
//}}}
//{{{ struct: BoundConstraints
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq)]
/// Sparse lower and upper bounds for a vector of variables.
pub struct BoundConstraints {
    num_variables: usize,
    bounds: BTreeMap<usize, (Option<f64>, Option<f64>)>,
}
//}}}
//{{{ enum: BoundStatus
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
/// Side of a bound that is active.
#[non_exhaustive]
pub enum BoundSide {
    /// Lower bound.
    Lower,
    /// Upper bound.
    Upper,
}
//}}}
//{{{ enum: BoundStatus
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, Copy, PartialEq)]
/// Position of a variable relative to its bounds.
#[non_exhaustive]
pub enum BoundStatus {
    /// Variable is not at a bound.
    Free,
    /// Variable is at its lower bound.
    AtLower(f64),
    /// Variable is at its upper bound.
    AtUpper(f64),
}
//}}}
//{{{ impl: BoundStatus
impl BoundStatus {
    /// Returns the active side, if the variable is bound.
    #[trace_fn]
    pub fn side(&self) -> Option<BoundSide> {
        match self {
            BoundStatus::Free => None,
            BoundStatus::AtLower(_) => Some(BoundSide::Lower),
            BoundStatus::AtUpper(_) => Some(BoundSide::Upper),
        }
    }

    /// Returns the active bound value, if any.
    #[trace_fn]
    pub fn value(&self) -> Option<f64> {
        match self {
            BoundStatus::Free => None,
            BoundStatus::AtLower(value) | BoundStatus::AtUpper(value) => Some(*value),
        }
    }
}
//}}}
//{{{ struct: BoundSignature
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
/// Hashable signature of the active bounds.
pub struct BoundSignature(Box<[(usize, BoundSide)]>);
//}}}
//{{{ impl: BoundConstraints
impl BoundConstraints {
    //{{{ fn: new
    /// Creates an empty bound set for `num_variables` variables.
    #[trace_fn]
    pub fn new(num_variables: usize) -> Self {
        Self {
            num_variables,
            bounds: BTreeMap::<usize, (Option<f64>, Option<f64>)>::new(),
        }
    }
    //}}}
    #[trace_fn]
    /// Returns whether no bounds have been added.
    pub fn is_empty(&self) -> bool {
        self.bounds.is_empty()
    }
    //{{{ fn: add_bounds
    /// Adds lower and/or upper bounds for one variable.
    ///
    /// Bound entries are iterated in ascending variable-index order. For a
    /// variable with both bounds, the lower-bound constraint comes first.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if the index is out of range, the variable
    /// already has bounds, neither bound is supplied, either bound is
    /// non-finite, or the lower bound exceeds the upper bound.
    #[trace_fn]
    pub fn add_bounds(
        &mut self,
        variable_index: usize,
        lower_bound: Option<f64>,
        upper_bound: Option<f64>,
    ) -> Result<(), ValidationError> {
        if variable_index >= self.dimension_domain() {
            return Err(ValidationError::BoundIndexOutOfRange {
                index: variable_index,
                num_variables: self.dimension_domain(),
            });
        }
        if self.bounds.contains_key(&variable_index) {
            return Err(ValidationError::DuplicateBound {
                index: variable_index,
            });
        }
        if lower_bound.is_none() && upper_bound.is_none() {
            return Err(ValidationError::EmptyBound {
                index: variable_index,
            });
        }
        if let Some(lower) = lower_bound {
            if !lower.is_finite() {
                return Err(ValidationError::InvalidFloat {
                    parameter: "lower_bound",
                    value: lower,
                    requirement: "must be finite",
                });
            }
        }
        if let Some(upper) = upper_bound {
            if !upper.is_finite() {
                return Err(ValidationError::InvalidFloat {
                    parameter: "upper_bound",
                    value: upper,
                    requirement: "must be finite",
                });
            }
        }
        if let (Some(lower), Some(upper)) = (lower_bound, upper_bound) {
            if lower > upper {
                return Err(ValidationError::InvalidBounds {
                    index: variable_index,
                    lower,
                    upper,
                });
            }
        }
        self.bounds
            .insert(variable_index, (lower_bound, upper_bound));
        Ok(())
    }
    //}}}
    //{{{ fn: get_lower
    /// Returns the lower bound for `idx`, if present.
    #[trace_fn]
    pub fn lower_bound(
        &self,
        idx: usize,
    ) -> Option<f64> {
        let op_bounds_i = self.bounds.get(&idx);
        if let Some(bounds_i) = op_bounds_i {
            return bounds_i.0;
        }
        None
    }

    /// Deprecated name for [`Self::lower_bound`].
    #[deprecated(since = "0.0.0", note = "renamed to `lower_bound`")]
    pub fn get_lower(
        &self,
        idx: usize,
    ) -> Option<f64> {
        self.lower_bound(idx)
    }
    //}}}
    //{{{ fn: get_higher
    /// Returns the upper bound for `idx`, if present.
    #[trace_fn]
    pub fn upper_bound(
        &self,
        idx: usize,
    ) -> Option<f64> {
        let op_bounds_i = self.bounds.get(&idx);
        if let Some(bounds_i) = op_bounds_i {
            return bounds_i.1;
        }
        None
    }

    /// Deprecated name for [`Self::upper_bound`].
    #[deprecated(since = "0.0.0", note = "renamed to `upper_bound`")]
    pub fn get_upper(
        &self,
        idx: usize,
    ) -> Option<f64> {
        self.upper_bound(idx)
    }
    //}}}
    //{{{ fn: num_inequality_constraints
    /// Counts the scalar inequality constraints represented by the bounds.
    #[trace_fn]
    pub fn num_inequality_constraints(&self) -> usize {
        let mut num_constraints = 0;
        for (lower_bound, upper_bound) in self.bounds.values() {
            if lower_bound.is_some() {
                num_constraints += 1;
            }
            if upper_bound.is_some() {
                num_constraints += 1;
            }
        }
        num_constraints
    }

    /// Deprecated name for [`Self::num_inequality_constraints`].
    #[deprecated(since = "0.0.0", note = "renamed to `num_inequality_constraints`")]
    pub fn num_ieq_constraints(&self) -> usize {
        self.num_inequality_constraints()
    }
    //}}}
    //{{{ fn: clamp
    /// Projects a point into the feasible box in place.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have the dimension supplied to [`Self::new`].
    #[trace_fn]
    pub fn clamp(
        &self,
        x: &mut Vector,
    ) {
        assert_eq!(
            x.len(),
            self.dimension_domain(),
            "point dimension must match bound dimension"
        );
        for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter() {
            if let Some(low_bound) = opt_low_bound {
                let xi = x[*variable_index];
                (*x)[*variable_index] = xi.max(*low_bound);
            }
            if let Some(high_bound) = opt_high_bound {
                let xi = x[*variable_index];
                (*x)[*variable_index] = xi.min(*high_bound);
            }
        }
    }
    //}}}
    //{{{ fn: projected_direction
    /// Returns the feasible displacement after projecting a trial step.
    ///
    /// # Panics
    ///
    /// Panics if `location` or `direction` does not have the bound dimension.
    #[trace_fn]
    pub fn projected_direction(
        &self,
        location: &Vector,
        direction: &Vector,
        alpha: f64,
    ) -> Vector {
        assert_eq!(
            location.len(),
            self.dimension_domain(),
            "location dimension must match bound dimension"
        );
        assert_eq!(
            direction.len(),
            self.dimension_domain(),
            "direction dimension must match bound dimension"
        );
        let mut new_location: Vector = (location + alpha * direction).into();
        self.clamp(&mut new_location);
        new_location -= location.clone();
        return new_location;
    }
    //}}}
    //{{{ fn: max_feasible_step
    /// Returns the largest nonnegative step before a bound is reached.
    ///
    /// # Panics
    ///
    /// Panics if `location` or `direction` does not have the bound dimension.
    #[trace_fn]
    pub fn max_feasible_step(
        &self,
        location: &Vector,
        direction: &Vector,
    ) -> f64 {
        self.cauchy_path(location, direction)
            .first()
            .map(|point| point.alpha)
            .unwrap_or(f64::INFINITY)
            .max(0.0)
    }
    //}}}
    //{{{ fn: cauchy_path
    /// Computes breakpoints along the projected search path.
    ///
    /// Breakpoints are sorted by increasing step length, then by variable
    /// index.
    ///
    /// # Panics
    ///
    /// Panics if `location` or `direction` does not have the bound dimension.
    #[trace_fn]
    pub fn cauchy_path(
        &self,
        location: &Vector,
        direction: &Vector,
    ) -> Vec<CauchyPathPoint> {
        let mut x_start = location.clone();
        self.clamp(&mut x_start);

        let mut breakpoints = Vec::<CauchyPathPoint>::with_capacity(self.bounds.len());

        for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter() {
            let vi = *variable_index;
            if let Some(low_bound) = opt_low_bound {
                let gi = direction[vi];
                if gi < 0.0 {
                    let xi = x_start[vi];
                    breakpoints.push(CauchyPathPoint {
                        alpha: (low_bound - xi) / gi,
                        variable_index: vi,
                        bound_status: BoundStatus::AtLower(*low_bound),
                    });
                }
            }
            if let Some(high_bound) = opt_high_bound {
                let gi = direction[vi];
                if gi > 0.0 {
                    let xi = x_start[vi];
                    breakpoints.push(CauchyPathPoint {
                        alpha: (high_bound - xi) / gi,
                        variable_index: vi,
                        bound_status: BoundStatus::AtUpper(*high_bound),
                    });
                }
            }
        }
        breakpoints.sort_by(|a, b| {
            a.alpha
                .total_cmp(&b.alpha)
                .then_with(|| a.variable_index.cmp(&b.variable_index))
        });
        breakpoints
    }
    //}}}
    //{{{ fn: bound_statuses
    /// Classifies variables as free or active at the current point.
    ///
    /// # Panics
    ///
    /// Panics if `location` or a supplied `direction` does not have the bound
    /// dimension.
    #[trace_fn]
    pub fn bound_statuses(
        &self,
        location: &Vector,
        direction: Option<&Vector>,
    ) -> Vec<BoundStatus> {
        assert_eq!(location.len(), self.dimension_domain());
        let mut statuses = vec![BoundStatus::Free; self.num_variables];

        if let Some(direction) = direction {
            assert_eq!(direction.len(), self.dimension_domain());

            let mut x_clamped = location.clone();
            self.clamp(&mut x_clamped);

            for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter() {
                if let Some(low_bound) = opt_low_bound {
                    let gi = direction[*variable_index];
                    let xi = x_clamped[*variable_index];
                    if gi < 0.0 && xi == *low_bound {
                        statuses[*variable_index] = BoundStatus::AtLower(*low_bound);
                        continue;
                    }
                }
                if let Some(high_bound) = opt_high_bound {
                    let gi = direction[*variable_index];
                    let xi = x_clamped[*variable_index];
                    if gi > 0.0 && xi == *high_bound {
                        statuses[*variable_index] = BoundStatus::AtUpper(*high_bound);
                    }
                }
            }
        } else {
            for (variable_index, (opt_low_bound, opt_high_bound)) in self.bounds.iter() {
                let xi = location[*variable_index];
                if let Some(low_bound) = opt_low_bound {
                    if xi <= *low_bound {
                        statuses[*variable_index] = BoundStatus::AtLower(*low_bound);
                        continue;
                    }
                }
                if let Some(high_bound) = opt_high_bound {
                    if xi >= *high_bound {
                        statuses[*variable_index] = BoundStatus::AtUpper(*high_bound);
                    }
                }
            }
        }

        statuses
    }
    //}}}
    //{{{ fn active_signature
    /// Builds a stable signature of the bounds active at `x`.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have the bound dimension.
    #[trace_fn]
    pub fn active_signature(
        &self,
        x: &Vector,
    ) -> BoundSignature {
        assert_eq!(
            x.len(),
            self.dimension_domain(),
            "point dimension must match bound dimension"
        );
        let mut sig = Vec::<(usize, BoundSide)>::with_capacity(self.bounds.len());

        for (&idx, (lower, upper)) in &self.bounds {
            let xi = x[idx];

            if lower.is_some_and(|lower| xi <= lower) {
                sig.push((idx, BoundSide::Lower));
            } else if upper.is_some_and(|upper| xi >= upper) {
                sig.push((idx, BoundSide::Upper));
            }
        }

        sig.sort_unstable_by_key(|(idx, _)| *idx);
        BoundSignature(sig.into_boxed_slice())
    }
    //}}}
    //{{{ fn: mask_gradient_in_place
    /// Zeros gradient components that cannot move into the feasible region.
    ///
    /// # Panics
    ///
    /// Panics if either vector does not have the bound dimension.
    #[trace_fn]
    pub fn mask_gradient_in_place(
        &self,
        x: &Vector,
        grad_f: &mut Vector,
    ) {
        assert_eq!(
            x.len(),
            self.dimension_domain(),
            "point dimension must match bound dimension"
        );
        assert_eq!(
            grad_f.len(),
            self.dimension_domain(),
            "gradient dimension must match bound dimension"
        );
        for (&idx, (lower, upper)) in &self.bounds {
            let xi = x[idx];

            if lower.is_some_and(|lower| xi <= lower) || upper.is_some_and(|upper| xi >= upper) {
                (*grad_f)[idx] = 0.0;
            }
        }
    }
    //}}}
    //{{{ fn: masked_gradient
    /// Returns a gradient with active-bound components masked.
    ///
    /// # Panics
    ///
    /// Panics if either vector does not have the bound dimension.
    #[trace_fn]
    pub fn masked_gradient(
        &self,
        x: &Vector,
        grad: &Vector,
    ) -> Vector {
        let mut out = grad.clone();
        self.mask_gradient_in_place(x, &mut out);
        return out;
    }

    //}}}
    //{{{ fn: minimum_distance
    /// Returns the smallest distance from `x` to any bound.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have the bound dimension.
    #[trace_fn]
    pub fn minimum_distance(
        &self,
        x: &Vector,
    ) -> f64 {
        assert_eq!(
            x.len(),
            self.dimension_domain(),
            "point dimension must match bound dimension"
        );
        let mut min_dist = f64::MAX;
        for (idx, (opt_lower, opt_upper)) in self.bounds.iter() {
            let xi = x[*idx];
            let mut lower_dist = f64::MAX;
            let mut upper_dist = f64::MAX;

            if let Some(lower) = opt_lower {
                lower_dist = xi - lower
            }

            if let Some(upper) = opt_upper {
                upper_dist = upper - xi;
            }

            min_dist = min_dist.min(f64::min(lower_dist, upper_dist));
        }
        min_dist
    }
    //}}}
    //{{{ fn: all_distances
    /// Returns each variable's distance to its nearest bound.
    ///
    /// # Panics
    ///
    /// Panics if `x` does not have the bound dimension.
    #[trace_fn]
    pub fn all_distances(
        &self,
        x: &Vector,
    ) -> Vector {
        assert_eq!(
            x.len(),
            self.dimension_domain(),
            "point dimension must match bound dimension"
        );
        let mut distances = Vector::from_value_vec(f64::INFINITY, x.len(), VecType::Col);
        for (idx, (opt_lower, opt_upper)) in self.bounds.iter() {
            let xi = x[*idx];
            let mut lower_dist = f64::MAX;
            let mut upper_dist = f64::MAX;

            if let Some(lower) = opt_lower {
                lower_dist = xi - lower
            }

            if let Some(upper) = opt_upper {
                upper_dist = upper - xi;
            }

            distances[*idx] = lower_dist.min(upper_dist);
        }
        distances
    }
    //}}}
}
//}}}
//{{{ impl: RealVectorFn for BoundConstraints
impl crate::DifferentiableFn for BoundConstraints {
    type Input = Vector;
    type Output = Vector;
    type Derivative = Matrix;
    #[trace_fn]
    fn dimension_domain(&self) -> usize {
        self.num_variables
    }

    #[trace_fn]
    fn dimension_range(&self) -> usize {
        self.num_inequality_constraints()
    }

    #[trace_fn]
    fn eval(
        &mut self,
        x: &Vector,
    ) -> Vector {
        assert_eq!(x.len(), self.dimension_domain());
        let mut val = Vector::zeros_vec(self.dimension_range(), VecType::Col);

        let mut constraint_index = 0;
        for (variable_index, (opt_lower, opt_upper)) in self.bounds.iter() {
            let xi = x[*variable_index];
            if let Some(lower) = opt_lower {
                val[constraint_index] = lower - xi;
                constraint_index += 1;
            }

            if let Some(upper) = opt_upper {
                val[constraint_index] = xi - upper;
                constraint_index += 1;
            }
        }
        val
    }

    #[trace_fn]
    fn derivative(
        &mut self,
        x: &Vector,
    ) -> Matrix {
        assert_eq!(x.len(), self.dimension_domain());
        let mut val = Matrix::zeros(self.dimension_domain(), self.dimension_range());
        let mut constraint_index = 0;
        for (variable_index, (opt_lower, opt_upper)) in self.bounds.iter() {
            if opt_lower.is_some() {
                val[(*variable_index, constraint_index)] = -1.0;
                constraint_index += 1;
            }

            if opt_upper.is_some() {
                val[(*variable_index, constraint_index)] = 1.0;
                constraint_index += 1;
            }
        }
        val
    }
}
//}}}
