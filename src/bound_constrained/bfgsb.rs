//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use crate::constraints::{BoundsConstraints, CauchyPathPoint};
use crate::Vector;
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_linalg::VecType::Col;
use topohedral_linalg::VectorOps;
use topohedral_tracing::trace_fn;
//}}}
//--------------------------------------------------------------------------------------------------

struct CauchyPoint
{
    pub cauchy_point: Vector,
    pub cauchy_curvature: Vector,
    pub free_variables: Vec<usize>,
}

#[trace_fn]
fn cauchy_point(
    bounds: &BoundsConstraints,
    x: &Vector,
    g: &Vector,
) -> CauchyPoint
{
    let mut d = -g.clone();
    let (active_set, inactive_set) = bounds.active_and_inactive_sets(x, Some(&d));

    if inactive_set.is_empty()
    {
        return CauchyPoint {
            cauchy_point: x.clone(),
            cauchy_curvature: Vector::zeros_vec(x.len(), Col),
            free_variables: Vec::new(),
        };
    }

    for (vi, _) in active_set
    {
        d[vi] = 0.0;
    }

    let cauchy_path = bounds.cauchy_path(x, &d);

    for i in 1..cauchy_path.len()
    {
        let CauchyPathPoint {
            alpha: alpha1,
            point: x1,
            direction: d1,
            variable_index: idx1,
        } = &cauchy_path[i - 1];

        let CauchyPathPoint {
            alpha: alpha2,
            point: x2,
            direction: d2,
            variable_index: idx2,
        } = &cauchy_path[i];
    }

    todo!()
}
