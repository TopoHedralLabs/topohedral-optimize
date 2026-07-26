//! Factory for line-search implementations.
//!
//! It hides algorithm-specific state behind the common line-search trait.
//--------------------------------------------------------------------------------------------------

//{{{ crate imports
use super::common::*;
use super::nocedal;
use super::thuente;
use crate::{RealFn1, ValidationError};
//}}}
//{{{ std imports
//}}}
//{{{ dep imports
use topohedral_tracing::*;
//}}}
//--------------------------------------------------------------------------------------------------

//{{{ enum: Method
/// Selects a line-search algorithm.
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub enum Method {
    /// More-Thuente line search.
    Thuente(thuente::Options),
    /// Nocedal line search.
    Nocedal(nocedal::Options),
}
//}}}
impl Method {
    /// Validates the selected algorithm's options.
    ///
    /// # Errors
    ///
    /// Returns [`ValidationError`] if any option is invalid.
    pub fn validate(&self) -> Result<(), ValidationError> {
        match self {
            Self::Thuente(options) => options.validate(),
            Self::Nocedal(options) => options.validate(),
        }
    }
}
//{{{ fun: create
#[trace_fn]
pub fn create<'a, F: RealFn1 + 'a>(
    fcn: F,
    method: Method,
) -> Box<dyn LineSearch<Function = F> + 'a> {
    match method {
        Method::Thuente(opts) => Box::new(thuente::Thuente::new(fcn, opts)),
        Method::Nocedal(opts) => Box::new(nocedal::Nocedal::new(fcn, opts)),
    }
}
//}}}
