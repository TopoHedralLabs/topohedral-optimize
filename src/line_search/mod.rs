//! Short Description of module
//!
//! Longer description of module
//--------------------------------------------------------------------------------------------------

//{{{ crate imports 
//}}}
//{{{ std imports 
//}}}
//{{{ dep imports 
//}}}
//--------------------------------------------------------------------------------------------------


pub trait LineSearcher {
    type Returns;
    type Error;
    fn search(&mut self, phi0: f64, dphi0: f64) ->  Result<Self::Returns, Self::Error>;
}