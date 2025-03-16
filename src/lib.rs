#[cfg(test)]
mod tests;

pub mod mat3x3;
pub mod quaternion;
pub mod vec2;
pub mod vec3;

#[doc(inline)]
pub use mat3x3::*;
#[doc(inline)]
pub use quaternion::*;
#[doc(inline)]
pub use vec2::*;
#[doc(inline)]
pub use vec3::*;
