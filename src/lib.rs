//! # Robomath
//!
//! A lightweight, efficient, and generic mathematics library for 3D applications, with a focus on
//! robotics, games, and simulation. `robomath` provides essential tools for working with vectors,
//! quaternions, and 3x3 matrices, emphasizing simplicity and correctness for real-time applications.
//!
//! ## Key Features
//!
//! - **Generic Vectors**: `Vec2<T>` and `Vec3<T>` supporting various numeric types with operations
//!   like addition, subtraction, multiplication, division, dot product, cross product, and clamping.
//! - **Quaternions**: Full-featured quaternion implementation for 3D rotations, including Euler angle
//!   conversion, rotation matrices, Gibbs vectors, and numerically stable methods.
//! - **3x3 Matrices**: Matrix operations for linear algebra and transformations, including transpose,
//!   determinant, trace, skew-symmetric matrices, and outer products.
//! - **Transformations**: Quaternion and matrix-based transformations for 3D rotations, with support
//!   for inertial-to-body and body-to-inertial frame conversions.
//! - **Comprehensive Tests**: Well-tested with high test coverage for reliability.
//!
//! ## Getting Started
//!
//! Add `robomath` to your project by including it in your `Cargo.toml`:
//!
//! ```toml
//! [dependencies]
//! robomath = "0.1.0"
//! ```
//!
//! Here's a simple example that demonstrates rotating a vector using a quaternion:
//!
//! ```rust
//! use robomath::{Quaternion, vec3, Vec3};
//! use std::f32::consts::PI;
//!
//! // Define a vector to rotate
//! let point = vec3(1.0, 0.0, 0.0);
//!
//! // Create a quaternion for a 90-degree rotation around the Y-axis
//! let rotation = Quaternion::from_euler(0.0, PI/2.0, 0.0);
//!
//! // Convert the point to a pure quaternion
//! let pure_q = Quaternion::new(0.0, point.x, point.y, point.z);
//!
//! // Apply the rotation: q * p * q^-1
//! let rotated_pure_q = rotation * pure_q * rotation.inverse();
//!
//! // Extract the rotated vector
//! let rotated_point = vec3(rotated_pure_q.x, rotated_pure_q.y, rotated_pure_q.z);
//! println!("Rotated point: {:?}", rotated_point); // Should be approximately [0.0, 0.0, -1.0]
//! ```
//!
//! ## Usage
//!
//! The library provides ergonomic implementations of common 3D math operations. For detailed
//! examples, see the [Usage Examples](https://github.com/wboayue/robomath#usage-examples) section
//! in the README.
//!
//! ### Vectors
//!
//! Work with generic `Vec2<T>` and `Vec3<T>` types for 2D and 3D vector operations:
//! - Arithmetic: addition, subtraction, multiplication, division
//! - Methods: dot product, cross product, magnitude, clamping, etc.
//!
//! ### Quaternions
//!
//! Use `Quaternion` for efficient and numerically stable 3D rotations:
//! - Create from Euler angles or rotation matrices
//! - Extract Euler angles, Gibbs vectors, or rotation matrices
//! - Perform quaternion multiplication and conjugation
//!
//! ### Matrices
//!
//! Use `Mat3x3` for linear algebra operations:
//! - Basic operations: transpose, determinant, trace
//! - Advanced operations: skew-symmetric matrices, outer products
//!
//! ## Advanced Features
//!
//! - **Generic Types**: Vectors are generic over their component type `T`, with trait bounds for
//!   arithmetic operations. See the README's [Generic Type Requirements](https://github.com/wboayue/robomath#generic-type-requirements) section.
//! - **Numerical Stability**: Methods like quaternion-to-rotation-matrix conversion are implemented
//!   with numerical stability in mind.
//!
//! ## Limitations
//!
//! - Some quaternion methods (e.g., `inverse`) assume unit quaternions.
//! - Performance optimizations like SIMD are not currently implemented, prioritizing simplicity.
//! - Floating-point precision may affect operations with very large or small values.
//!
//! For more details, see the [Limitations](https://github.com/wboayue/robomath#limitations-and-assumptions)
//! section in the README.
//!
//! ## Contributing
//!
//! Contributions are welcome! Please see the [CONTRIBUTING](https://github.com/wboayue/robomath/blob/main/CONTRIBUTING.md)
//! guide for more information.
//!
//! ## License
//!
//! This library is licensed under the MIT License. See the [LICENSE](https://github.com/wboayue/robomath/blob/main/LICENSE)
//! file for details.

#![cfg_attr(not(test), no_std)]

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
