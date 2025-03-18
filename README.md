[![Build](https://github.com/wboayue/robomath/workflows/test/badge.svg)](https://github.com/wboayue/robomath/actions/workflows/test.yml)
[![License:MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/robomath.svg)](https://crates.io/crates/robomath)
[![Documentation](https://img.shields.io/badge/Documentation-green.svg)](https://docs.rs/robomath/latest/robomath/)
[![Coverage Status](https://coveralls.io/repos/github/wboayue/robomath/badge.svg?branch=main)](https://coveralls.io/github/wboayue/robomath?branch=main&cb=1)

# Robomath

A lightweight, efficient, and generic mathematics library for 3D applications, with a focus on robotics, games, and simulation.

## Features

- 🧮 **Generic Vectors**: `Vec2<T>` and `Vec3<T>` supporting various numeric types with operations like addition, subtraction, multiplication, division, dot product, cross product, clamping, magnitude, and checking for finiteness.
- 🔄 **Quaternions**: Full-featured quaternion implementation for 3D rotations, including Euler angle conversion (both yaw-pitch-roll and roll-pitch-yaw), rotation matrices, Gibbs vectors, and numerically stable methods.
- 📐 **3x3 Matrices**: Matrix operations for linear algebra and transformations, including transpose, determinant, trace, skew-symmetric matrices, outer products, scalar multiplication, and addition.
- 🔀 **Transformations**: Quaternion and matrix-based transformations for 3D rotations, with support for inertial-to-body and body-to-inertial frame conversions.
- 🧪 **Comprehensive Tests**: Well-tested with high test coverage (see coverage badge).

## Installation

Add this to your `Cargo.toml`:

```toml
[dependencies]
robomath = "0.1.0"
```

Check the latest version on crates.io.

## Usage Examples

### Vectors

```rust
use robomath::{vec2, vec3, Vec2, Vec3};

// Create vectors
let v1 = vec3(1.0, 2.0, 3.0);
let v2 = vec3(4.0, 5.0, 6.0);

// Vector operations
let sum = v1 + v2;  // [5.0, 7.0, 9.0]
let diff = v2 - v1; // [3.0, 3.0, 3.0]
let product = v1 * v2; // Element-wise: [4.0, 10.0, 18.0]

// Scalar operations
let scaled = 2.0 * v1; // [2.0, 4.0, 6.0]
let divided = v2 / 2.0; // [2.0, 2.5, 3.0]

// Dot and cross products
let dot = v1.dot(v2); // 1*4 + 2*5 + 3*6 = 32.0
let cross = v1.cross(v2); // [-3.0, 6.0, -3.0]

// Check if finite
let finite = v1.is_finite(); // true
let infinite_vec = vec3(f32::INFINITY, 2.0, 3.0);
let not_finite = infinite_vec.is_finite(); // false

// Vector methods
let length_squared = v1.magnitude_squared(); // 1² + 2² + 3² = 14.0
let length = v1.magnitude(); // √14 ≈ 3.74
let clamped = v1.clamp(0.0, 2.0); // [1.0, 2.0, 2.0]

// 2D vectors work similarly
let v2d = vec2(3.0, 4.0);
let v2d_neg = -v2d; // [-3.0, -4.0]
```

### Quaternions

```rust
use robomath::{Quaternion, vec3};
use std::f32::consts::PI;

// Create a quaternion
let q_identity = Quaternion::identity(); // [1.0, 0.0, 0.0, 0.0]
let q_custom = Quaternion::new(0.707, 0.0, 0.707, 0.0); // w, x, y, z

// Create from Euler angles (yaw, pitch, roll)
let q_from_euler = Quaternion::from_euler(PI/4.0, 0.0, 0.0); // 45° yaw rotation

// Operations
let q1 = Quaternion::new(0.0, 1.0, 0.0, 0.0); // Pure quaternion i
let q2 = Quaternion::new(0.0, 0.0, 1.0, 0.0); // Pure quaternion j
let product = q1 * q2; // Hamilton product, should be k = [0.0, 0.0, 0.0, 1.0]

let conjugate = q_custom.conjugate(); // Conjugate
let inverse = q_custom.inverse(); // Inverse (same as conjugate for unit quaternions)

// Extract individual Euler angles
let yaw = q_from_euler.yaw(); // ≈ PI/4 radians
let pitch = q_from_euler.pitch(); // ≈ 0.0 radians
let roll = q_from_euler.roll(); // ≈ 0.0 radians

// Extract Euler angles as a vector
let euler = q_from_euler.to_euler(); // Returns Vec3 with [yaw, pitch, roll]
let rpy = q_from_euler.to_euler_rpy(); // Returns Vec3 with [roll, pitch, yaw]

// Convert to Gibbs vector
let gibbs = q_from_euler.to_gibbs_vector(); // Vec3 representing Rodrigues parameters

// Access components
println!("Quaternion: {} {} {} {}", q_custom.w, q_custom.x, q_custom.y, q_custom.z);
```

### 3x3 Matrices

```rust
use robomath::{Mat3x3, vec3};

// Create a matrix
let identity = Mat3x3::identity();
let zeros = Mat3x3::zeros();
let custom = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);

// Matrix operations
let transposed = custom.transpose();
let determinant = custom.determinant();
let trace = custom.trace();

// Element access (row, column)
let element = custom[[1, 2]]; // Element at row 1, column 2 (6.0)

// Advanced operations
let v = vec3(1.0, 2.0, 3.0);
let skew = Mat3x3::skew_symmetric(v); // Skew-symmetric matrix from vector
let outer = Mat3x3::outer_product(v, v); // Outer product
```

### Transformations

```rust
use robomath::{Quaternion, vec3};
use std::f32::consts::PI;

// Create a rotation quaternion (90° around Y axis)
let rotation = Quaternion::from_euler(0.0, PI/2.0, 0.0);

// Convert to rotation matrix
let rotation_matrix = rotation.rotation_matrix_i_wrt_b();

// Point to rotate
let point = vec3(1.0, 0.0, 0.0);

// Apply rotation using the quaternion
// First, create a pure quaternion from the point
let pure_q = Quaternion::new(0.0, point.x, point.y, point.z);
// Rotate using quaternion multiplication: q * p * q^-1
let rotated_pure_q = rotation * pure_q * rotation.inverse();
// Extract the vector part
let rotated_point = vec3(rotated_pure_q.x, rotated_pure_q.y, rotated_pure_q.z);
// rotated_point should be approximately [0.0, 0.0, -1.0]
```

## Advanced Features

### Generic Vector Types

The vector types `Vec2<T>` and `Vec3<T>` are generic over their component type, allowing you to use them with various numeric types:

```rust
let float_vec = vec3(1.0, 2.0, 3.0); // f32
let int_vec = vec3(1, 2, 3); // i32
let uint_vec = vec3(1u32, 2u32, 3u32); // u32
```

## Limitations

- Some quaternion methods (e.g., `inverse`) assume unit quaternions. Non-unit quaternions may require normalization first.
- Performance optimizations like SIMD are not currently implemented, prioritizing simplicity.
- Floating-point precision may affect operations with very large or small values. For example, `to_gibbs_vector` uses a large value (1e20) when `w=0` to approximate infinity.
- Euler angle conversions may encounter gimbal lock in certain configurations (e.g., pitch near ±90 degrees).
- No support for matrix inversion or higher-dimensional matrices (e.g., 4x4 matrices for homogeneous transformations).

### Numerical Stability

The quaternion implementation includes numerically stable methods for converting between rotation representations, handling edge cases gracefully. For example:

- `from_rotation_matrix`: Avoids singularities by selecting the largest component (w, x, y, or z) to compute first, preventing division by zero.
- Euler angle extraction (`yaw`, `pitch`, `roll`): Uses `atan2f` and `asinf` to handle edge cases like gimbal lock, though users should be aware of potential discontinuities.

## License

This library is licensed under the MIT License - see the LICENSE file for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
