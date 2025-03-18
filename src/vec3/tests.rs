use core::f32::EPSILON;

use super::*;

// Helper function to compare floating-point values
fn assert_float_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() <= epsilon,
        "Expected {} to be approximately equal to {}, but difference is greater than epsilon {}",
        a,
        b,
        epsilon
    );
}

#[test]
fn test_default() {
    let v: Vec3<f32> = Vec3::default();
    assert_eq!(v.x, 0.0);
    assert_eq!(v.y, 0.0);
    assert_eq!(v.z, 0.0);

    let v_int: Vec3<i32> = Vec3::default();
    assert_eq!(v_int.x, 0);
    assert_eq!(v_int.y, 0);
    assert_eq!(v_int.z, 0);
}

#[test]
fn test_addition() {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(4.0, 5.0, 6.0);
    let result = v1 + v2;
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 7.0);
    assert_eq!(result.z, 9.0);

    // Test with integers
    let v1_int = vec3(1, 2, 3);
    let v2_int = vec3(4, 5, 6);
    let result_int = v1_int + v2_int;
    assert_eq!(result_int.x, 5);
    assert_eq!(result_int.y, 7);
    assert_eq!(result_int.z, 9);
}

#[test]
fn test_subtraction() {
    let v1 = vec3(5.0, 7.0, 9.0);
    let v2 = vec3(1.0, 2.0, 3.0);
    let result = v1 - v2;
    assert_eq!(result.x, 4.0);
    assert_eq!(result.y, 5.0);
    assert_eq!(result.z, 6.0);

    // Test with integers
    let v1_int = vec3(5, 7, 9);
    let v2_int = vec3(1, 2, 3);
    let result_int = v1_int - v2_int;
    assert_eq!(result_int.x, 4);
    assert_eq!(result_int.y, 5);
    assert_eq!(result_int.z, 6);
}

#[test]
fn test_multiplication_vec3() {
    let v1 = vec3(2.0, 3.0, 4.0);
    let v2 = vec3(5.0, 6.0, 7.0);
    let result = v1 * v2;
    assert_eq!(result.x, 10.0);
    assert_eq!(result.y, 18.0);
    assert_eq!(result.z, 28.0);

    // Test with integers
    let v1_int = vec3(2, 3, 4);
    let v2_int = vec3(5, 6, 7);
    let result_int = v1_int * v2_int;
    assert_eq!(result_int.x, 10);
    assert_eq!(result_int.y, 18);
    assert_eq!(result_int.z, 28);
}

#[test]
fn test_scalar_multiplication() {
    let v = vec3(1.0, 2.0, 3.0);
    let scalar = 2.0;
    let result = scalar * v;
    assert_eq!(result.x, 2.0);
    assert_eq!(result.y, 4.0);
    assert_eq!(result.z, 6.0);
}

#[test]
fn test_division_by_scalar() {
    let v = vec3(4.0, 6.0, 8.0);
    let scalar = 2.0;
    let result = v / scalar;
    assert_eq!(result.x, 2.0);
    assert_eq!(result.y, 3.0);
    assert_eq!(result.z, 4.0);

    // Test with integers
    let v_int = vec3(4, 6, 8);
    let scalar_int = 2;
    let result_int = v_int / scalar_int;
    assert_eq!(result_int.x, 2);
    assert_eq!(result_int.y, 3);
    assert_eq!(result_int.z, 4);
}

#[test]
fn test_clone_and_copy() {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = v1; // Copy
    assert_eq!(v1.x, v2.x);
    assert_eq!(v1.y, v2.y);
    assert_eq!(v1.z, v2.z);

    let v3 = v1.clone(); // Explicit clone
    assert_eq!(v1.x, v3.x);
    assert_eq!(v1.y, v3.y);
    assert_eq!(v1.z, v3.z);
}

#[test]
fn test_debug_output() {
    let v = vec3(1.0, 2.0, 3.0);
    let debug_str = format!("{:?}", v);
    assert!(debug_str.contains("Vec3"));
    assert!(debug_str.contains("x: 1"));
    assert!(debug_str.contains("y: 2"));
    assert!(debug_str.contains("z: 3"));
}

#[test]
fn test_clamp() {
    // Test clamping within range
    let v = vec3(1.0, 2.0, 3.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 1.0);
    assert_eq!(result.y, 2.0);
    assert_eq!(result.z, 3.0);

    // Test clamping below minimum
    let v = vec3(-1.0, -2.0, -3.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 0.0);
    assert_eq!(result.z, 0.0);

    // Test clamping above maximum
    let v = vec3(6.0, 7.0, 8.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 5.0);
    assert_eq!(result.z, 5.0);

    // Test mixed clamping
    let v = vec3(-1.0, 3.0, 8.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 3.0);
    assert_eq!(result.z, 5.0);
}

#[test]
fn test_magnitude_squared() {
    // Test with positive values
    let v = vec3(3.0, 4.0, 0.0);
    assert_eq!(v.magnitude_squared(), 25.0); // 3² + 4² + 0² = 9 + 16 + 0 = 25

    // Test with negative values
    let v = vec3(-3.0, -4.0, 0.0);
    assert_eq!(v.magnitude_squared(), 25.0); // (-3)² + (-4)² + 0² = 9 + 16 + 0 = 25

    // Test with mixed values
    let v = vec3(1.0, -2.0, 2.0);
    assert_eq!(v.magnitude_squared(), 9.0); // 1² + (-2)² + 2² = 1 + 4 + 4 = 9

    // Test with zero vector
    let v = vec3(0.0, 0.0, 0.0);
    assert_eq!(v.magnitude_squared(), 0.0);

    // Test with integers
    let v = vec3(3, 4, 0);
    assert_eq!(v.magnitude_squared(), 25); // 3² + 4² + 0² = 9 + 16 + 0 = 25
}

#[test]
fn test_magnitude() {
    // Test with Pythagorean triple
    let v = vec3(3.0, 4.0, 0.0);
    assert_float_eq(v.magnitude(), 5.0, EPSILON); // √(3² + 4² + 0²) = √25 = 5

    // Test with negative values
    let v = vec3(-3.0, -4.0, 0.0);
    assert_float_eq(v.magnitude(), 5.0, EPSILON); // √((-3)² + (-4)² + 0²) = √25 = 5

    // Test with unit vector along x-axis
    let v = vec3(1.0, 0.0, 0.0);
    assert_float_eq(v.magnitude(), 1.0, EPSILON);

    // Test with zero vector
    let v = vec3(0.0, 0.0, 0.0);
    assert_float_eq(v.magnitude(), 0.0, EPSILON);

    // Test with 3D case
    let v = vec3(2.0, 3.0, 6.0);
    assert_float_eq(v.magnitude(), 7.0, EPSILON); // √(2² + 3² + 6²) = √(4 + 9 + 36) = √49 = 7
}

#[test]
fn test_dot() {
    let v1 = vec3(1.0, 2.0, 3.0);
    let v2 = vec3(4.0, 5.0, 6.0);
    assert_eq!(v1.dot(v2), 32.0);

    let v3 = vec3(1.0, 0.0, 0.0);
    let v4 = vec3(0.0, 1.0, 0.0);
    assert_eq!(v3.dot(v4), 0.0);
}

#[test]
fn test_cross() {
    let v1 = vec3(1.0, 0.0, 0.0);
    let v2 = vec3(0.0, 1.0, 0.0);
    assert_eq!(v1.cross(v2), vec3(0.0, 0.0, 1.0));

    let v3 = vec3(1.0, 2.0, 3.0);
    let v4 = vec3(2.0, 4.0, 6.0);
    assert_eq!(v3.cross(v4), vec3(0.0, 0.0, 0.0));
}

#[test]
fn test_division_by_zero() {
    let v: Vec3<f32> = vec3(1.0, 2.0, 3.0);
    assert!(v.is_finite());

    let result = v / 0.0;

    assert!(result.x.is_infinite());
    assert!(result.y.is_infinite());
    assert!(result.z.is_infinite());
}

#[test]
fn test_right_scalar_multiplication() {
    let v = vec3(1.0, 2.0, 3.0);
    let result = v * 2.0;
    assert_eq!(result, vec3(2.0, 4.0, 6.0));
}

#[test]
fn test_multiplication_edge_cases() {
    // Negative numbers
    let v1 = vec3(-2.0, -3.0, -4.0);
    let v2 = vec3(5.0, 6.0, 7.0);
    let result = v1 * v2;
    assert_eq!(result, vec3(-10.0, -18.0, -28.0));

    // Zeros
    let v3 = vec3(0.0, 0.0, 0.0);
    let v4 = vec3(1.0, 2.0, 3.0);
    let result = v3 * v4;
    assert_eq!(result, vec3(0.0, 0.0, 0.0));

    // Mixed positive/negative
    let v5 = vec3(-1.0, 2.0, -3.0);
    let v6 = vec3(4.0, -5.0, 6.0);
    let result = v5 * v6;
    assert_eq!(result, vec3(-4.0, -10.0, -18.0));
}

#[test]
fn test_scalar_multiplication_edge_cases() {
    // Negative scalar
    let v = vec3(1.0, 2.0, 3.0);
    let result = -2.0 * v;
    assert_eq!(result, vec3(-2.0, -4.0, -6.0));

    // Zero scalar
    let result = 0.0 * v;
    assert_eq!(result, vec3(0.0, 0.0, 0.0));

    // Large scalar
    let large = f32::MAX / 4.0; // Avoid overflow
    let result = large * vec3(1.0, 1.0, 1.0);
    assert!(result.x.is_finite() && result.x > 0.0);
}

#[test]
fn test_component_wise_division() {
    let v1 = vec3(4.0, 6.0, 8.0);
    let v2 = vec3(2.0, 3.0, 4.0);
    let result = v1 / v2;
    assert_eq!(result, vec3(2.0, 2.0, 2.0));

    // Negative components
    let v3 = vec3(-4.0, -6.0, -8.0);
    let result = v3 / v2;
    assert_eq!(result, vec3(-2.0, -2.0, -2.0));

    // Mixed positive/negative
    let v4 = vec3(-1.0, 2.0, -3.0);
    let v5 = vec3(4.0, -5.0, 6.0);
    let result = v4 / v5;
    assert_float_eq(result.x, -0.25, EPSILON);
    assert_float_eq(result.y, -0.4, EPSILON);
    assert_float_eq(result.z, -0.5, EPSILON);
}

#[test]
#[should_panic(expected = "attempt to divide by zero")]
fn test_division_by_zero_integer() {
    let v = vec3(1, 2, 3);
    let _ = v / 0; // Should panic for integers
}

#[test]
fn test_division_by_zero_float() {
    let v: Vec3<f64> = vec3(1.0, 2.0, 3.0);
    let result = v / 0.0;
    assert!(result.x.is_infinite() && result.x > 0.0);
    assert!(result.y.is_infinite() && result.y > 0.0);
    assert!(result.z.is_infinite() && result.z > 0.0);

    // Component-wise division by zero
    let v2 = vec3(0.0, 1.0, 0.0);
    let result = v / v2;
    assert!(result.x.is_infinite() && result.x > 0.0);
    assert_eq!(result.y, 2.0);
    assert!(result.z.is_infinite() && result.z > 0.0);
}

#[test]
fn test_division_by_scalar_edge_cases() {
    // Negative scalar
    let v = vec3(4.0, 6.0, 8.0);
    let result = v / -2.0;
    assert_eq!(result, vec3(-2.0, -3.0, -4.0));
}
