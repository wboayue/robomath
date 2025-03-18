use super::super::*;
use std::f32::consts::PI;
use std::f32::EPSILON;

// Helper function for approximate equality of f32 values
fn assert_float_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "Expected {} to be approximately equal to {}, difference: {}",
        a,
        b,
        (a - b).abs()
    );
}

// Helper function for quaternion equality
fn assert_quaternion_eq(q1: &Quaternion, q2: &Quaternion, epsilon: f32) {
    assert_float_eq(q1.w, q2.w, epsilon);
    assert_float_eq(q1.x, q2.x, epsilon);
    assert_float_eq(q1.y, q2.y, epsilon);
    assert_float_eq(q1.z, q2.z, epsilon);
}

#[test]
fn test_new() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert_eq!(q.w, 1.0);
    assert_eq!(q.x, 2.0);
    assert_eq!(q.y, 3.0);
    assert_eq!(q.z, 4.0);
}

#[test]
fn test_identity() {
    let q = Quaternion::identity();
    assert_quaternion_eq(&q, &Quaternion::new(1.0, 0.0, 0.0, 0.0), 1e-6);
}

#[test]
fn test_default() {
    let q = Quaternion::default();
    assert_quaternion_eq(&q, &Quaternion::identity(), 1e-6);
}

#[test]
fn test_conjugate() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let conj = q.conjugate();
    assert_quaternion_eq(&conj, &Quaternion::new(1.0, -2.0, -3.0, -4.0), 1e-6);
}

#[test]
fn test_inverse() {
    // Test inverse for a unit quaternion
    let q = Quaternion::from_euler(0.0, 0.0, PI / 2.0); // 90-degree yaw
    let inv = q.inverse();
    let expected = q.conjugate(); // Since q is unit quaternion
    assert_quaternion_eq(&inv, &expected, 1e-6);
}

#[test]
fn test_multiplication() {
    // Test quaternion multiplication with identity
    let q1 = Quaternion::identity();
    let q2 = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let result = q1 * q2;
    assert_quaternion_eq(&result, &q2, 1e-6);

    // Test with two pure quaternions
    let q1 = Quaternion::new(0.0, 1.0, 0.0, 0.0); // i
    let q2 = Quaternion::new(0.0, 0.0, 1.0, 0.0); // j
    let result = q1 * q2;
    // i * j = k
    assert_quaternion_eq(&result, &Quaternion::new(0.0, 0.0, 0.0, 1.0), 1e-6);
}

#[test]
fn test_from_euler() {
    // Test quaternion from Euler angles (yaw = 90 degrees, pitch = 0, roll = 0)
    let q = Quaternion::from_euler(PI / 2.0, 0.0, 0.0);
    let expected = Quaternion::new((PI / 4.0).cos(), 0.0, 0.0, (PI / 4.0).sin());
    assert_quaternion_eq(&q, &expected, 1e-6);
}

#[test]
fn test_to_euler() {
    // Test Euler angle extraction for a 90-degree yaw
    let q = Quaternion::from_euler(PI / 2.0, 0.0, 0.0);
    let euler = q.to_euler(); // (roll, pitch, yaw)
    assert_float_eq(euler.x, 0.0, 1e-6); // roll
    assert_float_eq(euler.y, 0.0, 1e-6); // pitch
    assert_float_eq(euler.z, PI / 2.0, 1e-6); // yaw
}

#[test]
fn test_roll() {
    // Test roll extraction for a 90-degree roll
    let q = Quaternion::from_euler(0.0, 0.0, PI / 2.0);
    let roll = q.roll();
    assert_float_eq(roll, PI / 2.0, 1e-6);
}

#[test]
fn test_pitch() {
    // Test pitch extraction for a 90-degree pitch
    let q = Quaternion::from_euler(0.0, PI / 2.0, 0.0);
    let pitch = q.pitch();
    assert_float_eq(pitch, PI / 2.0, 1e-3);
}

#[test]
fn test_yaw() {
    // Test yaw extraction for a 90-degree yaw
    let q = Quaternion::from_euler(PI / 2.0, 0.0, 0.0);
    let yaw = q.yaw();
    assert_float_eq(yaw, PI / 2.0, 1e-6);
}

#[test]
fn test_from_rotation_matrix() {
    // Test conversion from a rotation matrix (90-degree yaw)
    let cos_90 = (PI / 2.0).cos();
    let sin_90 = (PI / 2.0).sin();
    let mat = Mat3x3::new([cos_90, -sin_90, 0.0, sin_90, cos_90, 0.0, 0.0, 0.0, 1.0]);
    let q = Quaternion::from_rotation_matrix(&mat);
    let expected = Quaternion::new((PI / 4.0).cos(), 0.0, 0.0, (PI / 4.0).sin());
    assert_quaternion_eq(&q, &expected, 1e-6);
}

#[test]
fn test_rotation_matrix_i_wrt_b() {
    // Test rotation matrix for a 90-degree yaw
    let q = Quaternion::from_euler(PI / 2.0, 0.0, 0.0);
    let mat = q.rotation_matrix_i_wrt_b();
    let cos_90 = (PI / 2.0).cos();
    let sin_90 = (PI / 2.0).sin();
    let expected = Mat3x3::new([cos_90, -sin_90, 0.0, sin_90, cos_90, 0.0, 0.0, 0.0, 1.0]);
    for i in 0..9 {
        assert_float_eq(mat.data[i], expected.data[i], 1e-6);
    }
}

#[test]
fn test_rotation_matrix_b_wrt_i() {
    // Test body-to-inertial rotation matrix (transpose of i_wrt_b)
    let q = Quaternion::from_euler(PI / 2.0, 0.0, 0.0);
    let mat = q.rotation_matrix_b_wrt_i();
    let cos_90 = (PI / 2.0).cos();
    let sin_90 = (PI / 2.0).sin();
    let expected = Mat3x3::new([cos_90, sin_90, 0.0, -sin_90, cos_90, 0.0, 0.0, 0.0, 1.0]);
    for i in 0..9 {
        assert_float_eq(mat.data[i], expected.data[i], 1e-6);
    }
}

#[test]
fn test_to_gibbs_vector() {
    // Test conversion to Gibbs vector
    let q = Quaternion::new(1.0, 1.0, 0.0, 0.0);
    let gibbs = q.to_gibbs_vector();
    assert_float_eq(gibbs.x, 1.0, 1e-6);
    assert_float_eq(gibbs.y, 0.0, 1e-6);
    assert_float_eq(gibbs.z, 0.0, 1e-6);
}

#[test]
fn test_to_string() {
    let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    let s = q.to_string();
    assert_eq!(s, "1.000 2.000 3.000 4.000");
}

#[test]
fn test_from_rotation_matrix_edge_cases() {
    // Test with identity matrix
    let mat = Mat3x3::identity();
    let q = Quaternion::from_rotation_matrix(&mat);
    assert_quaternion_eq(&q, &Quaternion::identity(), 1e-6);

    // Test with rotation matrix where q0 is largest
    // 180-degree rotation about z-axis
    let mat = Mat3x3::new([-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0]);
    let q = Quaternion::from_rotation_matrix(&mat);
    // Expected quaternion: (0, 0, 0, 1) or (0, 0, 0, -1)
    // But our code chooses the one with positive w
    assert!(
        (q.w.abs() < 1e-6 && (q.z - 1.0).abs() < 1e-6)
            || (q.w.abs() < 1e-6 && (q.z + 1.0).abs() < 1e-6)
    );

    // Test with rotation matrix where q1 is largest
    // 180-degree rotation about x-axis
    let mat = Mat3x3::new([1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, -1.0]);
    let q = Quaternion::from_rotation_matrix(&mat);
    // Expected quaternion: (0, 1, 0, 0) or (0, -1, 0, 0)
    assert!(
        (q.w.abs() < 1e-6 && (q.x - 1.0).abs() < 1e-6)
            || (q.w.abs() < 1e-6 && (q.x + 1.0).abs() < 1e-6)
    );

    // Test with rotation matrix where q2 is largest
    // 180-degree rotation about y-axis
    let mat = Mat3x3::new([-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0]);
    let q = Quaternion::from_rotation_matrix(&mat);
    // Expected quaternion: (0, 0, 1, 0) or (0, 0, -1, 0)
    assert!(
        (q.w.abs() < 1e-6 && (q.y - 1.0).abs() < 1e-6)
            || (q.w.abs() < 1e-6 && (q.y + 1.0).abs() < 1e-6)
    );
}

#[test]
fn test_to_gibbs_vector_edge_cases() {
    // Test normal case
    let q = Quaternion::new(2.0, 1.0, 2.0, 3.0);
    let gibbs = q.to_gibbs_vector();
    assert_float_eq(gibbs.x, 0.5, EPSILON); // 1.0/2.0
    assert_float_eq(gibbs.y, 1.0, EPSILON); // 2.0/2.0
    assert_float_eq(gibbs.z, 1.5, EPSILON); // 3.0/2.0

    // Test with w = 0 (180-degree rotation, Gibbs vector goes to infinity)
    let q = Quaternion::new(0.0, 1.0, 0.0, 0.0);
    let gibbs = q.to_gibbs_vector();
    assert!(gibbs.x.abs() > 1e19); // Should be very large
    assert_float_eq(gibbs.y, 0.0, EPSILON);
    assert_float_eq(gibbs.z, 0.0, EPSILON);
}

#[test]
fn test_euler_conversions() {
    // Test all three Euler angles together
    let yaw = PI / 4.0; // 45 degrees
    let pitch = PI / 6.0; // 30 degrees
    let roll = PI / 3.0; // 60 degrees

    let q = Quaternion::from_euler(yaw, pitch, roll);
    let euler = q.to_euler();

    assert_float_eq(euler.z, yaw, 1e-5); // yaw
    assert_float_eq(euler.y, pitch, 1e-5); // pitch
    assert_float_eq(euler.x, roll, 1e-5); // roll
}

#[test]
fn test_magnitude() {
    let q = Quaternion::new(1.0, 0.0, 0.0, 0.0);
    assert!((q.magnitude() - 1.0).abs() < 1e-5);

    let q2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert!((q2.magnitude() - 5.477225).abs() < 1e-5);
}

#[test]
fn test_normalize() {
    let q = Quaternion::new(0.0, 2.0, 0.0, 0.0);
    let norm = q.normalize();
    assert!((norm.magnitude() - 1.0).abs() < 1e-5);
    assert!((norm.x - 1.0).abs() < 1e-5);

    let q_zero = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    assert_eq!(q_zero.normalize(), Quaternion::identity());
}

// #[test]
// fn test_inverse_non_unit() {
//     let q = Quaternion::new(1.0, 2.0, 3.0, 4.0); // Magnitude ≈ 5.477
//     let inv = q.inverse(); // Should be conjugate / mag^2
//     let expected = Quaternion::new(1.0 / 30.0, -2.0 / 30.0, -3.0 / 30.0, -4.0 / 30.0);
//     assert_quaternion_eq(&inv, &expected, 1e-6);
// }

#[test]
fn test_from_euler_gimbal_lock() {
    let q = Quaternion::from_euler(0.0, PI / 2.0 - 0.0001, 0.0); // Near 90° pitch
    let euler = q.to_euler();
    assert_float_eq(euler.y, PI / 2.0 - 0.0001, 1e-3);
}

#[test]
fn test_to_euler_singularity() {
    let q = Quaternion::from_euler(PI / 4.0, PI / 2.0, PI / 3.0); // Pitch = 90°
    let euler = q.to_euler();
    assert_float_eq(euler.y, PI / 2.0, 1e-3); // Pitch should be exact
                                              // Yaw and roll combine; exact values depend on implementation
}

#[test]
fn test_rotation_matrix_combined() {
    let q = Quaternion::from_euler(PI / 2.0, PI / 4.0, 0.0);
    let mat = q.rotation_matrix_i_wrt_b();
    let mat_t = q.rotation_matrix_b_wrt_i();
    // Verify orthogonality and inverse relationship
    let product = mat * mat_t;
    for i in 0..9 {
        assert_float_eq(product.data[i], if i % 4 == 0 { 1.0 } else { 0.0 }, 1e-6);
    }
}

#[test]
fn test_is_finite() {
    let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    assert!(q1.is_finite());
    let q2 = Quaternion::new(f32::INFINITY, 0.0, 0.0, 0.0);
    assert!(!q2.is_finite());
    let q3 = Quaternion::new(0.0, f32::NAN, 0.0, 0.0);
    assert!(!q3.is_finite());
}

#[test]
fn test_normalize_edge_cases() {
    let q = Quaternion::new(1e-20, -1e-20, 0.0, 0.0); // Near-zero magnitude
    let norm = q.normalize();
    assert_quaternion_eq(&norm, &Quaternion::identity(), 1e-6);
}

#[test]
fn test_magnitude_edge_cases() {
    let q = Quaternion::new(f32::INFINITY, 0.0, 0.0, 0.0);
    assert_eq!(q.magnitude(), f32::INFINITY);
    let q = Quaternion::new(f32::NAN, 0.0, 0.0, 0.0);
    assert!(q.magnitude().is_nan());
}

#[test]
fn test_numerical_stability() {
    let q = Quaternion::new(1e-10, 1e-10, 1e-10, 1e-10);
    let norm = q.normalize();
    assert!((norm.magnitude() - 1.0).abs() < 1e-5);
}
