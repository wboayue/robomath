use super::super::*;
use std::f32::EPSILON;

// Helper function to compare floating-point numbers with an epsilon
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
fn test_new() {
    let data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0];
    let mat = Mat3x3::new(data);
    assert_eq!(mat.data, data);
}

#[test]
fn test_zeros() {
    let mat = Mat3x3::zeros();
    let expected = [0.0; 9];
    assert_eq!(mat.data, expected);
}

#[test]
fn test_identity() {
    let mat = Mat3x3::identity();
    let expected = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
    assert_eq!(mat.data, expected);
}

#[test]
fn test_index() {
    let mat = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_eq!(mat[[0, 0]], 1.0);
    assert_eq!(mat[[0, 1]], 2.0);
    assert_eq!(mat[[0, 2]], 3.0);
    assert_eq!(mat[[1, 0]], 4.0);
    assert_eq!(mat[[1, 1]], 5.0);
    assert_eq!(mat[[1, 2]], 6.0);
    assert_eq!(mat[[2, 0]], 7.0);
    assert_eq!(mat[[2, 1]], 8.0);
    assert_eq!(mat[[2, 2]], 9.0);
}

#[test]
#[should_panic]
fn test_index_out_of_bounds() {
    let mat = Mat3x3::new([0.0; 9]);
    let _ = mat[[3, 0]]; // Should panic
}

#[test]
fn test_transpose() {
    let mat = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let transposed = mat.transpose();
    let expected = [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0];
    assert_eq!(transposed.data, expected);

    // Test that transposing twice gives the original matrix
    let double_transposed = transposed.transpose();
    assert_eq!(double_transposed.data, mat.data);
}

#[test]
fn test_determinant() {
    // Test with identity matrix (determinant should be 1.0)
    let mat = Mat3x3::identity();
    assert_float_eq(mat.determinant(), 1.0, EPSILON);

    // Test with a known matrix
    // Matrix: [1, 2, 3]
    //         [4, 5, 6]
    //         [7, 8, 9]
    // Determinant = 1*(5*9 - 6*8) - 2*(4*9 - 6*7) + 3*(4*8 - 5*7) = 0
    let mat = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_float_eq(mat.determinant(), 0.0, EPSILON);

    // Test with another matrix
    // Matrix: [2, 0, 0]
    //         [0, 3, 0]
    //         [0, 0, 4]
    // Determinant = 2*(3*4 - 0*0) - 0*(0*4 - 0*0) + 0*(0*0 - 3*0) = 24
    let mat = Mat3x3::new([2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0]);
    assert_float_eq(mat.determinant(), 24.0, EPSILON);
}

#[test]
fn test_trace() {
    // Test with identity matrix (trace should be 3.0)
    let mat = Mat3x3::identity();
    assert_float_eq(mat.trace(), 3.0, EPSILON);

    // Test with a known matrix
    // Matrix: [1, 2, 3]
    //         [4, 5, 6]
    //         [7, 8, 9]
    // Trace = 1 + 5 + 9 = 15
    let mat = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert_float_eq(mat.trace(), 15.0, EPSILON);

    // Test with zeros (trace should be 0.0)
    let mat = Mat3x3::zeros();
    assert_float_eq(mat.trace(), 0.0, EPSILON);
}

#[test]
fn test_skew_symmetric() {
    // Test with unit vectors
    let v = vec3(1.0, 0.0, 0.0);
    let skew = Mat3x3::skew_symmetric(v);
    // Expected skew-symmetric matrix for [1,0,0]:
    // [ 0  0  0]
    // [ 0  0 -1]
    // [ 0  1  0]
    let expected = [0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 1.0, 0.0];
    assert_eq!(skew.data, expected);

    let v = vec3(0.0, 1.0, 0.0);
    let skew = Mat3x3::skew_symmetric(v);
    // Expected for [0,1,0]:
    // [ 0  0  1]
    // [ 0  0  0]
    // [-1  0  0]
    let expected = [0.0, 0.0, 1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0];
    assert_eq!(skew.data, expected);

    let v = vec3(0.0, 0.0, 1.0);
    let skew = Mat3x3::skew_symmetric(v);
    // Expected for [0,0,1]:
    // [ 0 -1  0]
    // [ 1  0  0]
    // [ 0  0  0]
    let expected = [0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(skew.data, expected);

    // Test with arbitrary vector
    let v = vec3(1.0, 2.0, 3.0);
    let skew = Mat3x3::skew_symmetric(v);
    // Expected for [1,2,3]:
    // [ 0 -3  2]
    // [ 3  0 -1]
    // [-2  1  0]
    let expected = [0.0, -3.0, 2.0, 3.0, 0.0, -1.0, -2.0, 1.0, 0.0];
    assert_eq!(skew.data, expected);
}

#[test]
fn test_outer_product() {
    // Test with unit vectors
    let a = vec3(1.0, 0.0, 0.0);
    let b = vec3(0.0, 1.0, 0.0);
    let outer = Mat3x3::outer_product(a, b);
    // Expected outer product for [1,0,0] ⊗ [0,1,0]:
    // [ 0  1  0]
    // [ 0  0  0]
    // [ 0  0  0]
    let expected = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    assert_eq!(outer.data, expected);

    // Test with same vector
    let a = vec3(1.0, 2.0, 3.0);
    let outer = Mat3x3::outer_product(a, a);
    // Expected outer product for [1,2,3] ⊗ [1,2,3]:
    // [ 1  2  3]
    // [ 2  4  6]
    // [ 3  6  9]
    let expected = [1.0, 2.0, 3.0, 2.0, 4.0, 6.0, 3.0, 6.0, 9.0];
    assert_eq!(outer.data, expected);

    // Test with arbitrary vectors
    let a = vec3(1.0, 2.0, 3.0);
    let b = vec3(4.0, 5.0, 6.0);
    let outer = Mat3x3::outer_product(a, b);
    // Expected outer product for [1,2,3] ⊗ [4,5,6]:
    // [ 4  5  6]
    // [ 8 10 12]
    // [12 15 18]
    let expected = [4.0, 5.0, 6.0, 8.0, 10.0, 12.0, 12.0, 15.0, 18.0];
    assert_eq!(outer.data, expected);
}

#[test]
fn test_scalar_multiplication() {
    let mat = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let scaled = mat * 2.0;
    let expected = [2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0];
    assert_eq!(scaled.data, expected);

    // Test with zero scaling
    let scaled = mat * 0.0;
    let expected = [0.0; 9];
    assert_eq!(scaled.data, expected);

    // Test with negative scaling
    let scaled = mat * -1.0;
    let expected = [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0, -7.0, -8.0, -9.0];
    assert_eq!(scaled.data, expected);
}

#[test]
fn test_addition() {
    let mat1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let mat2 = Mat3x3::new([9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
    let sum = mat1 + mat2;
    let expected = [10.0; 9]; // Each element should be 10.0
    assert_eq!(sum.data, expected);

    // Test addition with zero matrix
    let zero = Mat3x3::zeros();
    let sum = mat1 + zero;
    assert_eq!(sum.data, mat1.data);

    // Test addition with identity matrix
    let identity = Mat3x3::identity();
    let sum = mat1 + identity;
    let expected = [2.0, 2.0, 3.0, 4.0, 6.0, 6.0, 7.0, 8.0, 10.0];
    assert_eq!(sum.data, expected);
}

#[test]
fn test_matrix_multiplication() {
    let m1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    let m2 = Mat3x3::identity();
    let product = m1 * m2;
    assert_eq!(product.data, m1.data);
}

#[test]
fn test_is_finite() {
    let m1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    assert!(m1.is_finite());

    let mut m2 = Mat3x3::zeros();
    m2[[0, 0]] = f32::INFINITY;
    assert!(!m2.is_finite());
}
