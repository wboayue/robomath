use super::super::*;

#[test]
fn test_creation_and_basic_ops() {
    // Test creation and default
    let v = vec2(1.0, 2.0);
    assert_eq!(v.x, 1.0);
    assert_eq!(v.y, 2.0);

    let v_default: Vec2<f32> = Vec2::default();
    assert_eq!(v_default.x, 0.0);
    assert_eq!(v_default.y, 0.0);

    // Test addition
    let v1 = vec2(1.0, 2.0);
    let v2 = vec2(3.0, 4.0);
    let sum = v1 + v2;
    assert_eq!(sum.x, 4.0);
    assert_eq!(sum.y, 6.0);

    // Test subtraction
    let diff = v2 - v1;
    assert_eq!(diff.x, 2.0);
    assert_eq!(diff.y, 2.0);

    // Test multiplication
    let prod = v1 * v2;
    assert_eq!(prod.x, 3.0); // 1.0 * 3.0
    assert_eq!(prod.y, 8.0); // 2.0 * 4.0

    // Test scalar multiplication
    let scaled = 2.0 * v1;
    assert_eq!(scaled.x, 2.0);
    assert_eq!(scaled.y, 4.0);

    // Test division by scalar
    let divided = v2 / 2.0;
    assert_eq!(divided.x, 1.5);
    assert_eq!(divided.y, 2.0);
}

#[test]
fn test_negation() {
    let v = vec2(1.0, -2.0);
    let neg = -v;
    assert_eq!(neg.x, -1.0);
    assert_eq!(neg.y, 2.0);

    // Test double negation
    let double_neg = -(-v);
    assert_eq!(double_neg.x, 1.0);
    assert_eq!(double_neg.y, -2.0);
}

#[test]
fn test_clamp() {
    // Test clamping within range
    let v = vec2(1.0, 2.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 1.0);
    assert_eq!(result.y, 2.0);

    // Test clamping below minimum
    let v = vec2(-1.0, -2.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 0.0);

    // Test clamping above maximum
    let v = vec2(6.0, 7.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 5.0);
    assert_eq!(result.y, 5.0);

    // Test mixed clamping
    let v = vec2(-1.0, 8.0);
    let result = v.clamp(0.0, 5.0);
    assert_eq!(result.x, 0.0);
    assert_eq!(result.y, 5.0);
}
