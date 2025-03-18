use super::*;

#[test]
fn test_to_radians() {
    // Test common angle conversions
    assert!(
        (to_radians(0.0) - 0.0).abs() < 1e-5,
        "0 degrees should be 0 radians"
    );
    assert!(
        (to_radians(90.0) - PI / 2.0).abs() < 1e-5,
        "90 degrees should be π/2 radians"
    );
    assert!(
        (to_radians(180.0) - PI).abs() < 1e-5,
        "180 degrees should be π radians"
    );
    assert!(
        (to_radians(360.0) - 2.0 * PI).abs() < 1e-5,
        "360 degrees should be 2π radians"
    );

    // Test negative angles
    assert!(
        (to_radians(-90.0) - (-PI / 2.0)).abs() < 1e-5,
        "-90 degrees should be -π/2 radians"
    );

    // Test small angles
    assert!(
        (to_radians(1.0) - PI / 180.0).abs() < 1e-5,
        "1 degree should be π/180 radians"
    );

    // Test floating point precision with non-integer input
    assert!(
        (to_radians(45.5) - (45.5 * PI / 180.0)).abs() < 1e-5,
        "45.5 degrees conversion failed"
    );

    // Test extreme values
    assert!(
        to_radians(f32::MAX).is_finite() || to_radians(f32::MAX).is_infinite(),
        "MAX value should be finite or infinite"
    );
    assert!(
        to_radians(f32::MIN).is_finite(),
        "MIN value should be finite"
    );
}
