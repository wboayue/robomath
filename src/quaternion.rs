use core::{
    fmt::{self, Display},
    ops::Mul,
};

use libm::{asinf, atan2f, cosf, sinf, sqrtf};

use crate::{vec3, Mat3x3, Vec3};

/// A quaternion representing a 3D rotation, stored in scalar-first notation (w, x, y, z).
///
/// `Quaternion` represents a quaternion as `w + xi + yj + zk`, where `w` is the scalar part
/// and `(x, y, z)` is the vector part. Quaternions are used to represent 3D rotations in a
/// numerically stable and gimbal-lock-free manner. The quaternion is typically normalized
/// (magnitude of 1) to represent a valid rotation.
///
/// Key features include:
/// - Creation from components, Euler angles, or rotation matrices.
/// - Operations like multiplication (Hamilton product), conjugate, and inverse.
/// - Conversion to Euler angles, Gibbs vectors, and rotation matrices (inertial-to-body and body-to-inertial).
///
/// # Conventions
///
/// - The quaternion is stored in scalar-first order: `[w, x, y, z]`.
/// - Euler angles follow the ZYX convention (yaw, pitch, roll) unless specified otherwise (e.g., `to_euler_rpy`).
/// - Rotation matrices use standard aerospace conventions: `rotation_matrix_i_wrt_b` for inertial-to-body frame,
///   and `rotation_matrix_b_wrt_i` for body-to-inertial frame.
///
/// # Examples
///
/// ```
/// use robomath::{Quaternion, vec3};
///
/// // Create a quaternion from components
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity quaternion
/// assert_eq!(q.w, 1.0);
/// assert_eq!(q.x, 0.0);
///
/// // Create from Euler angles (90 degrees around Z-axis)
/// let q_yaw = Quaternion::from_euler(90.0_f32.to_radians(), 0.0, 0.0);
/// let yaw = q_yaw.yaw();
/// assert!((yaw - 90.0_f32.to_radians()).abs() < 1e-5);
///
/// // Multiply quaternions
/// let q1 = Quaternion::from_euler(0.0, 0.0, 90.0_f32.to_radians());
/// let q2 = Quaternion::from_euler(0.0, 90.0_f32.to_radians(), 0.0);
/// let q_combined = q1 * q2;
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    /// Creates a new quaternion from its components.
    ///
    /// # Arguments
    ///
    /// * `w` - The scalar component.
    /// * `x` - The x-component of the vector part.
    /// * `y` - The y-component of the vector part.
    /// * `z` - The z-component of the vector part.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` with the specified components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity quaternion
    /// assert_eq!(q.w, 1.0);
    /// assert_eq!(q.x, 0.0);
    /// assert_eq!(q.y, 0.0);
    /// assert_eq!(q.z, 0.0);
    /// ```
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Creates an identity quaternion (no rotation).
    ///
    /// The identity quaternion has `w = 1` and `x = y = z = 0`, representing no rotation.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the identity rotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::identity();
    /// assert_eq!(q.w, 1.0);
    /// assert_eq!(q.x, 0.0);
    /// assert_eq!(q.y, 0.0);
    /// assert_eq!(q.z, 0.0);
    /// ```
    pub fn identity() -> Self {
        Self {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }

    /// Creates a quaternion from Euler angles in ZYX order (yaw, pitch, roll).
    ///
    /// The angles are applied in the order: yaw (Z), pitch (Y), roll (X).
    /// All angles must be in radians.
    ///
    /// # Arguments
    ///
    /// * `yaw` - Rotation around the Z-axis (in radians).
    /// * `pitch` - Rotation around the Y-axis (in radians).
    /// * `roll` - Rotation around the X-axis (in radians).
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the combined rotation.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, to_radians};
    ///
    /// let q = Quaternion::from_euler(to_radians(90.0), 0.0, to_radians(45.0));
    /// let yaw = q.yaw();
    /// let roll = q.roll();
    /// assert!((yaw - to_radians(90.0)).abs() < 1e-5);
    /// assert!((roll - to_radians(45.0)).abs() < 1e-5);
    /// ```    
    pub fn from_euler(yaw: f32, pitch: f32, roll: f32) -> Self {
        let roll = roll / 2.0;
        let pitch = pitch / 2.0;
        let yaw = yaw / 2.0;

        let q0 = cosf(roll) * cosf(pitch) * cosf(yaw) + sinf(roll) * sinf(pitch) * sinf(yaw);
        let q1 = -cosf(roll) * sinf(pitch) * sinf(yaw) + cosf(pitch) * cosf(yaw) * sinf(roll);
        let q2 = cosf(roll) * cosf(yaw) * sinf(pitch) + sinf(roll) * cosf(pitch) * sinf(yaw);
        let q3 = cosf(roll) * cosf(pitch) * sinf(yaw) - sinf(roll) * cosf(yaw) * sinf(pitch);

        Self {
            w: q0,
            x: q1,
            y: q2,
            z: q3,
        }
    }

    /// Creates a quaternion from a 3x3 rotation matrix.
    ///
    /// This method constructs a quaternion from a rotation matrix using a numerically stable
    /// algorithm that avoids singularities by selecting the largest component first.
    /// The input matrix must be orthogonal with determinant 1 (a valid rotation matrix).
    ///
    /// # Arguments
    ///
    /// * `m` - The `Mat3x3` rotation matrix to convert.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the same rotation as the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Mat3x3, Quaternion};
    ///
    /// let m = Mat3x3::identity();
    /// let q = Quaternion::from_rotation_matrix(&m);
    /// assert_eq!(q, Quaternion::identity());
    /// ```    
    pub fn from_rotation_matrix(rotation_matrix: &Mat3x3) -> Self {
        let r = rotation_matrix.data;

        // Compute the four possible denominators
        let t0 = 1.0 + r[0] + r[4] + r[8]; // 1 + R11 + R22 + R33
        let t1 = 1.0 + r[0] - r[4] - r[8]; // 1 + R11 - R22 - R33
        let t2 = 1.0 - r[0] + r[4] - r[8]; // 1 - R11 + R22 - R33
        let t3 = 1.0 - r[0] - r[4] + r[8]; // 1 - R11 - R22 + R33

        // Find the largest value to determine which component to compute first
        if t0 >= t1 && t0 >= t2 && t0 >= t3 {
            // q0 is largest
            let q0 = 0.5 * sqrtf(t0);
            let q1 = (r[7] - r[5]) / (4.0 * q0); // (R32 - R23) / (4 * q0)
            let q2 = (r[2] - r[6]) / (4.0 * q0); // (R13 - R31) / (4 * q0)
            let q3 = (r[3] - r[1]) / (4.0 * q0); // (R21 - R12) / (4 * q0)
            Self {
                w: q0,
                x: q1,
                y: q2,
                z: q3,
            }
        } else if t1 >= t2 && t1 >= t3 {
            // q1 is largest
            let q1 = 0.5 * sqrtf(t1);
            let q0 = (r[7] - r[5]) / (4.0 * q1); // (R32 - R23) / (4 * q1)
            let q2 = (r[1] + r[3]) / (4.0 * q1); // (R12 + R21) / (4 * q1)
            let q3 = (r[2] + r[6]) / (4.0 * q1); // (R13 + R31) / (4 * q1)
            Self {
                w: q0,
                x: q1,
                y: q2,
                z: q3,
            }
        } else if t2 >= t3 {
            // q2 is largest
            let q2 = 0.5 * sqrtf(t2);
            let q0 = (r[2] - r[6]) / (4.0 * q2); // (R13 - R31) / (4 * q2)
            let q1 = (r[1] + r[3]) / (4.0 * q2); // (R12 + R21) / (4 * q2)
            let q3 = (r[5] + r[7]) / (4.0 * q2); // (R23 + R32) / (4 * q2)
            Self {
                w: q0,
                x: q1,
                y: q2,
                z: q3,
            }
        } else {
            // q3 is largest
            let q3 = 0.5 * sqrtf(t3);
            let q0 = (r[3] - r[1]) / (4.0 * q3); // (R21 - R12) / (4 * q3)
            let q1 = (r[2] + r[6]) / (4.0 * q3); // (R13 + R31) / (4 * q3)
            let q2 = (r[5] + r[7]) / (4.0 * q3); // (R23 + R32) / (4 * q3)
            Self {
                w: q0,
                x: q1,
                y: q2,
                z: q3,
            }
        }
    }

    /// Computes the conjugate of the quaternion.
    ///
    /// The conjugate of a quaternion `w + xi + yj + zk` is `w - xi - yj - zk`.
    /// For a unit quaternion, the conjugate is also its inverse.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the conjugate.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let conj = q.conjugate();
    /// assert_eq!(conj.w, 1.0);
    /// assert_eq!(conj.x, -2.0);
    /// assert_eq!(conj.y, -3.0);
    /// assert_eq!(conj.z, -4.0);
    /// ```    
    pub fn conjugate(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Computes the inverse of the quaternion, assuming it is a unit quaternion.
    ///
    /// For a unit quaternion (magnitude 1), the inverse is equal to its conjugate.
    /// This method does not normalize the quaternion or check its magnitude.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the inverse.
    ///
    /// # Panics
    ///
    /// This method does not panic, but the result is only valid for unit quaternions.
    /// For non-unit quaternions, the inverse would require division by the magnitude squared,
    /// which is not implemented here.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity (unit quaternion)
    /// let inv = q.inverse();
    /// assert_eq!(inv, q.conjugate());
    /// ```    
    pub fn inverse(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Computes the rotation matrix from inertial to body frame.
    ///
    /// This matrix transforms vectors from the inertial frame to the body frame.
    ///
    /// # Returns
    ///
    /// A `Mat3x3` representing the rotation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, Mat3x3};
    ///
    /// let q = Quaternion::identity();
    /// let m = q.rotation_matrix_i_wrt_b();
    /// assert_eq!(m, Mat3x3::identity());
    /// ```
    pub fn rotation_matrix_i_wrt_b(&self) -> Mat3x3 {
        let qvec = vec3(self.x, self.y, self.z);
        let sk_q = Mat3x3::skew_symmetric(qvec); // Build the skew symmetric matrix of the imaginary part of quaternion
        Mat3x3::identity() * (self.w * self.w - qvec.magnitude_squared())
            + Mat3x3::outer_product(qvec, qvec) * 2.0
            + sk_q * 2.0 * self.w
    }

    /// Computes the rotation matrix from body to inertial frame.
    ///
    /// This matrix transforms vectors from the body frame to the inertial frame.
    /// It is the transpose of `rotation_matrix_i_wrt_b`.
    ///
    /// # Returns
    ///
    /// A `Mat3x3` representing the rotation matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, Mat3x3};
    ///
    /// let q = Quaternion::identity();
    /// let m = q.rotation_matrix_b_wrt_i();
    /// assert_eq!(m, Mat3x3::identity());
    /// ```    
    pub fn rotation_matrix_b_wrt_i(&self) -> Mat3x3 {
        self.rotation_matrix_i_wrt_b().transpose()
    }

    /// Extracts the yaw (Z-axis rotation) in radians.
    ///
    /// Yaw is computed from the quaternion using the ZYX Euler angle convention.
    /// Returns a value in the range `[-π, π]`.
    ///
    /// # Returns
    ///
    /// The yaw angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::from_euler(90.0_f32.to_radians(), 0.0, 0.0);
    /// let yaw = q.yaw();
    /// assert!((yaw - 90.0_f32.to_radians()).abs() < 1e-5);
    /// ```    
    pub fn yaw(&self) -> f32 {
        let yaw_denominator = self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z;
        atan2f(2.0 * (self.x * self.y + self.w * self.z), yaw_denominator)
    }

    /// Extracts the pitch (Y-axis rotation) in radians.
    ///
    /// Pitch is computed from the quaternion using the ZYX Euler angle convention.
    /// Returns a value in the range `[-π/2, π/2]`.
    ///
    /// # Returns
    ///
    /// The pitch angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, to_radians};
    ///
    /// let q = Quaternion::from_euler(0.0, to_radians(45.0), 0.0);
    /// let pitch = q.pitch();
    /// 
    /// assert!((pitch - to_radians(45.0)).abs() < 1e-5);
    /// ```
    pub fn pitch(&self) -> f32 {
        asinf(-2.0 * (self.x * self.z - self.w * self.y))
    }

    /// Extracts the roll (X-axis rotation) in radians.
    ///
    /// Roll is computed from the quaternion using the ZYX Euler angle convention.
    /// Returns a value in the range `[-π, π]`.
    ///
    /// # Returns
    ///
    /// The roll angle in radians.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::from_euler(0.0, 0.0, 90.0_f32.to_radians());
    /// let roll = q.roll();
    /// assert!((roll - 90.0_f32.to_radians()).abs() < 1e-5);
    /// ```    
    pub fn roll(&self) -> f32 {
        let roll_denominator =
            self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z;
        atan2f(2.0 * (self.y * self.z + self.w * self.x), roll_denominator)
    }

    /// Converts the quaternion to a Gibbs vector (rotation axis scaled by tangent of half-angle).
    ///
    /// The Gibbs vector is `(x/w, y/w, z/w)`. For small rotations, it approximates the rotation axis
    /// scaled by the rotation angle in radians.
    ///
    /// # Returns
    ///
    /// A `Vec3<f32>` representing the Gibbs vector.
    ///
    /// # Panics
    ///
    /// Does not panic, but returns a large value (1e20) for components when `w` is zero to approximate infinity.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, vec3};
    ///
    /// let q = Quaternion::new(0.7071, 0.7071, 0.0, 0.0); // 90-degree rotation around X
    /// let gibbs = q.to_gibbs_vector();
    /// assert!((gibbs.x - 1.0).abs() < 1e-4);
    /// assert!((gibbs.y - 0.0).abs() < 1e-4);
    /// assert!((gibbs.z - 0.0).abs() < 1e-4);
    /// ```    
    pub fn to_gibbs_vector(&self) -> Vec3<f32> {
        if self.w == 0.0 {
            1e20 * vec3(self.x, self.y, self.z)
        } else {
            vec3(self.x, self.y, self.z) / self.w
        }
    }

    /// Converts the quaternion to Euler angles in ZYX order (yaw, pitch, roll).
    ///
    /// # Returns
    ///
    /// A tuple `(yaw, pitch, roll)` in radians, where:
    /// - `yaw` is the Z-axis rotation (`[-π, π]`).
    /// - `pitch` is the Y-axis rotation (`[-π/2, π/2]`).
    /// - `roll` is the X-axis rotation (`[-π, π]`).
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Quaternion, Vec3, to_radians};
    ///
    /// let q = Quaternion::from_euler(to_radians(90.0), to_radians(45.0), to_radians(30.0));
    /// let Vec3{x: roll, y: pitch, z: yaw} = q.to_euler();
    ///
    /// assert!((yaw - to_radians(90.0)).abs() < 1e-5);
    /// assert!((pitch - to_radians(45.0)).abs() < 1e-5);
    /// assert!((roll - to_radians(30.0)).abs() < 1e-5);
    /// ```    
    pub fn to_euler(&self) -> Vec3<f32> {
        vec3(self.roll(), self.pitch(), self.yaw())
    }

    /// Checks if all components of the quaternion are finite.
    ///
    /// Returns `true` if all components (`w`, `x`, `y`, `z`) are neither infinite nor NaN,
    /// according to the IEEE 754 floating-point specification.
    ///
    /// # Returns
    ///
    /// `true` if all components are finite, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q1 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert!(q1.is_finite());
    ///
    /// let q2 = Quaternion::new(f32::INFINITY, 0.0, 0.0, 0.0);
    /// assert!(!q2.is_finite());
    ///
    /// let q3 = Quaternion::new(0.0, f32::NAN, 0.0, 0.0);
    /// assert!(!q3.is_finite());
    /// ```
    pub fn is_finite(&self) -> bool {
        self.w.is_finite() && self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }

    /// Computes the magnitude (Euclidean norm) of the quaternion.
    ///
    /// The magnitude is calculated as `sqrt(w^2 + x^2 + y^2 + z^2)`, representing the length
    /// of the quaternion as a 4D vector. For a unit quaternion (representing a valid rotation),
    /// the magnitude is 1.0.
    ///
    /// # Returns
    ///
    /// The magnitude as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity quaternion
    /// let mag = q.magnitude();
    /// assert!((mag - 1.0).abs() < 1e-5);
    ///
    /// let q2 = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let mag2 = q2.magnitude();
    /// assert!((mag2 - 5.477225).abs() < 1e-5); // sqrt(1^2 + 2^2 + 3^2 + 4^2) ≈ 5.477225
    /// ```    
    pub fn magnitude(&self) -> f32 {
        sqrtf(self.w * self.w + self.x * self.x + self.y * self.y + self.z * self.z)
    }

    /// Normalizes the quaternion to have a magnitude of 1.
    ///
    /// The normalized quaternion is computed by dividing each component by the magnitude:
    /// `(w/m, x/m, y/m, z/m)`, where `m` is the magnitude. A unit quaternion represents a valid
    /// 3D rotation. If the magnitude is zero (or very close to zero), the identity quaternion
    /// is returned to avoid division by zero or numerical instability.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` with magnitude 1, or the identity quaternion if the original magnitude is zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(0.0, 2.0, 0.0, 0.0);
    /// let norm = q.normalize();
    /// assert!((norm.x - 1.0).abs() < 1e-5);
    /// assert!((norm.magnitude() - 1.0).abs() < 1e-5);
    ///
    /// let q_zero = Quaternion::new(0.0, 0.0, 0.0, 0.0);
    /// let norm_zero = q_zero.normalize();
    /// assert_eq!(norm_zero, Quaternion::identity());
    /// ```   
    pub fn normalize(&self) -> Quaternion {
        let mag = self.magnitude();
        if mag.abs() < f32::EPSILON {
            Quaternion::identity() // Handle zero magnitude case
        } else {
            Quaternion::new(self.w / mag, self.x / mag, self.y / mag, self.z / mag)
        }
    }
}

impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    /// Multiplies two quaternions using the Hamilton product.
    ///
    /// The Hamilton product combines two quaternions to represent the composition of their rotations.
    /// For quaternions `q1 = (w1, x1, y1, z1)` and `q2 = (w2, x2, y2, z2)`, the product is:
    /// - `w = w1*w2 - x1*x2 - y1*y2 - z1*z2`
    /// - `x = w1*x2 + x1*w2 + y1*z2 - z1*y2`
    /// - `y = w1*y2 - x1*z2 + y1*w2 + z1*x2`
    /// - `z = w1*z2 + x1*y2 - y1*x2 + z1*w2`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The quaternion to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `Quaternion` representing the product.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q1 = Quaternion::from_euler(0.0, 0.0, 90.0_f32.to_radians());
    /// let q2 = Quaternion::from_euler(0.0, 90.0_f32.to_radians(), 0.0);
    /// let q = q1 * q2;
    /// // q represents a combined rotation
    /// ```    
    fn mul(self, p: Quaternion) -> Quaternion {
        let q = self;

        let c0 = q.w * p.w - q.x * p.x - q.y * p.y - q.z * p.z; // w = w1w2 - x1x2 - y1y2 - z1z2
        let c1 = q.w * p.x + q.x * p.w + q.y * p.z - q.z * p.y; // x = w1x2 + x1w2 + y1z2 - z1y2
        let c2 = q.w * p.y + q.y * p.w + q.z * p.x - q.x * p.z; // y = w1y2 + y1w2 + z1x2 - x1z2
        let c3 = q.w * p.z + q.z * p.w + q.x * p.y - q.y * p.x; // z = w1z2 + z1w2 + x1y2 - y1x2

        Quaternion {
            w: c0,
            x: c1,
            y: c2,
            z: c3,
        }
    }
}

impl Default for Quaternion {
    /// Provides the default value for a `Quaternion`, which is the identity quaternion.
    ///
    /// # Returns
    /// The identity quaternion `(1, 0, 0, 0)`.
    fn default() -> Self {
        Quaternion {
            w: 1.0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }
    }
}

impl Display for Quaternion {
    /// Formats the quaternion as a string.
    ///
    /// # Returns
    /// A string in the format `"w x y z"`, with each component formatted to three decimal places.
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// assert_eq!(q.to_string(), "1.000 2.000 3.000 4.000");
    /// ```
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:.3} {:.3} {:.3} {:.3}", self.w, self.x, self.y, self.z)
    }
}
