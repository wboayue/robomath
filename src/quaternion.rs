use core::{
    fmt::{self, Display},
    ops::Mul,
};

use libm::{asinf, atan2f, cosf, sinf, sqrtf};

use crate::{vec3, Mat3x3, Vec3};

/// A quaternion representing a 3D rotation or orientation.
///
/// Quaternions are mathematical objects of the form `w + xi + yj + zk`, where `w, x, y, z` are
/// real numbers, and `i, j, k` are imaginary units satisfying `i^2 = j^2 = k^2 = ijk = -1`.
/// In this implementation, quaternions are primarily used to represent 3D rotations, where a unit
/// quaternion (i.e., `w^2 + x^2 + y^2 + z^2 = 1`) encodes a rotation in 3D space.
///
/// # Fields
/// - `w`: The scalar (real) part of the quaternion.
/// - `x`: The coefficient of the `i` imaginary unit (first vector component).
/// - `y`: The coefficient of the `j` imaginary unit (second vector component).
/// - `z`: The coefficient of the `k` imaginary unit (third vector component).
///
/// # Examples
/// ```
/// use robomath::Quaternion;
///
/// let q = Quaternion::new(1.0, 0.0, 0.0, 0.0); // Identity quaternion
///
/// assert_eq!(q.w, 1.0);
/// assert_eq!(q.x, 0.0);
/// assert_eq!(q.y, 0.0);
/// assert_eq!(q.z, 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Quaternion {
    pub w: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Quaternion {
    /// Creates a new quaternion with the given components.
    ///
    /// # Arguments
    /// * `w` - The scalar part of the quaternion.
    /// * `x` - The `i` component (first vector part).
    /// * `y` - The `j` component (second vector part).
    /// * `z` - The `k` component (third vector part).
    ///
    /// # Returns
    /// A new `Quaternion` instance with the specified components.
    pub fn new(w: f32, x: f32, y: f32, z: f32) -> Self {
        Self { w, x, y, z }
    }

    /// Creates an identity quaternion representing no rotation.
    ///
    /// The identity quaternion is `(1, 0, 0, 0)`, which corresponds to a zero rotation.
    ///
    /// # Returns
    /// A `Quaternion` representing the identity (no rotation).
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::identity();
    ///
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

    /// Creates a quaternion from Euler angles (roll, pitch, yaw) in the ZYX convention.
    ///
    /// The Euler angles follow the "usual SLR sequence":
    /// 1. Rotate about the global Z-axis by `yaw`.
    /// 2. Rotate about the resulting Y-axis by `pitch`.
    /// 3. Rotate about the resulting X-axis by `roll`.
    ///
    /// The resulting quaternion represents the orientation of a body frame after applying these rotations.
    ///
    /// # Arguments
    /// * `yaw` - Rotation about the Z-axis (in radians).
    /// * `pitch` - Rotation about the Y-axis (in radians).
    /// * `roll` - Rotation about the X-axis (in radians).
    ///
    /// # Returns
    /// A `Quaternion` representing the combined rotation.
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    /// use libm::{cosf, sinf};
    ///
    /// const PI: f32 = 3.1415927;
    ///
    /// let q = Quaternion::from_euler(0.0, 0.0, PI / 2.0); // 90-degree yaw
    ///
    /// let expected = Quaternion::new(cosf(PI / 4.0), 0.0, 0.0, sinf(PI / 4.0));
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
    /// Converts a rotation matrix (assumed to be orthogonal with determinant 1) into its equivalent quaternion representation.
    /// This implementation uses a numerically stable algorithm that avoids division by zero by selecting the largest component.
    ///
    /// # Arguments
    /// * `rotation_matrix` - A `Mat3x3` representing a rotation (in row-major order).
    ///
    /// # Returns
    /// A `Quaternion` representing the same rotation as the input matrix.
    ///
    /// # Examples
    /// ```
    /// use robomath::{Quaternion, Mat3x3};
    /// use std::f32::consts::PI;
    ///
    /// let cos_90 = (PI / 2.0).cos();
    /// let sin_90 = (PI / 2.0).sin();
    /// let mat = Mat3x3::new([
    ///     cos_90, -sin_90, 0.0,
    ///     sin_90, cos_90,  0.0,
    ///     0.0,    0.0,     1.0,
    /// ]);
    /// let q = Quaternion::from_rotation_matrix(mat);
    ///
    /// let expected = Quaternion::new((PI / 4.0).cos(), 0.0, 0.0, (PI / 4.0).sin());
    /// ```
    pub fn from_rotation_matrix(rotation_matrix: Mat3x3) -> Self {
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
    /// The conjugate of a quaternion \( q = (w, x, y, z) \) is \( q^* = (w, -x, -y, -z) \).
    /// For unit quaternions, the conjugate is also the inverse.
    ///
    /// # Returns
    /// The conjugate of this quaternion.
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q = Quaternion::new(1.0, 2.0, 3.0, 4.0);
    /// let conj = q.conjugate();
    ///
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
    /// For a unit quaternion (\( w^2 + x^2 + y^2 + z^2 = 1 \)), the inverse is equal to its conjugate.
    /// If the quaternion is not a unit quaternion, this method does not compute the correct inverse
    /// (which would require division by the norm).
    ///
    /// # Returns
    /// The inverse of this quaternion, assuming it is a unit quaternion.
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    /// use std::f32::consts::PI;
    ///
    /// let q = Quaternion::from_euler(0.0, 0.0, PI / 2.0); // Unit quaternion
    /// let inv = q.inverse();
    ///
    /// let expected = q.conjugate();
    /// assert_eq!(inv.w, expected.w);
    /// assert_eq!(inv.x, expected.x);
    /// assert_eq!(inv.y, expected.y);
    /// assert_eq!(inv.z, expected.z);
    /// ```
    pub fn inverse(&self) -> Self {
        Self {
            w: self.w,
            x: -self.x,
            y: -self.y,
            z: -self.z,
        }
    }

    /// Computes the rotation matrix from inertial frame to body frame.
    ///
    /// The resulting matrix \( R \) transforms a vector from the body frame to the inertial frame:
    /// \( v_I = R v_B \). Assumes the quaternion is a unit quaternion.
    ///
    /// # Returns
    /// A `Mat3x3` representing the rotation from inertial to body frame.
    pub fn rotation_matrix_i_wrt_b(&self) -> Mat3x3 {
        let qvec = vec3(self.x, self.y, self.z);
        let sk_q = Mat3x3::skew_symmetric(qvec); // Build the skew symmetric matrix of the imaginary part of quaternion
        Mat3x3::identity() * (self.w * self.w - qvec.magnitude_squared())
            + Mat3x3::outer_product(qvec, qvec) * 2.0
            + sk_q * 2.0 * self.w
    }

    /// Computes the rotation matrix from body frame to inertial frame.
    ///
    /// The resulting matrix \( R \) transforms a vector from the inertial frame to the body frame:
    /// \( v_B = R v_I \). This is the transpose of `rotation_matrix_i_wrt_b`.
    ///
    /// # Returns
    /// A `Mat3x3` representing the rotation from body to inertial frame.
    pub fn rotation_matrix_b_wrt_i(&self) -> Mat3x3 {
        self.rotation_matrix_i_wrt_b().transpose()
    }

    /// Extracts the yaw angle (rotation about Z-axis) from the quaternion.
    ///
    /// Assumes the quaternion represents a rotation in the ZYX Euler angle convention.
    ///
    /// # Returns
    /// The yaw angle in radians.
    pub fn yaw(&self) -> f32 {
        let yaw_denominator = self.w * self.w + self.x * self.x - self.y * self.y - self.z * self.z;
        atan2f(2.0 * (self.x * self.y + self.w * self.z), yaw_denominator)
    }

    /// Extracts the pitch angle (rotation about Y-axis) from the quaternion.
    ///
    /// Assumes the quaternion represents a rotation in the ZYX Euler angle convention.
    ///
    /// # Returns
    /// The pitch angle in radians.
    pub fn pitch(&self) -> f32 {
        asinf(-2.0 * (self.x * self.z - self.w * self.y))
    }

    /// Extracts the roll angle (rotation about X-axis) from the quaternion.
    ///
    /// Assumes the quaternion represents a rotation in the ZYX Euler angle convention.
    ///
    /// # Returns
    /// The roll angle in radians.
    pub fn roll(&self) -> f32 {
        let roll_denominator =
            self.w * self.w - self.x * self.x - self.y * self.y + self.z * self.z;
        atan2f(2.0 * (self.y * self.z + self.w * self.x), roll_denominator)
    }

    /// Converts the quaternion to a Gibbs vector (Rodrigues parameters).
    ///
    /// The Gibbs vector is defined as \( (x/w, y/w, z/w) \) if \( w \neq 0 \). If \( w = 0 \),
    /// a large scalar is used to approximate infinity.
    ///
    /// # Returns
    /// A `Vec3<f32>` representing the Gibbs vector.
    pub fn to_gibbs_vector(&self) -> Vec3<f32> {
        if self.w == 0.0 {
            1e20 * vec3(self.x, self.y, self.z)
        } else {
            vec3(self.x, self.y, self.z) / self.w
        }
    }

    /// Converts the quaternion to Euler angles in yaw-pitch-roll order.
    ///
    /// # Returns
    /// A `Vec3<f32>` containing `(yaw, pitch, roll)` in radians.
    pub fn to_euler(&self) -> Vec3<f32> {
        vec3(self.yaw(), self.pitch(), self.roll())
    }

    /// Converts the quaternion to Euler angles in roll-pitch-yaw order.
    ///
    /// # Returns
    /// A `Vec3<f32>` containing `(roll, pitch, yaw)` in radians.
    pub fn to_euler_rpy(&self) -> Vec3<f32> {
        vec3(self.roll(), self.pitch(), self.yaw())
    }
}

/// Implements quaternion multiplication.
///
/// Quaternion multiplication follows the standard Hamilton product rules:
/// - \( w = w_1 w_2 - x_1 x_2 - y_1 y_2 - z_1 z_2 \)
/// - \( x = w_1 x_2 + x_1 w_2 + y_1 z_2 - z_1 y_2 \)
/// - \( y = w_1 y_2 + y_1 w_2 + z_1 x_2 - x_1 z_2 \)
/// - \( z = w_1 z_2 + z_1 w_2 + x_1 y_2 - y_1 x_2 \)
impl Mul<Quaternion> for Quaternion {
    type Output = Quaternion;

    /// Multiplies two quaternions.
    ///
    /// # Arguments
    /// * `p` - The right-hand quaternion to multiply with.
    ///
    /// # Returns
    /// A new `Quaternion` representing the product of `self` and `p`.
    ///
    /// # Examples
    /// ```
    /// use robomath::Quaternion;
    ///
    /// let q1 = Quaternion::new(0.0, 1.0, 0.0, 0.0); // i
    /// let q2 = Quaternion::new(0.0, 0.0, 1.0, 0.0); // j
    /// let result = q1 * q2;
    ///
    /// assert_eq!(result.w, 0.0);
    /// assert_eq!(result.x, 0.0);
    /// assert_eq!(result.y, 0.0);
    /// assert_eq!(result.z, 1.0); // i * j = k
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
