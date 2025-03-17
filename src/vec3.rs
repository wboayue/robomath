use core::ops::{Add, Div, Mul, Sub};

use libm::sqrtf;

/// A 3D vector with generic components, suitable for mathematical operations in 3D space.
///
/// `Vec3<T>` represents a 3D vector with components `x`, `y`, and `z`, where `T` is a numeric type.
/// It supports various arithmetic operations such as addition, subtraction, multiplication,
/// and division. For `T = f32`, additional methods like `clamp`, `dot`, `cross`, and `is_finite`
/// are available, along with magnitude calculations.
///
/// The generic type `T` must implement certain traits depending on the operations used:
/// - For basic instantiation: No constraints.
/// - For arithmetic operations: `T: Add`, `T: Sub`, `T: Mul`, `T: Div`.
/// - For magnitude calculations: `T: Mul + Add` for `magnitude_squared`, and `T: Into<f32>` for `magnitude`.
/// - For `Default` implementation: `T: Default`.
///
/// # Examples
///
/// ```
/// use robomath::{Vec3, vec3};
///
/// // Create a Vec3 with f32 components
/// let v1 = vec3(1.0, 2.0, 3.0);
/// let v2 = vec3(4.0, 5.0, 6.0);
///
/// // Perform arithmetic operations
/// let sum = v1 + v2;
/// assert_eq!(sum, vec3(5.0, 7.0, 9.0));
///
/// // Scalar multiplication
/// let scaled = 2.0 * v1;
/// assert_eq!(scaled, vec3(2.0, 4.0, 6.0));
///
/// // Compute dot and cross products (only available for f32)
/// let dot = v1.dot(v2);
/// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
///
/// let cross = v1.cross(v2);
/// assert_eq!(cross, vec3(-3.0, 6.0, -3.0));
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

/// Creates a new `Vec3` with the given components.
///
/// This helper function provides a concise way to instantiate a `Vec3`.
///
/// # Arguments
///
/// * `x` - The x-coordinate of the vector.
/// * `y` - The y-coordinate of the vector.
/// * `z` - The z-coordinate of the vector.
///
/// # Returns
///
/// A new `Vec3<T>` with the specified components.
///
/// # Examples
///
/// ```
/// use robomath::vec3;
///
/// let v = vec3(1.0, 2.0, 3.0);
/// assert_eq!(v.x, 1.0);
/// assert_eq!(v.y, 2.0);
/// assert_eq!(v.z, 3.0);
/// ```
pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
}

impl Vec3<f32> {
    /// Clamps the components of the vector to be within the specified range.
    ///
    /// Each component (`x`, `y`, `z`) is clamped to the interval `[min, max]`. If a component
    /// is less than `min`, it is set to `min`. If it is greater than `max`, it is set to `max`.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for each component.
    /// * `max` - The maximum value for each component.
    ///
    /// # Returns
    ///
    /// A new `Vec3<f32>` with components clamped to the specified range.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(-1.0, 5.0, 10.0);
    /// let clamped = v.clamp(0.0, 2.0);
    /// assert_eq!(clamped, vec3(0.0, 2.0, 2.0));
    /// ```
    pub fn clamp(&self, min: f32, max: f32) -> Vec3<f32> {
        Vec3 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }

    /// Computes the dot product of two vectors.
    ///
    /// The dot product is calculated as `self.x * rhs.x + self.y * rhs.y + self.z * rhs.z`.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The other vector to compute the dot product with.
    ///
    /// # Returns
    ///
    /// The dot product as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(1.0, 2.0, 3.0);
    /// let v2 = vec3(4.0, 5.0, 6.0);
    /// let dot = v1.dot(v2);
    /// assert_eq!(dot, 32.0); // 1*4 + 2*5 + 3*6
    /// ```    
    pub fn dot(&self, rhs: Vec3<f32>) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }

    /// Computes the cross product of two vectors.
    ///
    /// The cross product results in a vector perpendicular to both input vectors.
    /// It is calculated as:
    /// - `x = self.y * rhs.z - self.z * rhs.y`
    /// - `y = self.z * rhs.x - self.x * rhs.z`
    /// - `z = self.x * rhs.y - self.y * rhs.x`
    ///
    /// # Arguments
    ///
    /// * `rhs` - The other vector to compute the cross product with.
    ///
    /// # Returns
    ///
    /// A new `Vec3<f32>` representing the cross product.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(1.0, 2.0, 3.0);
    /// let v2 = vec3(4.0, 5.0, 6.0);
    /// let cross = v1.cross(v2);
    /// assert_eq!(cross, vec3(-3.0, 6.0, -3.0));
    /// ```    
    pub fn cross(&self, rhs: Vec3<f32>) -> Vec3<f32> {
        vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }

    /// Checks if all components of the vector are finite.
    ///
    /// Returns `true` if all components (`x`, `y`, `z`) are neither infinite nor NaN,
    /// according to the IEEE 754 floating-point specification.
    ///
    /// # Returns
    ///
    /// `true` if all components are finite, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(1.0, 2.0, 3.0);
    /// assert!(v1.is_finite());
    ///
    /// let v2 = vec3(f32::INFINITY, 0.0, 0.0);
    /// assert!(!v2.is_finite());
    ///
    /// let v3 = vec3(0.0, f32::NAN, 0.0);
    /// assert!(!v3.is_finite());
    /// ```    
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

impl<T: Sub<Output = T>> Sub for Vec3<T> {
    type Output = Vec3<T>;

    /// Subtracts two `Vec3`s component-wise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec3<T>` where each component is the difference of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(5.0, 7.0, 9.0);
    /// let v2 = vec3(1.0, 2.0, 3.0);
    /// let result = v1 - v2;
    /// assert_eq!(result, vec3(4.0, 5.0, 6.0));
    /// ```    
    fn sub(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
            z: self.z - rhs.z,
        }
    }
}

impl<T: Add<Output = T>> Add for Vec3<T> {
    type Output = Vec3<T>;

    /// Adds two `Vec3`s component-wise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec3<T>` where each component is the sum of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(1.0, 2.0, 3.0);
    /// let v2 = vec3(4.0, 5.0, 6.0);
    /// let result = v1 + v2;
    /// assert_eq!(result, vec3(5.0, 7.0, 9.0));
    /// ```    
    fn add(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
            z: self.z + rhs.z,
        }
    }
}

impl<T: Mul<Output = T>> Mul for Vec3<T> {
    type Output = Vec3<T>;

    /// Multiplies two `Vec3`s component-wise (element-wise multiplication).
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec3<T>` where each component is the product of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(2.0, 3.0, 4.0);
    /// let v2 = vec3(5.0, 6.0, 7.0);
    /// let result = v1 * v2;
    /// assert_eq!(result, vec3(10.0, 18.0, 28.0));
    /// ```    
    fn mul(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

impl Mul<Vec3<f32>> for f32 {
    type Output = Vec3<f32>;

    /// Scales a `Vec3<f32>` by a scalar value.
    ///
    /// Each component of the vector is multiplied by the scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to scale.
    ///
    /// # Returns
    ///
    /// A new `Vec3<f32>` with each component scaled by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(1.0, 2.0, 3.0);
    /// let scaled = 2.0 * v;
    /// assert_eq!(scaled, vec3(2.0, 4.0, 6.0));
    /// ```    
    fn mul(self, rhs: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

impl Mul<f32> for Vec3<f32> {
    type Output = Vec3<f32>;

    /// Scales a `Vec3<f32>` by a scalar value (right-hand side).
    ///
    /// Each component of the vector is multiplied by the scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar to multiply with.
    ///
    /// # Returns
    ///
    /// A new `Vec3<f32>` with each component scaled by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(1.0, 2.0, 3.0);
    /// let scaled = v * 2.0;
    /// assert_eq!(scaled, vec3(2.0, 4.0, 6.0));
    /// ```    
    fn mul(self, rhs: f32) -> Vec3<f32> {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div<T> for Vec3<T> {
    type Output = Vec3<T>;

    /// Divides each component of the `Vec3` by a scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar to divide by.
    ///
    /// # Returns
    ///
    /// A new `Vec3<T>` with each component divided by the scalar.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero and `T` does not handle division by zero gracefully (e.g., for integers).
    /// For `T = f32`, division by zero results in infinity or NaN as per IEEE 754.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(4.0, 6.0, 8.0);
    /// let result = v / 2.0;
    /// assert_eq!(result, vec3(2.0, 3.0, 4.0));
    /// ```    
    fn div(self, rhs: T) -> Vec3<T> {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

impl<T: Div<Output = T> + Copy> Div<Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;

    /// Divides two `Vec3`s component-wise (element-wise division).
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to divide by.
    ///
    /// # Returns
    ///
    /// A new `Vec3<T>` where each component is the quotient of the corresponding components.
    ///
    /// # Panics
    ///
    /// Panics if any component of `rhs` is zero and `T` does not handle division by zero gracefully.
    /// For `T = f32`, division by zero results in infinity or NaN as per IEEE 754.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v1 = vec3(4.0, 6.0, 8.0);
    /// let v2 = vec3(2.0, 3.0, 4.0);
    /// let result = v1 / v2;
    /// assert_eq!(result, vec3(2.0, 2.0, 2.0));
    /// ```    
    fn div(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl<T> Vec3<T> {
    /// Computes the squared magnitude (length) of the vector.
    ///
    /// The squared magnitude is calculated as `x * x + y * y + z * z`.
    /// This method is generic and works for any type `T` that supports multiplication and addition.
    ///
    /// # Returns
    ///
    /// The squared magnitude as type `T`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(3.0, 4.0, 0.0);
    /// let mag_sq = v.magnitude_squared();
    /// assert_eq!(mag_sq, 25.0); // 3^2 + 4^2 + 0^2 = 25
    /// ```    
    pub fn magnitude_squared(&self) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Copy,
    {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    /// Computes the magnitude (length) of the vector.
    ///
    /// The magnitude is the square root of the squared magnitude, calculated as `sqrt(x * x + y * y + z * z)`.
    /// This method requires `T: Into<f32>` because it uses `libm::sqrtf` to compute the square root.
    ///
    /// # Returns
    ///
    /// The magnitude as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec3;
    ///
    /// let v = vec3(3.0, 4.0, 0.0);
    /// let mag = v.magnitude();
    /// assert_eq!(mag, 5.0); // sqrt(3^2 + 4^2 + 0^2) = 5
    /// ```    
    pub fn magnitude(&self) -> f32
    where
        T: Into<f32> + Mul<Output = T> + Add<Output = T> + Copy,
    {
        sqrtf(self.magnitude_squared().into())
    }
}

impl<T: Default> Default for Vec3<T> {
    /// Provides a default `Vec3` where each component is `T::default()`.
    ///
    /// For numeric types, this typically means zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Vec3;
    ///
    /// let v: Vec3<f32> = Vec3::default();
    /// assert_eq!(v, Vec3 { x: 0.0, y: 0.0, z: 0.0 });
    ///
    /// let v_int: Vec3<i32> = Vec3::default();
    /// assert_eq!(v_int, Vec3 { x: 0, y: 0, z: 0 });
    /// ```    
    fn default() -> Self {
        Vec3 {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}
