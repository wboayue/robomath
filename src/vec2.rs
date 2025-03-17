use core::ops::{Add, Div, Mul, Neg, Sub};

/// A 2D vector with generic components, suitable for mathematical operations in 2D space.
///
/// `Vec2<T>` represents a 2D vector with components `x` and `y`, where `T` is a numeric type.
/// It supports various arithmetic operations such as addition, subtraction, multiplication,
/// division, and negation. For `T = f32`, additional methods like `clamp` are available.
///
/// The generic type `T` must implement certain traits depending on the operations used:
/// - For basic instantiation: `T: Default`.
/// - For arithmetic operations: `T: Add`, `T: Sub`, `T: Mul`, `T: Div`.
/// - For negation: `T: Neg`.
///
/// # Examples
///
/// ```
/// use robomath::{Vec2, vec2};
///
/// // Create a Vec2 with f32 components
/// let v1 = vec2(1.0, 2.0);
/// let v2 = vec2(3.0, 4.0);
///
/// // Perform arithmetic operations
/// let sum = v1 + v2;
/// assert_eq!(sum, vec2(4.0, 6.0));
///
/// // Scalar multiplication
/// let scaled = 2.0 * v1;
/// assert_eq!(scaled, vec2(2.0, 4.0));
///
/// // Clamp components (only available for f32)
/// let clamped = v1.clamp(0.0, 1.5);
/// assert_eq!(clamped, vec2(1.0, 1.5));
/// ```
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec2<T: Default> {
    pub x: T,
    pub y: T,
}

impl Vec2<f32> {
    /// Clamps the components of the vector to be within the specified range.
    ///
    /// Each component (`x`, `y`) is clamped to the interval `[min, max]`. If a component
    /// is less than `min`, it is set to `min`. If it is greater than `max`, it is set to `max`.
    ///
    /// # Arguments
    ///
    /// * `min` - The minimum value for each component.
    /// * `max` - The maximum value for each component.
    ///
    /// # Returns
    ///
    /// A new `Vec2<f32>` with components clamped to the specified range.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v = vec2(-1.0, 5.0);
    /// let clamped = v.clamp(0.0, 2.0);
    /// assert_eq!(clamped, vec2(0.0, 2.0));
    /// ```    
    pub fn clamp(&self, min: f32, max: f32) -> Vec2<f32> {
        Vec2 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
        }
    }
}

/// Creates a new `Vec2` with the given components.
///
/// This helper function provides a concise way to instantiate a `Vec2`.
///
/// # Arguments
///
/// * `x` - The x-coordinate of the vector.
/// * `y` - The y-coordinate of the vector.
///
/// # Returns
///
/// A new `Vec2<T>` with the specified components.
///
/// # Examples
///
/// ```
/// use robomath::vec2;
///
/// let v = vec2(1.0, 2.0);
/// assert_eq!(v.x, 1.0);
/// assert_eq!(v.y, 2.0);
/// ```
pub fn vec2<T: Default>(x: T, y: T) -> Vec2<T> {
    Vec2 { x, y }
}

impl<T: Sub<Output = T> + Default> Sub for Vec2<T> {
    type Output = Vec2<T>;

    /// Subtracts two `Vec2`s component-wise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to subtract from `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec2<T>` where each component is the difference of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v1 = vec2(5.0, 7.0);
    /// let v2 = vec2(1.0, 2.0);
    /// let result = v1 - v2;
    /// assert_eq!(result, vec2(4.0, 5.0));
    /// ```    
    fn sub(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Add<Output = T> + Default> Add for Vec2<T> {
    type Output = Vec2<T>;

    /// Adds two `Vec2`s component-wise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec2<T>` where each component is the sum of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v1 = vec2(1.0, 2.0);
    /// let v2 = vec2(3.0, 4.0);
    /// let result = v1 + v2;
    /// assert_eq!(result, vec2(4.0, 6.0));
    /// ```    
    fn add(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Mul<Output = T> + Default> Mul for Vec2<T> {
    type Output = Vec2<T>;

    /// Multiplies two `Vec2`s component-wise (element-wise multiplication).
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `Vec2<T>` where each component is the product of the corresponding components.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v1 = vec2(2.0, 3.0);
    /// let v2 = vec2(5.0, 6.0);
    /// let result = v1 * v2;
    /// assert_eq!(result, vec2(10.0, 18.0));
    /// ```    
    fn mul(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

impl Mul<Vec2<f32>> for f32 {
    type Output = Vec2<f32>;

    /// Scales a `Vec2<f32>` by a scalar value.
    ///
    /// Each component of the vector is multiplied by the scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The vector to scale.
    ///
    /// # Returns
    ///
    /// A new `Vec2<f32>` with each component scaled by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v = vec2(1.0, 2.0);
    /// let scaled = 2.0 * v;
    /// assert_eq!(scaled, vec2(2.0, 4.0));
    /// ```    
    fn mul(self, rhs: Vec2<f32>) -> Vec2<f32> {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

impl<T: Div<Output = T> + Copy + Default> Div<T> for Vec2<T> {
    type Output = Vec2<T>;

    /// Divides each component of the `Vec2` by a scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar to divide by.
    ///
    /// # Returns
    ///
    /// A new `Vec2<T>` with each component divided by the scalar.
    ///
    /// # Panics
    ///
    /// Panics if `rhs` is zero and `T` does not handle division by zero gracefully (e.g., for integers).
    /// For `T = f32`, division by zero results in infinity or NaN as per IEEE 754.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v = vec2(4.0, 6.0);
    /// let result = v / 2.0;
    /// assert_eq!(result, vec2(2.0, 3.0));
    /// ```    
    fn div(self, rhs: T) -> Vec2<T> {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: Default> Default for Vec2<T> {
    /// Provides a default `Vec2` where each component is `T::default()`.
    ///
    /// For numeric types, this typically means zero.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Vec2;
    ///
    /// let v: Vec2<f32> = Vec2::default();
    /// assert_eq!(v, Vec2 { x: 0.0, y: 0.0 });
    ///
    /// let v_int: Vec2<i32> = Vec2::default();
    /// assert_eq!(v_int, Vec2 { x: 0, y: 0 });
    /// ```    
    fn default() -> Self {
        Vec2 {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl<T: Neg<Output = T> + Default> Neg for Vec2<T> {
    type Output = Vec2<T>;

    /// Negates each component of the `Vec2`.
    ///
    /// # Returns
    ///
    /// A new `Vec2<T>` with each component negated.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::vec2;
    ///
    /// let v = vec2(1.0, -2.0);
    /// let neg = -v;
    /// assert_eq!(neg, vec2(-1.0, 2.0));
    /// ```    
    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}
