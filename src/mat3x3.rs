use core::ops::{Add, Index, IndexMut, Mul};

use crate::Vec3;

/// A 3x3 matrix stored in row-major order, suitable for linear algebra and transformations in 3D space.
///
/// `Mat3x3` represents a 3x3 matrix using a single `[f32; 9]` array in row-major order.
/// This means the elements are stored as:
/// - `data[0..3]` represents the first row (`[m00, m01, m02]`),
/// - `data[3..6]` represents the second row (`[m10, m11, m12]`),
/// - `data[6..9]` represents the third row (`[m20, m21, m22]`).
///
/// Elements can be accessed using the `[row, col]` syntax via the `Index` and `IndexMut` traits.
/// The matrix supports various operations such as addition, scalar multiplication, transposition,
/// determinant, trace, and specialized constructions like skew-symmetric matrices and outer products.
///
/// # Examples
///
/// ```
/// use robomath::{Mat3x3, vec3};
///
/// // Create a 3x3 matrix
/// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
///
/// // Access elements
/// assert_eq!(m[1][1], 5.0); // Element at row 1, column 1
///
/// // Compute determinant
/// let det = m.determinant();
/// assert_eq!(det, 0.0); // 1*(5*9 - 6*8) - 2*(4*9 - 6*7) + 3*(4*8 - 5*7)
///
/// // Create an identity matrix
/// let id = Mat3x3::identity();
/// assert_eq!(id[0][0], 1.0);
/// assert_eq!(id[0][1], 0.0);
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3x3 {
    /// Matrix data stored in row-major order as a flat array.
    pub data: [f32; 9],
}

impl Mat3x3 {
    /// Creates a new `Mat3x3` from a flat array of 9 elements in row-major order.
    ///
    /// The input array is interpreted as:
    /// - `data[0..3]`: First row (`[m00, m01, m02]`),
    /// - `data[3..6]`: Second row (`[m10, m11, m12]`),
    /// - `data[6..9]`: Third row (`[m20, m21, m22]`).
    ///
    /// # Arguments
    ///
    /// * `data` - A 9-element array containing the matrix elements in row-major order.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` with the specified elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// assert_eq!(m[0][0], 1.0);
    /// assert_eq!(m[1][1], 5.0);
    /// assert_eq!(m[2][2], 9.0);
    /// ```    
    pub fn new(data: [f32; 9]) -> Self {
        Self { data }
    }

    /// Creates a zero matrix (all elements are 0.0).
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` with all elements set to 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::zeros();
    /// assert_eq!(m.data, [0.0; 9]);
    /// ```
    pub fn zeros() -> Self {
        Self { data: [0.0; 9] }
    }

    /// Creates an identity matrix (1s on the diagonal, 0s elsewhere).
    ///
    /// The identity matrix has the form:
    /// ```text
    /// [1 0 0]
    /// [0 1 0]
    /// [0 0 1]
    /// ```
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` representing the identity matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::identity();
    /// assert_eq!(m[0][0], 1.0);
    /// assert_eq!(m[0][1], 0.0);
    /// assert_eq!(m[1][1], 1.0);
    /// assert_eq!(m[2][2], 1.0);
    /// ```
    pub fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a skew-symmetric matrix from a 3D vector.
    ///
    /// Given a vector `[x, y, z]`, the skew-symmetric matrix is:
    /// ```text
    /// [ 0 -z  y]
    /// [ z  0 -x]
    /// [-y  x  0]
    /// ```
    /// This matrix can be used to represent the cross product as a matrix multiplication.
    ///
    /// # Arguments
    ///
    /// * `v` - The `Vec3<f32>` to construct the skew-symmetric matrix from.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` representing the skew-symmetric matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Mat3x3, vec3};
    ///
    /// let v = vec3(1.0, 2.0, 3.0);
    /// let m = Mat3x3::skew_symmetric(v);
    /// assert_eq!(m[0][0], 0.0);
    /// assert_eq!(m[0][1], -3.0);
    /// assert_eq!(m[0][2], 2.0);
    /// assert_eq!(m[1][0], 3.0);
    /// assert_eq!(m[1][1], 0.0);
    /// assert_eq!(m[2][2], 0.0);
    /// ```    
    pub fn skew_symmetric(q: Vec3<f32>) -> Mat3x3 {
        Self {
            data: [0.0, -q.z, q.y, q.z, 0.0, -q.x, -q.y, q.x, 0.0],
        }
    }

    /// Computes the outer product of two 3D vectors.
    ///
    /// The outer product of vectors `u` and `v` is a matrix where element `[i, j]`
    /// is `u[i] * v[j]`. This results in a 3x3 matrix.
    ///
    /// # Arguments
    ///
    /// * `u` - The first `Vec3<f32>` (left operand).
    /// * `v` - The second `Vec3<f32>` (right operand).
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` representing the outer product.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::{Mat3x3, vec3};
    ///
    /// let u = vec3(1.0, 2.0, 3.0);
    /// let v = vec3(4.0, 5.0, 6.0);
    /// let m = Mat3x3::outer_product(u, v);
    /// assert_eq!(m[0][0], 4.0); // 1*4
    /// assert_eq!(m[0][1], 5.0); // 1*5
    /// assert_eq!(m[1][2], 12.0); // 2*6
    /// assert_eq!(m[2][2], 18.0); // 3*6
    /// ```
    pub fn outer_product(a: Vec3<f32>, b: Vec3<f32>) -> Mat3x3 {
        let mut data = [0.0; 9];

        data[0] = a.x * b.x;
        data[1] = a.x * b.y;
        data[2] = a.x * b.z;

        data[3] = a.y * b.x;
        data[4] = a.y * b.y;
        data[5] = a.y * b.z;

        data[6] = a.z * b.x;
        data[7] = a.z * b.y;
        data[8] = a.z * b.z;

        Self { data }
    }

    /// Computes the transpose of the matrix.
    ///
    /// The transpose swaps rows and columns: element at `[row, col]` moves to `[col, row]`.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` representing the transpose of the matrix.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let mt = m.transpose();
    /// assert_eq!(mt[0][0], 1.0);
    /// assert_eq!(mt[0][1], 4.0);
    /// assert_eq!(mt[1][0], 2.0);
    /// assert_eq!(mt[2][2], 9.0);
    /// ```
    pub fn transpose(&self) -> Self {
        Self {
            data: [
                self.data[0],
                self.data[3],
                self.data[6],
                self.data[1],
                self.data[4],
                self.data[7],
                self.data[2],
                self.data[5],
                self.data[8],
            ],
        }
    }

    /// Computes the determinant of the matrix.
    ///
    /// The determinant is computed using the formula:
    /// ```text
    /// det = m00*(m11*m22 - m12*m21) - m01*(m10*m22 - m12*m20) + m02*(m10*m21 - m11*m20)
    /// ```
    ///
    /// # Returns
    ///
    /// The determinant as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let det = m.determinant();
    /// assert_eq!(det, 0.0); // This matrix is singular
    ///
    /// let m2 = Mat3x3::identity();
    /// assert_eq!(m2.determinant(), 1.0);
    /// ```    
    pub fn determinant(&self) -> f32 {
        let data = &self.data;

        data[0] * (data[4] * data[8] - data[7] * data[5])
            - data[3] * (data[1] * data[8] - data[7] * data[2])
            + data[6] * (data[1] * data[5] - data[4] * data[2])
    }

    /// Computes the trace of the matrix.
    ///
    /// The trace is the sum of the diagonal elements: `m00 + m11 + m22`.
    ///
    /// # Returns
    ///
    /// The trace as an `f32`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let trace = m.trace();
    /// assert_eq!(trace, 15.0); // 1 + 5 + 9
    ///
    /// let id = Mat3x3::identity();
    /// assert_eq!(id.trace(), 3.0); // 1 + 1 + 1
    /// ```
    pub fn trace(&self) -> f32 {
        self.data[0] + self.data[4] + self.data[8]
    }

    /// Checks if all elements of the matrix are finite.
    ///
    /// Returns `true` if all elements are neither infinite nor NaN,
    /// according to the IEEE 754 floating-point specification.
    ///
    /// # Returns
    ///
    /// `true` if all elements are finite, `false` otherwise.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// assert!(m1.is_finite());
    ///
    /// let mut m2 = Mat3x3::zeros();
    /// m2[0][0] = f32::INFINITY;
    /// assert!(!m2.is_finite());
    ///
    /// let mut m3 = Mat3x3::zeros();
    /// m3[1][1] = f32::NAN;
    /// assert!(!m3.is_finite());
    /// ```
    pub fn is_finite(&self) -> bool {
        self.data.iter().all(|&x| x.is_finite())
    }
}

impl Index<[usize; 2]> for Mat3x3 {
    type Output = f32;

    /// Provides row-wise indexing into the matrix.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0, 1, or 2).
    ///
    /// # Returns
    ///
    /// A slice representing the row, which can be further indexed by column.
    ///
    /// # Panics
    ///
    /// Panics if `row` is not in the range `[0, 2]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// assert_eq!(m[0][0], 1.0);
    /// assert_eq!(m[1][1], 5.0);
    /// ```    
    fn index(&self, ndx: [usize; 2]) -> &Self::Output {
        &self.data[ndx[0] * 3 + ndx[1]]
    }
}

impl IndexMut<[usize; 2]> for Mat3x3 {
    /// Provides mutable row-wise indexing into the matrix.
    ///
    /// # Arguments
    ///
    /// * `row` - The row index (0, 1, or 2).
    ///
    /// # Returns
    ///
    /// A mutable slice representing the row, which can be further indexed by column.
    ///
    /// # Panics
    ///
    /// Panics if `row` is not in the range `[0, 2]`.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let mut m = Mat3x3::zeros();
    /// m[0][0] = 1.0;
    /// m[1][1] = 5.0;
    /// assert_eq!(m[0][0], 1.0);
    /// assert_eq!(m[1][1], 5.0);
    /// ```
    fn index_mut(&mut self, row: [usize; 2]) -> &mut f32 {
        &mut self.data[row[0] * 3 + row[1]]
    }
}

impl Mul<f32> for Mat3x3 {
    type Output = Mat3x3;

    /// Scales a `Mat3x3` by a scalar value.
    ///
    /// Each element of the matrix is multiplied by the scalar.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The scalar to multiply with.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` with each element scaled by the scalar.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let scaled = m * 2.0;
    /// assert_eq!(scaled[0][0], 2.0);
    /// assert_eq!(scaled[1][1], 10.0);
    /// assert_eq!(scaled[2][2], 18.0);
    /// ```    
    fn mul(self, scalar: f32) -> Mat3x3 {
        let mut data = [0.0; 9];
        for (i, item) in data.iter_mut().enumerate() {
            *item = self.data[i] * scalar;
        }
        Mat3x3 { data }
    }
}

impl Add<Mat3x3> for Mat3x3 {
    type Output = Mat3x3;

    /// Adds two `Mat3x3` matrices element-wise.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to add to `self`.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` where each element is the sum of the corresponding elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let m2 = Mat3x3::identity();
    /// let sum = m1 + m2;
    /// assert_eq!(sum[0][0], 2.0); // 1 + 1
    /// assert_eq!(sum[0][1], 2.0); // 2 + 0
    /// assert_eq!(sum[1][1], 6.0); // 5 + 1
    /// ```    
    fn add(self, rhs: Mat3x3) -> Mat3x3 {
        let mut data = [0.0; 9];
        for (i, item) in data.iter_mut().enumerate() {
            *item = self.data[i] + rhs.data[i];
        }
        Mat3x3 { data }
    }
}

impl Mul<Mat3x3> for Mat3x3 {
    type Output = Mat3x3;

    /// Multiplies two `Mat3x3` matrices using standard matrix multiplication.
    ///
    /// Matrix multiplication computes each element of the resulting matrix as the dot product
    /// of a row from the left matrix (`self`) and a column from the right matrix (`rhs`).
    /// Specifically, for element `[i, j]` of the result:
    /// - `result[i][j] = sum(self[i][k] * rhs[k][j])` for `k` from 0 to 2.
    ///
    /// This operation assumes row-major storage, consistent with the `Mat3x3` struct.
    ///
    /// # Arguments
    ///
    /// * `rhs` - The matrix to multiply with `self`.
    ///
    /// # Returns
    ///
    /// A new `Mat3x3` representing the product of the two matrices.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    ///
    /// let m1 = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let m2 = Mat3x3::identity();
    /// let product = m1 * m2;
    /// assert_eq!(product, m1); // Multiplying by identity yields the same matrix
    ///
    /// let m3 = Mat3x3::new([2.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 2.0]);
    /// let m4 = Mat3x3::new([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
    /// let product = m3 * m4;
    /// assert_eq!(product[0][0], 6.0); // 2*(1+1+1)
    /// assert_eq!(product[1][1], 6.0); // 2*(1+1+1)
    /// assert_eq!(product[2][2], 6.0); // 2*(1+1+1)
    /// ```
    fn mul(self, rhs: Mat3x3) -> Mat3x3 {
        let mut data = [0.0; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut sum = 0.0;
                for k in 0..3 {
                    sum += self[[i, k]] * rhs[[k, j]];
                }
                data[i * 3 + j] = sum;
            }
        }
        Mat3x3 { data }
    }
}
