use std::ops::{Add, Index, Mul};

use crate::Vec3;

/// A 3x3 matrix with single-precision floating-point elements, stored in row-major order.
///
/// The matrix is represented internally as a flat array of 9 `f32` values, where the elements
/// are ordered as `[a11, a12, a13, a21, a22, a23, a31, a32, a33]`. This struct provides basic
/// linear algebra operations for 3x3 matrices, such as transposition, determinant calculation,
/// and trace computation.
///
/// # Examples
///
/// ```
/// use robomath::Mat3x3;
///
/// // Create an identity matrix
/// let identity = Mat3x3::identity();
/// assert_eq!(identity.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
///
/// // Create a custom matrix
/// let matrix = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
/// assert_eq!(matrix[[1, 2]], 6.0); // Access element at row 1, column 2
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mat3x3 {
    pub data: [f32; 9],
}

impl Mat3x3 {
    /// Creates a new 3x3 matrix from a flat array of 9 elements in row-major order.
    ///
    /// The input array represents the matrix elements as `[a11, a12, a13, a21, a22, a23, a31, a32, a33]`.
    ///
    /// # Arguments
    ///
    /// * `data` - A 9-element array of `f32` values representing the matrix elements.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let matrix = Mat3x3::new([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// assert_eq!(matrix.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn new(data: [f32; 9]) -> Self {
        Self { data }
    }

    /// Creates a 3x3 zero matrix, where all elements are 0.0.
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let zero = Mat3x3::zeros();
    /// assert_eq!(zero.data, [0.0; 9]);
    /// ```
    pub fn zeros() -> Self {
        Self { data: [0.0; 9] }
    }

    /// Creates a 3x3 identity matrix, where diagonal elements are 1.0 and all other elements are 0.0.
    ///
    /// The resulting matrix is:
    ///
    /// \[
    /// \begin{bmatrix}
    /// 1 & 0 & 0 \\
    /// 0 & 1 & 0 \\
    /// 0 & 0 & 1
    /// \end{bmatrix}
    /// \]
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let identity = Mat3x3::identity();
    /// assert_eq!(identity.data, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    /// ```
    pub fn identity() -> Self {
        Self {
            data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        }
    }

    /// Creates a skew-symmetric matrix from a 3D vector.
    ///
    /// A skew-symmetric matrix `S` has the property that `S^T = -S`. For a 3D vector
    /// `v = [x, y, z]`, the corresponding skew-symmetric matrix is:
    ///
    /// ```text
    /// [  0  -z   y ]
    /// [  z   0  -x ]
    /// [ -y   x   0 ]
    /// ```
    ///
    /// This matrix is useful in various applications:
    ///
    /// - Computing cross products: For vectors `a` and `b`, the cross product `a × b`
    ///   can be computed as `skew_symmetric(a) * b`
    /// - Representing infinitesimal rotations in SO(3)
    /// - Converting between quaternions and rotation matrices
    ///
    /// # Arguments
    ///
    /// * `q` - A 3D vector whose components will be used to construct the skew-symmetric matrix
    ///
    /// # Returns
    ///
    /// A 3×3 skew-symmetric matrix corresponding to the input vector
    pub fn skew_symmetric(q: Vec3<f32>) -> Mat3x3 {
        Self {
            data: [0.0, -q.z, q.y, q.z, 0.0, -q.x, -q.y, q.x, 0.0],
        }
    }

    /// Creates a matrix representing the outer product of two 3D vectors.
    ///
    /// The outer product of vectors `a` and `b` is the matrix `M = a ⊗ b` where
    /// each element `M[i,j] = a[i] * b[j]`. For 3D vectors, this produces a 3×3 matrix.
    ///
    /// The resulting matrix has the following structure:
    /// ```text
    /// [ a.x*b.x  a.x*b.y  a.x*b.z ]
    /// [ a.y*b.x  a.y*b.y  a.y*b.z ]
    /// [ a.z*b.x  a.z*b.y  a.z*b.z ]
    /// ```
    ///
    /// # Applications
    ///
    /// The outer product has various applications in physics, computer graphics, and numerical methods:
    /// - Constructing projection matrices
    /// - Forming part of tensor products
    /// - Computing dyadic products in physics
    /// - Creating rotation matrices from quaternions
    ///
    /// # Arguments
    ///
    /// * `a` - First 3D vector
    /// * `b` - Second 3D vector
    ///
    /// # Returns
    ///
    /// A 3×3 matrix representing the outer product of the input vectors
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
    /// The transpose of a matrix swaps its rows and columns. For a 3x3 matrix \( A \), the element
    /// at position \((i, j)\) in \( A \) becomes the element at position \((j, i)\) in the transpose \( A^T \).
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let matrix = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let transposed = matrix.transpose();
    /// assert_eq!(transposed.data, [1.0, 4.0, 7.0, 2.0, 5.0, 8.0, 3.0, 6.0, 9.0]);
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
    /// For a 3x3 matrix:
    ///
    /// \[
    /// A = \begin{bmatrix}
    /// a & b & c \\
    /// d & e & f \\
    /// g & h & i
    /// \end{bmatrix},
    /// \]
    ///
    /// the determinant is calculated as:
    ///
    /// \[
    /// \text{det}(A) = a(ei - fh) - d(bi - ch) + g(bf - ce)
    /// \]
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let matrix = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let det = matrix.determinant();
    /// assert_eq!(det, 0.0); // 1*(5*9 - 6*8) - 4*(2*9 - 3*8) + 7*(2*6 - 3*5) = 0
    /// ```
    pub fn determinant(&self) -> f32 {
        let data = &self.data;

        data[0] * (data[4] * data[8] - data[7] * data[5])
            - data[3] * (data[1] * data[8] - data[7] * data[2])
            + data[6] * (data[1] * data[5] - data[4] * data[2])
    }

    /// Computes the trace of the matrix.
    ///
    /// The trace of a matrix is the sum of the elements on its main diagonal, i.e., \( a_{11} + a_{22} + a_{33} \).
    ///
    /// # Examples
    ///
    /// ```
    /// use robomath::Mat3x3;
    /// let matrix = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
    /// let trace = matrix.trace();
    /// assert_eq!(trace, 15.0); // 1 + 5 + 9 = 15
    /// ```
    pub fn trace(&self) -> f32 {
        self.data[0] + self.data[4] + self.data[8]
    }
}

/// Provides indexing into the matrix using `[row, col]` syntax.
///
/// The matrix is stored in row-major order, so the element at position \((i, j)\) corresponds to
/// the flat array index \( i \cdot 3 + j \). Indices must be in the range \([0, 2]\) for both row
/// and column; otherwise, the program will panic at runtime.
///
/// # Examples
///
/// ```
/// use robomath::Mat3x3;
/// let matrix = Mat3x3::new([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
/// assert_eq!(matrix[[0, 1]], 2.0); // Element at row 0, column 1
/// assert_eq!(matrix[[2, 2]], 9.0); // Element at row 2, column 2
/// ```
impl Index<[u32; 2]> for Mat3x3 {
    type Output = f32;

    fn index(&self, ndx: [u32; 2]) -> &Self::Output {
        &self.data[ndx[0] as usize * 3 + ndx[1] as usize]
    }
}

impl Mul<f32> for Mat3x3 {
    type Output = Mat3x3;

    fn mul(self, scalar: f32) -> Mat3x3 {
        let mut data = [0.0; 9];
        for i in 0..9 {
            data[i] = self.data[i] * scalar;
        }
        Mat3x3 { data }
    }
}

impl Add<Mat3x3> for Mat3x3 {
    type Output = Mat3x3;

    fn add(self, rhs: Mat3x3) -> Mat3x3 {
        let mut data = [0.0; 9];
        for (i, item) in data.iter_mut().enumerate() {
            *item = self.data[i] + rhs.data[i];
        }
        Mat3x3 { data }
    }
}
