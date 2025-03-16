use core::ops::{Add, Div, Mul, Sub};

use libm::sqrtf;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Vec3<T> {
    pub x: T,
    pub y: T,
    pub z: T,
}

impl Vec3<f32> {
    pub fn clamp(&self, min: f32, max: f32) -> Vec3<f32> {
        Vec3 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
            z: self.z.clamp(min, max),
        }
    }
    pub fn dot(&self, rhs: Vec3<f32>) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    pub fn cross(&self, rhs: Vec3<f32>) -> Vec3<f32> {
        vec3(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    pub fn is_finite(&self) -> bool {
        self.x.is_finite() && self.y.is_finite() && self.z.is_finite()
    }
}

// Helper function to create Vec3 instances easily
pub fn vec3<T>(x: T, y: T, z: T) -> Vec3<T> {
    Vec3 { x, y, z }
}

impl<T: Sub<Output = T>> Sub for Vec3<T> {
    type Output = Vec3<T>;

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

    fn mul(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
            z: self.z * rhs.z,
        }
    }
}

// x * vec3
impl Mul<Vec3<f32>> for f32 {
    type Output = Vec3<f32>;

    fn mul(self, rhs: Vec3<f32>) -> Vec3<f32> {
        Vec3 {
            x: self * rhs.x,
            y: self * rhs.y,
            z: self * rhs.z,
        }
    }
}

// vec3 * x
impl Mul<f32> for Vec3<f32> {
    type Output = Vec3<f32>;

    fn mul(self, rhs: f32) -> Vec3<f32> {
        Vec3 {
            x: self.x * rhs,
            y: self.y * rhs,
            z: self.z * rhs,
        }
    }
}

// vec3 / x
impl<T: Div<Output = T> + Copy> Div<T> for Vec3<T> {
    type Output = Vec3<T>;

    fn div(self, rhs: T) -> Vec3<T> {
        Vec3 {
            x: self.x / rhs,
            y: self.y / rhs,
            z: self.z / rhs,
        }
    }
}

// vec3 / vec3
impl<T: Div<Output = T> + Copy> Div<Vec3<T>> for Vec3<T> {
    type Output = Vec3<T>;

    fn div(self, rhs: Vec3<T>) -> Vec3<T> {
        Vec3 {
            x: self.x / rhs.x,
            y: self.y / rhs.y,
            z: self.z / rhs.z,
        }
    }
}

impl<T> Vec3<T> {
    pub fn magnitude_squared(&self) -> T
    where
        T: Mul<Output = T> + Add<Output = T> + Copy,
    {
        self.x * self.x + self.y * self.y + self.z * self.z
    }

    pub fn magnitude(&self) -> f32
    where
        T: Into<f32> + Mul<Output = T> + Add<Output = T> + Copy,
    {
        sqrtf(self.magnitude_squared().into())
    }
}

impl<T: Default> Default for Vec3<T> {
    fn default() -> Self {
        Vec3 {
            x: T::default(),
            y: T::default(),
            z: T::default(),
        }
    }
}
