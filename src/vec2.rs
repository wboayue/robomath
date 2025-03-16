use core::ops::{Add, Div, Mul, Neg, Sub};
// use libm::{F32Ext, cosf, sinf, sqrtf, atan2f, asinf};

#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct Vec2<T: Default> {
    pub x: T,
    pub y: T,
}

impl Vec2<f32> {
    pub fn clamp(&self, min: f32, max: f32) -> Vec2<f32> {
        Vec2 {
            x: self.x.clamp(min, max),
            y: self.y.clamp(min, max),
        }
    }
}

// Helper function to create Vec3 instances easily
pub fn vec2<T: Default>(x: T, y: T) -> Vec2<T> {
    Vec2 { x, y }
}

impl<T: Sub<Output = T> + Default> Sub for Vec2<T> {
    type Output = Vec2<T>;

    fn sub(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x - rhs.x,
            y: self.y - rhs.y,
        }
    }
}

impl<T: Add<Output = T> + Default> Add for Vec2<T> {
    type Output = Vec2<T>;

    fn add(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x + rhs.x,
            y: self.y + rhs.y,
        }
    }
}

impl<T: Mul<Output = T> + Default> Mul for Vec2<T> {
    type Output = Vec2<T>;

    fn mul(self, rhs: Vec2<T>) -> Vec2<T> {
        Vec2 {
            x: self.x * rhs.x,
            y: self.y * rhs.y,
        }
    }
}

// x * vec2
impl Mul<Vec2<f32>> for f32 {
    type Output = Vec2<f32>;

    fn mul(self, rhs: Vec2<f32>) -> Vec2<f32> {
        Vec2 {
            x: self * rhs.x,
            y: self * rhs.y,
        }
    }
}

// vec3 / x
impl<T: Div<Output = T> + Copy + Default> Div<T> for Vec2<T> {
    type Output = Vec2<T>;

    fn div(self, rhs: T) -> Vec2<T> {
        Vec2 {
            x: self.x / rhs,
            y: self.y / rhs,
        }
    }
}

impl<T: Default> Default for Vec2<T> {
    fn default() -> Self {
        Vec2 {
            x: T::default(),
            y: T::default(),
        }
    }
}

impl<T: Neg<Output = T> + Default> Neg for Vec2<T> {
    type Output = Vec2<T>;

    fn neg(self) -> Self::Output {
        Vec2 {
            x: -self.x,
            y: -self.y,
        }
    }
}
