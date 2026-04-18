//! Alpha premultiplication / unpremultiplication on pipeline buffers.

use super::buffer::Buffer;

/// In-place: RGB *= A. Used after load when the source is
/// [`AlphaMode::Straight`](crate::alpha::AlphaMode::Straight).
pub fn premultiply_f32(buf: &mut Buffer<f32>) {
    profiling::scope!("premultiply_f32");
    for p in buf.pixels.iter_mut() {
        let a = p[3];
        p[0] *= a;
        p[1] *= a;
        p[2] *= a;
    }
}

/// In-place: RGB /= A. Used before store when the target is
/// [`AlphaMode::Straight`](crate::alpha::AlphaMode::Straight).
pub fn unpremultiply_f32(buf: &mut Buffer<f32>) {
    profiling::scope!("unpremultiply_f32");
    for p in buf.pixels.iter_mut() {
        let a = p[3];
        if a > 0.0 {
            p[0] /= a;
            p[1] /= a;
            p[2] /= a;
        }
    }
}

pub fn premultiply_f64(buf: &mut Buffer<f64>) {
    profiling::scope!("premultiply_f64");
    for p in buf.pixels.iter_mut() {
        let a = p[3];
        p[0] *= a;
        p[1] *= a;
        p[2] *= a;
    }
}

pub fn unpremultiply_f64(buf: &mut Buffer<f64>) {
    profiling::scope!("unpremultiply_f64");
    for p in buf.pixels.iter_mut() {
        let a = p[3];
        if a > 0.0 {
            p[0] /= a;
            p[1] /= a;
            p[2] /= a;
        }
    }
}
