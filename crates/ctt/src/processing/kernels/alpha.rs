//! Alpha premultiplication / unpremultiplication on pipeline buffers.
//!
//! The loops are plain scalar code; each entry point runs them under
//! `dispatch!` so LLVM autovectorizes them at the best target-feature level
//! the host supports.

use fearless_simd::{Level, dispatch};

use crate::processing::Buffer;

/// In-place: RGB *= A. Used after load when the source is
/// [`AlphaMode::Straight`](crate::alpha::AlphaMode::Straight).
pub fn premultiply_f32(buf: &mut Buffer<f32>) {
    profiling::scope!("premultiply_f32");
    dispatch!(Level::new(), _simd => premultiply_rows_f32(&mut buf.pixels));
}

/// In-place: RGB /= A. Used before store when the target is
/// [`AlphaMode::Straight`](crate::alpha::AlphaMode::Straight).
pub fn unpremultiply_f32(buf: &mut Buffer<f32>) {
    profiling::scope!("unpremultiply_f32");
    dispatch!(Level::new(), _simd => unpremultiply_rows_f32(&mut buf.pixels));
}

pub fn premultiply_f64(buf: &mut Buffer<f64>) {
    profiling::scope!("premultiply_f64");
    dispatch!(Level::new(), _simd => premultiply_rows_f64(&mut buf.pixels));
}

pub fn unpremultiply_f64(buf: &mut Buffer<f64>) {
    profiling::scope!("unpremultiply_f64");
    dispatch!(Level::new(), _simd => unpremultiply_rows_f64(&mut buf.pixels));
}

#[inline(always)]
fn premultiply_rows_f32(pixels: &mut [[f32; 4]]) {
    for p in pixels {
        let a = p[3];
        p[0] *= a;
        p[1] *= a;
        p[2] *= a;
    }
}

#[inline(always)]
fn unpremultiply_rows_f32(pixels: &mut [[f32; 4]]) {
    for p in pixels {
        // Selecting the divisor (not the quotient) keeps the loop branchless
        // and vectorizable while dividing by exactly `a` when it applies.
        let d = if p[3] > 0.0 { p[3] } else { 1.0 };
        p[0] /= d;
        p[1] /= d;
        p[2] /= d;
    }
}

#[inline(always)]
fn premultiply_rows_f64(pixels: &mut [[f64; 4]]) {
    for p in pixels {
        let a = p[3];
        p[0] *= a;
        p[1] *= a;
        p[2] *= a;
    }
}

#[inline(always)]
fn unpremultiply_rows_f64(pixels: &mut [[f64; 4]]) {
    for p in pixels {
        let d = if p[3] > 0.0 { p[3] } else { 1.0 };
        p[0] /= d;
        p[1] /= d;
        p[2] /= d;
    }
}
