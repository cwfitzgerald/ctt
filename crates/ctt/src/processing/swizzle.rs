//! Channel swizzle on any pipeline buffer (float or integer lanes).

use bytemuck::Pod;

use super::buffer::Buffer;

/// A single channel source for swizzling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwizzleChannel {
    R,
    G,
    B,
    A,
    Zero,
    One,
}

/// A 4-component swizzle pattern.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Swizzle(pub [SwizzleChannel; 4]);

impl Swizzle {
    pub const IDENTITY: Self = Self([
        SwizzleChannel::R,
        SwizzleChannel::G,
        SwizzleChannel::B,
        SwizzleChannel::A,
    ]);
}

/// In-place swizzle. `zero` and `one` are the values to substitute for
/// [`SwizzleChannel::Zero`] and [`SwizzleChannel::One`] in this buffer's
/// lane type.
pub fn apply<T: Copy + Pod>(buf: &mut Buffer<T>, swizzle: &Swizzle, zero: T, one: T) {
    if *swizzle == Swizzle::IDENTITY {
        return;
    }
    let src_of = |sw: SwizzleChannel, px: &[T; 4]| -> T {
        match sw {
            SwizzleChannel::R => px[0],
            SwizzleChannel::G => px[1],
            SwizzleChannel::B => px[2],
            SwizzleChannel::A => px[3],
            SwizzleChannel::Zero => zero,
            SwizzleChannel::One => one,
        }
    };
    for p in buf.pixels.iter_mut() {
        let src = *p;
        p[0] = src_of(swizzle.0[0], &src);
        p[1] = src_of(swizzle.0[1], &src);
        p[2] = src_of(swizzle.0[2], &src);
        p[3] = src_of(swizzle.0[3], &src);
    }
}

/// Apply swizzle to a f32 buffer using `0.0` / `1.0` for `Zero`/`One`.
pub fn apply_f32(buf: &mut Buffer<f32>, swizzle: &Swizzle) {
    profiling::scope!("swizzle_f32");
    apply(buf, swizzle, 0.0, 1.0);
}

pub fn apply_f64(buf: &mut Buffer<f64>, swizzle: &Swizzle) {
    profiling::scope!("swizzle_f64");
    apply(buf, swizzle, 0.0, 1.0);
}

pub fn apply_u32(buf: &mut Buffer<u32>, swizzle: &Swizzle) {
    profiling::scope!("swizzle_u32");
    apply(buf, swizzle, 0, u32::MAX);
}

pub fn apply_u64(buf: &mut Buffer<u64>, swizzle: &Swizzle) {
    profiling::scope!("swizzle_u64");
    apply(buf, swizzle, 0, u64::MAX);
}
