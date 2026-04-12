use bytemuck::Pod;

use crate::vk_format::ChannelKind;

/// A numeric type that can be read from / written to a pixel buffer as a single sample.
///
/// Implementations exist for the four sample types used in uncompressed texture formats:
/// `u8`, `u16`, [`half::f16`], and `f32`.
///
/// The trait is `'static` so that [`TypeId`][std::any::TypeId]-based specialization works — comparisons on
/// `TypeId::of::<S>()` are const-folded during monomorphization, giving zero-cost dispatch.
pub trait Sample: 'static + Copy + Pod {
    /// The [`ChannelKind`] that corresponds to this sample type.
    const KIND: ChannelKind;

    /// Size of one sample in bytes.
    const BYTE_SIZE: usize;

    /// Convert a typed sample value to `f32`.
    ///
    /// Integer types are normalized to `[0, 1]`.
    fn to_f32(self) -> f32;

    /// Convert an `f32` value to a typed sample.
    ///
    /// Integer types clamp and quantize from `[0, 1]`.
    fn from_f32(val: f32) -> Self;

    /// Read one sample from a byte buffer at the given offset.
    fn read(data: &[u8], offset: usize) -> f32;

    /// Write one sample to a byte buffer at the given offset.
    fn write(data: &mut [u8], offset: usize, val: f32);
}

impl Sample for u8 {
    const KIND: ChannelKind = ChannelKind::U8;
    const BYTE_SIZE: usize = 1;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 255.0
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        (val.clamp(0.0, 1.0) * 255.0).round() as u8
    }

    #[inline(always)]
    fn read(data: &[u8], offset: usize) -> f32 {
        data[offset].to_f32()
    }

    #[inline(always)]
    fn write(data: &mut [u8], offset: usize, val: f32) {
        data[offset] = Self::from_f32(val);
    }
}

impl Sample for u16 {
    const KIND: ChannelKind = ChannelKind::U16;
    const BYTE_SIZE: usize = 2;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self as f32 / 65535.0
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        (val.clamp(0.0, 1.0) * 65535.0).round() as u16
    }

    #[inline(always)]
    fn read(data: &[u8], offset: usize) -> f32 {
        let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
        v.to_f32()
    }

    #[inline(always)]
    fn write(data: &mut [u8], offset: usize, val: f32) {
        let v = Self::from_f32(val);
        data[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
    }
}

impl Sample for half::f16 {
    const KIND: ChannelKind = ChannelKind::F16;
    const BYTE_SIZE: usize = 2;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        half::f16::to_f32(self)
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        half::f16::from_f32(val)
    }

    #[inline(always)]
    fn read(data: &[u8], offset: usize) -> f32 {
        let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
        half::f16::from_bits(bits).to_f32()
    }

    #[inline(always)]
    fn write(data: &mut [u8], offset: usize, val: f32) {
        let h = half::f16::from_f32(val);
        data[offset..offset + 2].copy_from_slice(&h.to_le_bytes());
    }
}

impl Sample for f32 {
    const KIND: ChannelKind = ChannelKind::F32;
    const BYTE_SIZE: usize = 4;

    #[inline(always)]
    fn to_f32(self) -> f32 {
        self
    }

    #[inline(always)]
    fn from_f32(val: f32) -> Self {
        val
    }

    #[inline(always)]
    fn read(data: &[u8], offset: usize) -> f32 {
        f32::from_le_bytes([
            data[offset],
            data[offset + 1],
            data[offset + 2],
            data[offset + 3],
        ])
    }

    #[inline(always)]
    fn write(data: &mut [u8], offset: usize, val: f32) {
        data[offset..offset + 4].copy_from_slice(&val.to_le_bytes());
    }
}

/// Dispatch a call to `$f::<S>($($args),*)` based on a [`ChannelKind`] value.
///
/// ```ignore
/// dispatch_sample!(kind, my_function(arg1, arg2))
/// ```
#[expect(unused_macros)]
macro_rules! dispatch_sample {
    ($kind:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match $kind {
            $crate::vk_format::ChannelKind::U8 => $f::<u8>($($arg),*),
            $crate::vk_format::ChannelKind::U16 => $f::<u16>($($arg),*),
            $crate::vk_format::ChannelKind::F16 => $f::<half::f16>($($arg),*),
            $crate::vk_format::ChannelKind::F32 => $f::<f32>($($arg),*),
            $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
        }
    };
}

#[expect(unused_imports)]
pub(crate) use dispatch_sample;

/// Double-dispatch on two [`ChannelKind`] values: `$f::<S, D>($($args),*)`.
///
/// ```ignore
/// dispatch_sample2!(src_kind, dst_kind, my_function(arg1, arg2))
/// ```
#[expect(unused_macros)]
macro_rules! dispatch_sample2 {
    ($src_kind:expr, $dst_kind:expr, $f:ident ( $($arg:expr),* $(,)? )) => {
        match $src_kind {
            $crate::vk_format::ChannelKind::U8 => match $dst_kind {
                $crate::vk_format::ChannelKind::U8 => $f::<u8, u8>($($arg),*),
                $crate::vk_format::ChannelKind::U16 => $f::<u8, u16>($($arg),*),
                $crate::vk_format::ChannelKind::F16 => $f::<u8, half::f16>($($arg),*),
                $crate::vk_format::ChannelKind::F32 => $f::<u8, f32>($($arg),*),
                $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
            },
            $crate::vk_format::ChannelKind::U16 => match $dst_kind {
                $crate::vk_format::ChannelKind::U8 => $f::<u16, u8>($($arg),*),
                $crate::vk_format::ChannelKind::U16 => $f::<u16, u16>($($arg),*),
                $crate::vk_format::ChannelKind::F16 => $f::<u16, half::f16>($($arg),*),
                $crate::vk_format::ChannelKind::F32 => $f::<u16, f32>($($arg),*),
                $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
            },
            $crate::vk_format::ChannelKind::F16 => match $dst_kind {
                $crate::vk_format::ChannelKind::U8 => $f::<half::f16, u8>($($arg),*),
                $crate::vk_format::ChannelKind::U16 => $f::<half::f16, u16>($($arg),*),
                $crate::vk_format::ChannelKind::F16 => $f::<half::f16, half::f16>($($arg),*),
                $crate::vk_format::ChannelKind::F32 => $f::<half::f16, f32>($($arg),*),
                $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
            },
            $crate::vk_format::ChannelKind::F32 => match $dst_kind {
                $crate::vk_format::ChannelKind::U8 => $f::<f32, u8>($($arg),*),
                $crate::vk_format::ChannelKind::U16 => $f::<f32, u16>($($arg),*),
                $crate::vk_format::ChannelKind::F16 => $f::<f32, half::f16>($($arg),*),
                $crate::vk_format::ChannelKind::F32 => $f::<f32, f32>($($arg),*),
                $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
            },
            $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
        }
    };
}

#[expect(unused_imports)]
pub(crate) use dispatch_sample2;

/// Triple-dispatch on two [`ChannelKind`] values and a shared channel count:
/// `$f::<S, D, N>($($args),*)`.
///
/// Extracts channel kind and count from both formats, asserts channel counts match,
/// then dispatches across `S × D × N`.
///
/// Requires [`FormatExt`](crate::vk_format::FormatExt) to be in scope at the call site.
///
/// ```ignore
/// dispatch_sample3!(src_format, dst_format, my_function(arg1, arg2))
/// ```
macro_rules! dispatch_sample3 {
    ($src_fmt:expr, $dst_fmt:expr, $f:ident ( $($arg:expr),* $(,)? )) => {{
        let __src_fmt = $src_fmt;
        let __dst_fmt = $dst_fmt;
        let __src_ck = __src_fmt.channel_kind().expect("unknown src channel kind");
        let __dst_ck = __dst_fmt.channel_kind().expect("unknown dst channel kind");
        let __cc = __src_fmt.channel_count().expect("unknown src channel count");
        debug_assert_eq!(
            __cc,
            __dst_fmt.channel_count().expect("unknown dst channel count"),
            "dispatch_sample3: channel counts must match"
        );

        macro_rules! __dispatch_sd {
            ($n:literal) => {
                match __src_ck {
                    $crate::vk_format::ChannelKind::U8 => match __dst_ck {
                        $crate::vk_format::ChannelKind::U8 => $f::<u8, u8, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U16 => $f::<u8, u16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F16 => $f::<u8, half::f16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F32 => $f::<u8, f32, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
                    },
                    $crate::vk_format::ChannelKind::U16 => match __dst_ck {
                        $crate::vk_format::ChannelKind::U8 => $f::<u16, u8, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U16 => $f::<u16, u16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F16 => $f::<u16, half::f16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F32 => $f::<u16, f32, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
                    },
                    $crate::vk_format::ChannelKind::F16 => match __dst_ck {
                        $crate::vk_format::ChannelKind::U8 => $f::<half::f16, u8, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U16 => $f::<half::f16, u16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F16 => $f::<half::f16, half::f16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F32 => $f::<half::f16, f32, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
                    },
                    $crate::vk_format::ChannelKind::F32 => match __dst_ck {
                        $crate::vk_format::ChannelKind::U8 => $f::<f32, u8, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U16 => $f::<f32, u16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F16 => $f::<f32, half::f16, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::F32 => $f::<f32, f32, $n>($($arg),*),
                        $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
                    },
                    $crate::vk_format::ChannelKind::U32 => unreachable!("U32 not supported as a Sample type"),
                }
            };
        }

        match __cc {
            1 => __dispatch_sd!(1),
            2 => __dispatch_sd!(2),
            3 => __dispatch_sd!(3),
            4 => __dispatch_sd!(4),
            _ => unreachable!("unsupported channel count {}", __cc),
        }
    }};
}

pub(crate) use dispatch_sample3;
