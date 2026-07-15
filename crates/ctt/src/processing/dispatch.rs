//! Runtime SIMD kernel dispatch.

/// Try each listed SIMD kernel in descending preference order, calling the
/// first one whose ISA the host supports and returning its result. Falls
/// through (no return) when none is available, so the caller writes its
/// serial fallback directly after the invocation:
///
/// ```ignore
/// pub fn load_foo_f32(surface: &Surface) -> Result<Buffer<f32>> {
///     dispatch_simd! {
///         x86_64: {
///             avx512: load_foo_f32_avx512(surface),
///             avx2_fma: load_foo_f32_avx2_fma(surface),
///             sse4_1: load_foo_f32_sse4_1(surface),
///         },
///         aarch64: {
///             neon: load_foo_f32_neon(surface),
///         },
///     }
///     load_foo_f32_serial(surface)
/// }
/// ```
///
/// Every tier is optional; `avx2` (without FMA) is available for kernels
/// that don't use fused multiply-adds. Each arm is a call to an `unsafe fn`
/// whose only safety requirement is the ISA named by its tier — the macro
/// pairs the runtime feature check with the call, which is what makes the
/// `unsafe` sound.
macro_rules! dispatch_simd {
    (
        $(x86_64: {
            $(avx512: $avx512:expr,)?
            $(avx2_fma: $avx2_fma:expr,)?
            $(avx2: $avx2:expr,)?
            $(sse4_1: $sse4_1:expr,)?
        },)?
        $(aarch64: {
            neon: $neon:expr $(,)?
        },)?
    ) => {
        #[cfg(target_arch = "x86_64")]
        {
            $(
                $(if $crate::processing::x86::has_avx512() {
                    // SAFETY: the check above confirms avx512f+vl+bw.
                    return unsafe { $avx512 };
                })?
                $(if $crate::processing::x86::has_avx2_fma() {
                    // SAFETY: the check above confirms avx2+fma.
                    return unsafe { $avx2_fma };
                })?
                $(if is_x86_feature_detected!("avx2") {
                    // SAFETY: the check above confirms avx2.
                    return unsafe { $avx2 };
                })?
                $(if is_x86_feature_detected!("sse4.1") {
                    // SAFETY: the check above confirms sse4.1.
                    return unsafe { $sse4_1 };
                })?
            )?
        }

        #[cfg(target_arch = "aarch64")]
        {
            $(if std::arch::is_aarch64_feature_detected!("neon") {
                // SAFETY: the check above confirms NEON.
                return unsafe { $neon };
            })?
        }
    };
}
pub(crate) use dispatch_simd;
