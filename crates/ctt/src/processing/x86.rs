//! Shared x86 SIMD layout operations used by load and store kernels.

use std::arch::x86_64::*;

/// Runtime check for the AVX-512 feature set the kernels rely on.
pub fn has_avx512() -> bool {
    is_x86_feature_detected!("avx512f")
        && is_x86_feature_detected!("avx512vl")
        && is_x86_feature_detected!("avx512bw")
}

/// Transpose four packed float vectors. The operation is its own inverse, so
/// it serves both AoS-to-SoA and SoA-to-AoS conversions.
#[target_feature(enable = "sse")]
#[inline]
pub(super) unsafe fn transpose4_ps(
    mut a: __m128,
    mut b: __m128,
    mut c: __m128,
    mut d: __m128,
) -> (__m128, __m128, __m128, __m128) {
    _MM_TRANSPOSE4_PS(&mut a, &mut b, &mut c, &mut d);
    (a, b, c, d)
}

/// Load four interleaved f32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 16 f32 reads.
#[target_feature(enable = "sse")]
#[inline]
pub(super) unsafe fn load_planar4_ps(src: *const f32) -> (__m128, __m128, __m128, __m128) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        transpose4_ps(
            _mm_loadu_ps(src),
            _mm_loadu_ps(src.add(4)),
            _mm_loadu_ps(src.add(8)),
            _mm_loadu_ps(src.add(12)),
        )
    }
}

/// Load four interleaved u32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 16 u32 reads.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) unsafe fn load_planar4_epi32(src: *const u32) -> (__m128i, __m128i, __m128i, __m128i) {
    // Integer and float transposes are the same bitwise shuffle.
    // SAFETY: guaranteed by the caller; x86 casts preserve all lane bits.
    let (a, b, c, d) = unsafe {
        transpose4_ps(
            _mm_castsi128_ps(_mm_loadu_si128(src.cast())),
            _mm_castsi128_ps(_mm_loadu_si128(src.add(4).cast())),
            _mm_castsi128_ps(_mm_loadu_si128(src.add(8).cast())),
            _mm_castsi128_ps(_mm_loadu_si128(src.add(12).cast())),
        )
    };
    (
        _mm_castps_si128(a),
        _mm_castps_si128(b),
        _mm_castps_si128(c),
        _mm_castps_si128(d),
    )
}

/// Store up to four planar f32 pixels in interleaved order.
///
/// # Safety
/// `dst` must be valid for `count * 4` f32 writes and `count` must be at most 4.
#[target_feature(enable = "sse")]
#[inline]
pub(super) unsafe fn store_interleaved_ps(
    r: __m128,
    g: __m128,
    b: __m128,
    a: __m128,
    dst: *mut f32,
    count: usize,
) {
    debug_assert!(count <= 4);
    // SAFETY: guaranteed by the caller.
    let (p0, p1, p2, p3) = unsafe { transpose4_ps(r, g, b, a) };
    if count > 0 {
        unsafe { _mm_storeu_ps(dst, p0) };
    }
    if count > 1 {
        unsafe { _mm_storeu_ps(dst.add(4), p1) };
    }
    if count > 2 {
        unsafe { _mm_storeu_ps(dst.add(8), p2) };
    }
    if count > 3 {
        unsafe { _mm_storeu_ps(dst.add(12), p3) };
    }
}

/// Store four planar f32 pixels in interleaved order.
///
/// # Safety
/// `dst` must be valid for 16 f32 writes.
#[target_feature(enable = "sse")]
#[inline]
pub(super) unsafe fn store_interleaved4_ps(
    r: __m128,
    g: __m128,
    b: __m128,
    a: __m128,
    dst: *mut f32,
) {
    // SAFETY: guaranteed by the caller.
    unsafe { store_interleaved_ps(r, g, b, a, dst, 4) }
}

/// Store four planar u32 pixels in interleaved order.
///
/// # Safety
/// `dst` must be valid for 16 u32 writes.
#[target_feature(enable = "sse2")]
#[inline]
pub(super) unsafe fn store_interleaved4_epi32(
    r: __m128i,
    g: __m128i,
    b: __m128i,
    a: __m128i,
    dst: *mut u32,
) {
    // Integer and float transposes are the same bitwise shuffle.
    // SAFETY: x86 casts preserve all lane bits.
    let (p0, p1, p2, p3) = unsafe {
        transpose4_ps(
            _mm_castsi128_ps(r),
            _mm_castsi128_ps(g),
            _mm_castsi128_ps(b),
            _mm_castsi128_ps(a),
        )
    };
    // SAFETY: the caller guarantees space for all four pixels.
    unsafe {
        _mm_storeu_si128(dst.cast(), _mm_castps_si128(p0));
        _mm_storeu_si128(dst.add(4).cast(), _mm_castps_si128(p1));
        _mm_storeu_si128(dst.add(8).cast(), _mm_castps_si128(p2));
        _mm_storeu_si128(dst.add(12).cast(), _mm_castps_si128(p3));
    }
}

/// Join two four-lane float vectors in low-to-high order.
#[target_feature(enable = "avx")]
#[inline]
pub(super) fn combine2_ps(low: __m128, high: __m128) -> __m256 {
    _mm256_set_m128(high, low)
}

/// Join two four-lane integer vectors in low-to-high order.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) fn combine2_epi32(low: __m128i, high: __m128i) -> __m256i {
    _mm256_set_m128i(high, low)
}

/// Load eight interleaved f32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 32 f32 reads.
#[target_feature(enable = "avx")]
#[inline]
pub(super) unsafe fn load_planar8_ps(src: *const f32) -> (__m256, __m256, __m256, __m256) {
    // SAFETY: guaranteed by the caller.
    let (r0, g0, b0, a0) = unsafe { load_planar4_ps(src) };
    let (r1, g1, b1, a1) = unsafe { load_planar4_ps(src.add(16)) };
    (
        combine2_ps(r0, r1),
        combine2_ps(g0, g1),
        combine2_ps(b0, b1),
        combine2_ps(a0, a1),
    )
}

/// Load eight interleaved u32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 32 u32 reads.
#[target_feature(enable = "avx2")]
#[inline]
pub(super) unsafe fn load_planar8_epi32(src: *const u32) -> (__m256i, __m256i, __m256i, __m256i) {
    // SAFETY: guaranteed by the caller.
    let (r0, g0, b0, a0) = unsafe { load_planar4_epi32(src) };
    let (r1, g1, b1, a1) = unsafe { load_planar4_epi32(src.add(16)) };
    (
        combine2_epi32(r0, r1),
        combine2_epi32(g0, g1),
        combine2_epi32(b0, b1),
        combine2_epi32(a0, a1),
    )
}

// ---- AVX-512 16-pixel SoA <-> AoS ----
//
// A 16-pixel block is four 512-bit vectors of interleaved `[R, G, B, A]`
// lanes. Both directions run as two levels of two-source permutes (8
// `vpermt2d`/`vpermt2pd` total), which beats assembling planes out of 128-bit
// transposes. Partial blocks use masked loads/stores; disabled lanes are never
// accessed, so a `pixels < 16` call may sit at the very end of a buffer.

/// Per-output-vector store masks covering `pixels * 4` interleaved lanes.
#[inline]
fn interleaved16_masks(pixels: usize) -> [__mmask16; 4] {
    debug_assert!(pixels <= 16);
    let lanes = pixels as u32 * 4;
    let live = 1u64.checked_shl(lanes).map_or(u64::MAX, |bit| bit - 1);
    std::array::from_fn(|k| (live >> (16 * k)) as __mmask16)
}

/// Interleave four 16-lane integer channel planes into four vectors of four
/// consecutive `[R, G, B, A]` pixels each.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn interleave16_epi32(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    // Stage 1: pair channels lane-wise — rg = [r0,g0 .. r7,g7] / [r8,g8 ..],
    // ba likewise.
    let lo = _mm512_setr_epi32(0, 16, 1, 17, 2, 18, 3, 19, 4, 20, 5, 21, 6, 22, 7, 23);
    let hi = _mm512_setr_epi32(8, 24, 9, 25, 10, 26, 11, 27, 12, 28, 13, 29, 14, 30, 15, 31);
    let rg_lo = _mm512_permutex2var_epi32(r, lo, g);
    let rg_hi = _mm512_permutex2var_epi32(r, hi, g);
    let ba_lo = _mm512_permutex2var_epi32(b, lo, a);
    let ba_hi = _mm512_permutex2var_epi32(b, hi, a);
    // Stage 2: interleave the (r,g) and (b,a) pairs at 64-bit granularity.
    let q_lo = _mm512_setr_epi64(0, 8, 1, 9, 2, 10, 3, 11);
    let q_hi = _mm512_setr_epi64(4, 12, 5, 13, 6, 14, 7, 15);
    (
        _mm512_permutex2var_epi64(rg_lo, q_lo, ba_lo),
        _mm512_permutex2var_epi64(rg_lo, q_hi, ba_lo),
        _mm512_permutex2var_epi64(rg_hi, q_lo, ba_hi),
        _mm512_permutex2var_epi64(rg_hi, q_hi, ba_hi),
    )
}

/// Split four vectors of interleaved `[R, G, B, A]` pixels into 16-lane
/// integer channel planes.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) fn deinterleave16_epi32(
    p0: __m512i,
    p1: __m512i,
    p2: __m512i,
    p3: __m512i,
) -> (__m512i, __m512i, __m512i, __m512i) {
    // Stage 1: split each vector pair into (r,g) and (b,a) 64-bit pairs.
    let q_even = _mm512_setr_epi64(0, 2, 4, 6, 8, 10, 12, 14);
    let q_odd = _mm512_setr_epi64(1, 3, 5, 7, 9, 11, 13, 15);
    let rg_lo = _mm512_permutex2var_epi64(p0, q_even, p1);
    let ba_lo = _mm512_permutex2var_epi64(p0, q_odd, p1);
    let rg_hi = _mm512_permutex2var_epi64(p2, q_even, p3);
    let ba_hi = _mm512_permutex2var_epi64(p2, q_odd, p3);
    // Stage 2: split the pairs into single-channel planes.
    let even = _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
    let odd = _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31);
    (
        _mm512_permutex2var_epi32(rg_lo, even, rg_hi),
        _mm512_permutex2var_epi32(rg_lo, odd, rg_hi),
        _mm512_permutex2var_epi32(ba_lo, even, ba_hi),
        _mm512_permutex2var_epi32(ba_lo, odd, ba_hi),
    )
}

/// Store sixteen planar f32 pixels in interleaved order.
///
/// # Safety
/// `dst` must be valid for 64 f32 writes.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn store_interleaved16_ps(
    r: __m512,
    g: __m512,
    b: __m512,
    a: __m512,
    dst: *mut f32,
) {
    let (p0, p1, p2, p3) = interleave16_epi32(
        _mm512_castps_si512(r),
        _mm512_castps_si512(g),
        _mm512_castps_si512(b),
        _mm512_castps_si512(a),
    );
    // SAFETY: guaranteed by the caller.
    unsafe {
        _mm512_storeu_si512(dst.cast(), p0);
        _mm512_storeu_si512(dst.add(16).cast(), p1);
        _mm512_storeu_si512(dst.add(32).cast(), p2);
        _mm512_storeu_si512(dst.add(48).cast(), p3);
    }
}

/// Store the first `pixels` (≤ 16) of sixteen planar f32 pixels in interleaved
/// order via masked stores.
///
/// # Safety
/// `dst` must be valid for `pixels * 4` f32 writes.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn store_interleaved16_mask_ps(
    r: __m512,
    g: __m512,
    b: __m512,
    a: __m512,
    dst: *mut f32,
    pixels: usize,
) {
    let (p0, p1, p2, p3) = interleave16_epi32(
        _mm512_castps_si512(r),
        _mm512_castps_si512(g),
        _mm512_castps_si512(b),
        _mm512_castps_si512(a),
    );
    let [m0, m1, m2, m3] = interleaved16_masks(pixels);
    // SAFETY: masked stores only touch enabled lanes, all within
    // `pixels * 4` per the caller.
    unsafe {
        _mm512_mask_storeu_epi32(dst.cast(), m0, p0);
        _mm512_mask_storeu_epi32(dst.add(16).cast(), m1, p1);
        _mm512_mask_storeu_epi32(dst.add(32).cast(), m2, p2);
        _mm512_mask_storeu_epi32(dst.add(48).cast(), m3, p3);
    }
}

/// Store sixteen planar u32 pixels in interleaved order.
///
/// # Safety
/// `dst` must be valid for 64 u32 writes.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn store_interleaved16_epi32(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
    dst: *mut u32,
) {
    let (p0, p1, p2, p3) = interleave16_epi32(r, g, b, a);
    // SAFETY: guaranteed by the caller.
    unsafe {
        _mm512_storeu_si512(dst.cast(), p0);
        _mm512_storeu_si512(dst.add(16).cast(), p1);
        _mm512_storeu_si512(dst.add(32).cast(), p2);
        _mm512_storeu_si512(dst.add(48).cast(), p3);
    }
}

/// Store the first `pixels` (≤ 16) of sixteen planar u32 pixels in interleaved
/// order via masked stores.
///
/// # Safety
/// `dst` must be valid for `pixels * 4` u32 writes.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn store_interleaved16_mask_epi32(
    r: __m512i,
    g: __m512i,
    b: __m512i,
    a: __m512i,
    dst: *mut u32,
    pixels: usize,
) {
    let (p0, p1, p2, p3) = interleave16_epi32(r, g, b, a);
    let [m0, m1, m2, m3] = interleaved16_masks(pixels);
    // SAFETY: masked stores only touch enabled lanes, all within
    // `pixels * 4` per the caller.
    unsafe {
        _mm512_mask_storeu_epi32(dst.cast(), m0, p0);
        _mm512_mask_storeu_epi32(dst.add(16).cast(), m1, p1);
        _mm512_mask_storeu_epi32(dst.add(32).cast(), m2, p2);
        _mm512_mask_storeu_epi32(dst.add(48).cast(), m3, p3);
    }
}

/// Load sixteen interleaved f32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 64 f32 reads.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn load_planar16_ps(src: *const f32) -> (__m512, __m512, __m512, __m512) {
    // SAFETY: guaranteed by the caller.
    let (r, g, b, a) = unsafe {
        deinterleave16_epi32(
            _mm512_loadu_si512(src.cast()),
            _mm512_loadu_si512(src.add(16).cast()),
            _mm512_loadu_si512(src.add(32).cast()),
            _mm512_loadu_si512(src.add(48).cast()),
        )
    };
    (
        _mm512_castsi512_ps(r),
        _mm512_castsi512_ps(g),
        _mm512_castsi512_ps(b),
        _mm512_castsi512_ps(a),
    )
}

/// Load the first `pixels` (≤ 16) interleaved f32 pixels via masked loads and
/// return their channel planes. Lanes past `pixels` are zero.
///
/// # Safety
/// `src` must be valid for `pixels * 4` f32 reads.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn load_planar16_mask_ps(
    src: *const f32,
    pixels: usize,
) -> (__m512, __m512, __m512, __m512) {
    let [m0, m1, m2, m3] = interleaved16_masks(pixels);
    // SAFETY: masked loads only touch enabled lanes, all within `pixels * 4`
    // per the caller.
    let (r, g, b, a) = unsafe {
        deinterleave16_epi32(
            _mm512_maskz_loadu_epi32(m0, src.cast()),
            _mm512_maskz_loadu_epi32(m1, src.add(16).cast()),
            _mm512_maskz_loadu_epi32(m2, src.add(32).cast()),
            _mm512_maskz_loadu_epi32(m3, src.add(48).cast()),
        )
    };
    (
        _mm512_castsi512_ps(r),
        _mm512_castsi512_ps(g),
        _mm512_castsi512_ps(b),
        _mm512_castsi512_ps(a),
    )
}

/// Load sixteen interleaved u32 pixels and return their channel planes.
///
/// # Safety
/// `src` must be valid for 64 u32 reads.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn load_planar16_epi32(src: *const u32) -> (__m512i, __m512i, __m512i, __m512i) {
    // SAFETY: guaranteed by the caller.
    unsafe {
        deinterleave16_epi32(
            _mm512_loadu_si512(src.cast()),
            _mm512_loadu_si512(src.add(16).cast()),
            _mm512_loadu_si512(src.add(32).cast()),
            _mm512_loadu_si512(src.add(48).cast()),
        )
    }
}

/// Load the first `pixels` (≤ 16) interleaved u32 pixels via masked loads and
/// return their channel planes. Lanes past `pixels` are zero.
///
/// # Safety
/// `src` must be valid for `pixels * 4` u32 reads.
#[target_feature(enable = "avx512f")]
#[inline]
pub(super) unsafe fn load_planar16_mask_epi32(
    src: *const u32,
    pixels: usize,
) -> (__m512i, __m512i, __m512i, __m512i) {
    let [m0, m1, m2, m3] = interleaved16_masks(pixels);
    // SAFETY: masked loads only touch enabled lanes, all within `pixels * 4`
    // per the caller.
    unsafe {
        deinterleave16_epi32(
            _mm512_maskz_loadu_epi32(m0, src.cast()),
            _mm512_maskz_loadu_epi32(m1, src.add(16).cast()),
            _mm512_maskz_loadu_epi32(m2, src.add(32).cast()),
            _mm512_maskz_loadu_epi32(m3, src.add(48).cast()),
        )
    }
}
