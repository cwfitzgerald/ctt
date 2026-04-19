//! Helpers for feeding block-encoders images whose dimensions aren't
//! multiples of the block size.
//!
//! The backends (ISPC, etcpak, compressonator) all require block-aligned
//! input. We preserve the zero-copy fast path for aligned dimensions and
//! only synthesize a small scratch buffer for the partial-edge blocks, so
//! the interior of the image is never copied.

/// Fill a `4 × 4 × bpp` tightly-packed block scratch from the surface, clamping
/// sample coordinates into the source extent. `block_x, block_y` are in blocks
/// (not pixels): the destination covers source pixels
/// `[block_x*4 .. block_x*4+4) × [block_y*4 .. block_y*4+4)`.
///
/// Any sample outside `0..width × 0..height` is replaced by the nearest edge
/// pixel (clamp-to-edge).
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_clamped_block(
    src: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    bpp: u32,
    block_x: u32,
    block_y: u32,
    dst: &mut [u8],
) {
    debug_assert_eq!(dst.len(), (4 * 4 * bpp) as usize);
    let max_x = width - 1;
    let max_y = height - 1;
    let bpp_usize = bpp as usize;
    for py in 0..4u32 {
        let y = (block_y * 4 + py).min(max_y);
        for px in 0..4u32 {
            let x = (block_x * 4 + px).min(max_x);
            let src_off = (y * stride + x * bpp) as usize;
            let dst_off = ((py * 4 + px) * bpp) as usize;
            dst[dst_off..dst_off + bpp_usize].copy_from_slice(&src[src_off..src_off + bpp_usize]);
        }
    }
}

/// Fill a `row_w × 4 × bpp` tightly-packed block-row scratch from the surface.
/// Rows are taken from source y-coordinates `[block_y*4 .. block_y*4+4)`
/// (clamped), columns from `0..row_w` (clamped to `width-1`).
///
/// `row_w` must be a multiple of 4 and >= the source width so every source
/// column is represented.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fill_clamped_block_row(
    src: &[u8],
    width: u32,
    height: u32,
    stride: u32,
    bpp: u32,
    block_y: u32,
    row_w: u32,
    dst: &mut [u8],
) {
    debug_assert!(row_w.is_multiple_of(4));
    debug_assert!(row_w >= width);
    debug_assert_eq!(dst.len(), (row_w * 4 * bpp) as usize);
    let max_x = width - 1;
    let max_y = height - 1;
    let bpp_usize = bpp as usize;
    let dst_stride = (row_w * bpp) as usize;
    for py in 0..4u32 {
        let y = (block_y * 4 + py).min(max_y);
        let src_row_off = (y * stride) as usize;
        let dst_row_off = (py as usize) * dst_stride;
        // Copy the in-range portion verbatim (zero work per pixel).
        let copy_bytes = (width * bpp) as usize;
        dst[dst_row_off..dst_row_off + copy_bytes]
            .copy_from_slice(&src[src_row_off..src_row_off + copy_bytes]);
        // Right-edge replication for x in [width, row_w).
        let edge_src_off = src_row_off + (max_x * bpp) as usize;
        let edge_pixel = &src[edge_src_off..edge_src_off + bpp_usize];
        for px in width..row_w {
            let dst_off = dst_row_off + (px * bpp) as usize;
            dst[dst_off..dst_off + bpp_usize].copy_from_slice(edge_pixel);
        }
        // y-clamping already handled for top/bottom via `y = min(..)`.
        let _ = max_y;
    }
}
