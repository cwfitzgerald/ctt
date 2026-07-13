//! Regression test for the BC7 bit-packing / anchor-index bug.
//!
//! ISPC silently drops `--` (decrement) on unsigned integers, so
//! `encode_bc7_block`'s anchor index-bit reduction (`uint32_t n = ...; n--;`)
//! never fired. Every block was therefore packed one bit too long (129 bits):
//!
//!   * `set_block_bits` wrote one byte past the 16-byte block (memory
//!     corruption), and
//!   * the anchor index was stored full-width, shifting the remaining indices
//!     by one bit, so the output was not spec-compliant.
//!
//! The fix rewrites those decrements as `n -= 1`. This test encodes an opaque
//! gradient with the encoder pinned to BC7 mode 6 (`m_mode6_only`) and decodes
//! it with a spec-compliant mode-6 decoder; a mis-sized anchor index shows up
//! as a large reconstruction error. With the bug present the MSE is ~14; once
//! fixed it is well under 1.

use ctt_bc7enc_rdo as bc7e;

/// 4-bit BC7 interpolation weights.
const W4: [i32; 16] = [0, 4, 9, 13, 17, 21, 26, 30, 34, 38, 43, 47, 51, 55, 60, 64];

struct BitReader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl BitReader<'_> {
    fn read(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for i in 0..n {
            let bit = (self.bytes[self.pos >> 3] >> (self.pos & 7)) & 1;
            v |= u32::from(bit) << i;
            self.pos += 1;
        }
        v
    }
}

/// Decode a single BC7 **mode 6** block (16 bytes) into 16 RGBA texels.
/// Panics if the block is not mode 6 (the encoder is pinned to mode 6 via
/// `m_mode6_only`, so every block must be mode 6).
fn decode_mode6(block: &[u8; 16]) -> [[u8; 4]; 16] {
    let mut r = BitReader {
        bytes: block,
        pos: 0,
    };

    // Mode 6 marker: six 0 bits then a 1 (0b100_0000 == 0x40).
    assert_eq!(r.read(7), 0x40, "expected BC7 mode 6");

    // Endpoint order: R0 R1 G0 G1 B0 B1 A0 A1, 7 bits each.
    let raw: [[u32; 2]; 4] = {
        let mut e = [[0u32; 2]; 4];
        for channel in &mut e {
            channel[0] = r.read(7);
            channel[1] = r.read(7);
        }
        e
    };
    let p0 = r.read(1);
    let p1 = r.read(1);

    // Each endpoint gets a trailing p-bit, giving 8-bit channel values.
    let e0 = [0, 1, 2, 3].map(|c| ((raw[c][0] << 1) | p0) as i32);
    let e1 = [0, 1, 2, 3].map(|c| ((raw[c][1] << 1) | p1) as i32);

    // 16 indices, 4 bits each, except the anchor (texel 0) which is 3 bits
    // (its high bit is implicit 0).
    let mut out = [[0u8; 4]; 16];
    for (t, texel) in out.iter_mut().enumerate() {
        let bits = if t == 0 { 3 } else { 4 };
        let w = W4[(r.read(bits) & 15) as usize];
        for c in 0..4 {
            texel[c] = ((e0[c] * (64 - w) + e1[c] * w + 32) >> 6) as u8;
        }
    }
    out
}

/// Build one 4x4 opaque gradient block, and return (pixels, reference RGBA).
fn gradient_block(seed: u32) -> ([u32; 16], [[u8; 4]; 16]) {
    let mut pixels = [0u32; 16];
    let mut refr = [[0u8; 4]; 16];
    for p in 0..16u32 {
        let x = ((seed * 16 + p) & 0xff) as u8;
        let (r, g, b, a) = (x, x, 255 - x, 255);
        pixels[p as usize] =
            u32::from(r) | (u32::from(g) << 8) | (u32::from(b) << 16) | (u32::from(a) << 24);
        refr[p as usize] = [r, g, b, a];
    }
    (pixels, refr)
}

#[test]
fn mode6_anchor_index_is_spec_compliant() {
    const NUM_BLOCKS: usize = 4;

    let mut pixels = Vec::with_capacity(NUM_BLOCKS * 16);
    let mut refs = Vec::with_capacity(NUM_BLOCKS);
    for b in 0..NUM_BLOCKS as u32 {
        let (px, rf) = gradient_block(b);
        pixels.extend_from_slice(&px);
        refs.push(rf);
    }

    // Force every block to BC7 mode 6 so the mode-6 decoder below is always
    // valid; without this the encoder is free to pick another mode for opaque
    // input and the anchor-index check would never run.
    let mut params = bc7e::params_init_basic(false);
    params.m_mode6_only = true;
    let blocks = bc7e::compress_blocks_alloc(NUM_BLOCKS, &pixels, &params);
    assert_eq!(blocks.len(), NUM_BLOCKS * 2);

    let mut total_sq_err = 0.0f64;
    for (b, refr) in refs.iter().enumerate() {
        let mut bytes = [0u8; 16];
        bytes[..8].copy_from_slice(&blocks[b * 2].to_le_bytes());
        bytes[8..].copy_from_slice(&blocks[b * 2 + 1].to_le_bytes());

        let decoded = decode_mode6(&bytes);
        for (dec, exp) in decoded.iter().zip(refr) {
            for c in 0..4 {
                let d = f64::from(dec[c]) - f64::from(exp[c]);
                total_sq_err += d * d;
            }
        }
    }
    let mse = total_sq_err / (NUM_BLOCKS * 16 * 4) as f64;

    // A spec-compliant encode reconstructs the gradient almost exactly
    // (MSE < 1). The anchor-index bug shifts the indices and pushes MSE to ~14.
    assert!(
        mse < 4.0,
        "BC7 mode-6 output is not spec-compliant: reconstruction MSE = {mse:.2} \
         (expected < 4.0). The anchor index bit-width reduction is likely not \
         being applied (ISPC `--` on unsigned is a no-op; use `-= 1`)."
    );
}
