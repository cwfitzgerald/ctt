//! Safe BC7F compression for blocks of 16 RGBA8 pixels in row order.

pub mod bindings;

use bindings::*;
use std::sync::Once;

static INIT: Once = Once::new();

/// Compresses blocks of 64 RGBA8 bytes into blocks of 16 BC7 bytes.
/// Use the `cPackBC7Flag*` constants in [`bindings`] to select encoder flags.
///
/// # Panics
/// Panics if the input contains an incomplete block or the flags are invalid.
#[must_use]
pub fn compress_blocks(pixels: &[u8], flags: u32) -> Vec<u8> {
    assert!(
        pixels.len().is_multiple_of(64),
        "input must contain complete RGBA8 blocks"
    );
    let mut output = vec![0; pixels.len() / 64 * 16];
    compress_blocks_into(pixels, &mut output, flags);
    output
}

/// Compresses blocks of 64 RGBA8 bytes into the supplied BC7 buffer.
///
/// # Panics
/// Panics if input blocks are incomplete, output size is not exact, or flags are invalid.
pub fn compress_blocks_into(pixels: &[u8], output: &mut [u8], flags: u32) {
    assert!(
        pixels.len().is_multiple_of(64),
        "input must contain complete RGBA8 blocks"
    );
    assert_eq!(
        output.len(),
        pixels.len() / 64 * 16,
        "output must contain 16 bytes per block"
    );
    assert_eq!(flags & !0x3fff, 0, "unknown BC7F flags");
    for (flag, required) in [
        (cPackBC7FlagUse3SubsetsRGB, cPackBC7FlagUse2SubsetsRGB),
        (
            cPackBC7FlagNonAnalyticalRGB,
            cPackBC7FlagPartiallyAnalyticalRGB,
        ),
        (
            cPackBC7FlagNonAnalyticalRGBA,
            cPackBC7FlagPartiallyAnalyticalRGBA,
        ),
    ] {
        assert!(
            flags & flag == 0 || flags & required != 0,
            "missing required BC7F flag"
        );
    }
    // SAFETY: Once serializes table initialization before any compression call.
    INIT.call_once(|| unsafe { ctt_bc7f_init() });
    // SAFETY: The slices contain the exact block count. C++ copies each input
    // block into aligned pixel storage and writes 16 bytes per output block.
    unsafe {
        ctt_bc7f_compress_blocks(
            output.as_mut_ptr(),
            pixels.as_ptr(),
            pixels.len() / 64,
            flags,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_input() {
        assert!(compress_blocks(&[], cPackBC7FlagDefault).is_empty());
    }

    #[test]
    #[should_panic(expected = "complete RGBA8 blocks")]
    fn incomplete_input() {
        let _ = compress_blocks(&[0; 63], cPackBC7FlagDefault);
    }

    #[test]
    #[should_panic(expected = "16 bytes per block")]
    fn wrong_output_size() {
        compress_blocks_into(&[0; 64], &mut [0; 15], cPackBC7FlagDefault);
    }

    #[test]
    #[should_panic(expected = "missing required")]
    fn invalid_flag_combination() {
        let _ = compress_blocks(&[0; 64], cPackBC7FlagNonAnalyticalRGBA);
    }

    #[test]
    fn unaligned_slices_preserve_guards() {
        let pixels: Vec<_> = (0..129).map(|i| (i * 37) as u8).collect();
        let mut output = [0xAB; 34];
        compress_blocks_into(&pixels[1..], &mut output[1..33], cPackBC7FlagDefault);
        assert_eq!(output[0], 0xAB);
        assert_eq!(output[33], 0xAB);
        assert_eq!(
            &output[1..33],
            compress_blocks(&pixels[1..], cPackBC7FlagDefault)
        );
    }

    #[test]
    fn batch_matches_individual_blocks() {
        let pixels: Vec<_> = (0..192).map(|i| (i * 37) as u8).collect();
        for flags in [
            cPackBC7FlagDefaultFastest,
            cPackBC7FlagDefaultFaster,
            cPackBC7FlagDefaultFast,
            cPackBC7FlagDefault,
            cPackBC7FlagDefaultPartiallyAnalytical,
            cPackBC7FlagDefaultNonAnalytical,
        ] {
            let batch = compress_blocks(&pixels, flags);
            for (input, output) in pixels.chunks_exact(64).zip(batch.chunks_exact(16)) {
                assert_eq!(compress_blocks(input, flags), output);
                assert_ne!(output[0], 0, "BC7 mode must be valid");
            }
        }
    }
}
