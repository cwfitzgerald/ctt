//! Block-compression step — runs an encoder backend across every surface in
//! an [`Image`]. Input must already be the format the encoder requires
//! (typically `R8G8B8A8_UNORM` / `R32G32B32A32_SFLOAT`).

use crate::encoders::Encoder;
#[cfg_attr(
    not(any(
        feature = "encoder-bc7enc",
        feature = "encoder-intel",
        feature = "encoder-etcpak",
        feature = "encoder-amd",
        feature = "encoder-astcenc",
    )),
    expect(unused_imports)
)]
use crate::encoders::backend::Encoder as _;
use crate::error::{Error, Result};
use crate::quality::Quality;
use crate::surface::{Image, Surface};
use crate::vk_format::FormatExt;

/// A resolved encoder step: what format to encode to, the chosen encoder
/// (and its settings), and the universal quality preset.
pub struct EncoderStep {
    pub target_format: ktx2::Format,
    pub quality: Quality,
    pub encoder: Encoder,
}

impl EncoderStep {
    /// The uncompressed format the encoder requires as input.
    pub fn required_input(&self) -> Result<ktx2::Format> {
        let raw = required_input_for(&self.encoder, self.target_format)?;
        let (required, _) = raw.normalize();
        Ok(required)
    }
}

/// Run the encoder across every surface of the image.
///
/// With the `rayon` feature, all surfaces encode concurrently as one flat job
/// list, so small mips from different layers fill workers left idle by the
/// row-chunk jobs of larger surfaces.
pub fn encode_all(image: Image, step: &EncoderStep) -> Result<Image> {
    profiling::scope!("encode_all");

    let indexed: Vec<Vec<(usize, usize, Surface)>> = image
        .surfaces
        .into_iter()
        .enumerate()
        .map(|(layer_idx, layer)| {
            layer
                .into_iter()
                .enumerate()
                .map(|(mip_idx, surface)| (layer_idx, mip_idx, surface))
                .collect()
        })
        .collect();

    let new_surfaces = super::map_nested(indexed, |(layer_idx, mip_idx, surface)| {
        profiling::scope!("encode_mip", encoder_name(&step.encoder));
        log::debug!(
            "Compressing layer {layer_idx}, mip {mip_idx}: {}x{} to {:?} using {}",
            surface.width,
            surface.height,
            step.target_format,
            encoder_name(&step.encoder),
        );

        let output_format = step.target_format.denormalize(surface.color_space);
        let data = compress_with(&step.encoder, &surface, output_format, step.quality)?;

        let bpp_block = step.target_format.bytes_per_block().unwrap_or(16) as u32;
        let (bw, _bh) = step.target_format.block_size().unwrap_or((4, 4));
        let blocks_x = surface.width.div_ceil(bw as u32);

        Ok(Surface {
            data,
            width: surface.width,
            height: surface.height,
            depth: surface.depth,
            stride: blocks_x * bpp_block,
            slice_stride: surface.slice_stride,
            format: step.target_format,
            color_space: surface.color_space,
            alpha: surface.alpha,
        })
    })?;

    Ok(Image {
        surfaces: new_surfaces,
        kind: image.kind,
    })
}

/// Auto-pick: first compiled-in backend that supports `target` wins.
///
/// Delegates to [`crate::encoders::resolve_auto_encoder`] so that the
/// public resolution API and the encode step can never disagree about
/// which backend `Encoder::Auto` selects.
fn pick_auto(target: ktx2::Format) -> Result<Encoder> {
    crate::encoders::resolve_auto_encoder(target).ok_or_else(|| {
        Error::UnsupportedFormat(format!("no compiled-in encoder supports {target:?}"))
    })
}

#[cfg_attr(
    not(any(
        feature = "encoder-bc7enc",
        feature = "encoder-intel",
        feature = "encoder-etcpak",
        feature = "encoder-amd",
        feature = "encoder-astcenc",
    )),
    expect(dead_code)
)]
fn require_supports(supported: &[ktx2::Format], target: ktx2::Format, name: &str) -> Result<()> {
    if supported.contains(&target) {
        Ok(())
    } else {
        Err(Error::UnsupportedFormat(format!(
            "encoder '{name}' does not support {target:?}"
        )))
    }
}

fn required_input_for(encoder: &Encoder, target: ktx2::Format) -> Result<ktx2::Format> {
    #[cfg(feature = "encoder-astcenc")]
    use crate::encoders::astcenc::AstcencEncoder;
    #[cfg(feature = "encoder-bc7enc")]
    use crate::encoders::bc7enc::Bc7encEncoder;
    #[cfg(feature = "encoder-amd")]
    use crate::encoders::compressonator::CompressonatorEncoder;
    #[cfg(feature = "encoder-etcpak")]
    use crate::encoders::etcpak::EtcpakEncoder;
    #[cfg(feature = "encoder-intel")]
    use crate::encoders::ispc::IspcEncoder;

    match encoder {
        Encoder::Auto => {
            let resolved = pick_auto(target)?;
            required_input_for(&resolved, target)
        }
        #[cfg(feature = "encoder-bc7enc")]
        Encoder::Bc7enc(s) => {
            require_supports(Bc7encEncoder::supported_formats(), target, "bc7e")?;
            Ok(Bc7encEncoder::required_input_format(target, s))
        }
        #[cfg(feature = "encoder-intel")]
        Encoder::Intel(s) => {
            require_supports(IspcEncoder::supported_formats(), target, "intel")?;
            Ok(IspcEncoder::required_input_format(target, s))
        }
        #[cfg(feature = "encoder-etcpak")]
        Encoder::Etcpak(s) => {
            require_supports(EtcpakEncoder::supported_formats(), target, "etcpak")?;
            Ok(EtcpakEncoder::required_input_format(target, s))
        }
        #[cfg(feature = "encoder-amd")]
        Encoder::Amd(s) => {
            require_supports(CompressonatorEncoder::supported_formats(), target, "amd")?;
            Ok(CompressonatorEncoder::required_input_format(target, s))
        }
        #[cfg(feature = "encoder-astcenc")]
        Encoder::Astcenc(s) => {
            require_supports(AstcencEncoder::supported_formats(), target, "astcenc")?;
            Ok(AstcencEncoder::required_input_format(target, s))
        }
    }
}

fn compress_with(
    encoder: &Encoder,
    surface: &Surface,
    output_format: ktx2::Format,
    quality: Quality,
) -> Result<Vec<u8>> {
    let (base, _) = output_format.normalize();
    match encoder {
        Encoder::Auto => {
            let resolved = pick_auto(base)?;
            compress_with(&resolved, surface, output_format, quality)
        }
        #[cfg(feature = "encoder-bc7enc")]
        Encoder::Bc7enc(s) => {
            crate::encoders::bc7enc::Bc7encEncoder::compress(surface, output_format, quality, s)
        }
        #[cfg(feature = "encoder-intel")]
        Encoder::Intel(s) => {
            crate::encoders::ispc::IspcEncoder::compress(surface, output_format, quality, s)
        }
        #[cfg(feature = "encoder-etcpak")]
        Encoder::Etcpak(s) => {
            crate::encoders::etcpak::EtcpakEncoder::compress(surface, output_format, quality, s)
        }
        #[cfg(feature = "encoder-amd")]
        Encoder::Amd(s) => crate::encoders::compressonator::CompressonatorEncoder::compress(
            surface,
            output_format,
            quality,
            s,
        ),
        #[cfg(feature = "encoder-astcenc")]
        Encoder::Astcenc(s) => {
            crate::encoders::astcenc::AstcencEncoder::compress(surface, output_format, quality, s)
        }
    }
}

fn encoder_name(encoder: &Encoder) -> &'static str {
    match encoder {
        Encoder::Auto => "auto",
        #[cfg(feature = "encoder-bc7enc")]
        Encoder::Bc7enc(_) => crate::encoders::bc7enc::Bc7encEncoder::name(),
        #[cfg(feature = "encoder-intel")]
        Encoder::Intel(_) => crate::encoders::ispc::IspcEncoder::name(),
        #[cfg(feature = "encoder-etcpak")]
        Encoder::Etcpak(_) => crate::encoders::etcpak::EtcpakEncoder::name(),
        #[cfg(feature = "encoder-amd")]
        Encoder::Amd(_) => crate::encoders::compressonator::CompressonatorEncoder::name(),
        #[cfg(feature = "encoder-astcenc")]
        Encoder::Astcenc(_) => crate::encoders::astcenc::AstcencEncoder::name(),
    }
}
