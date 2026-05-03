pub(crate) use crate::quality::Quality;

pub(crate) mod backend;

#[cfg(any(feature = "encoder-intel", feature = "encoder-etcpak"))]
mod edge;

#[cfg(feature = "encoder-intel")]
pub mod ispc;

#[cfg(feature = "encoder-bc7enc")]
pub mod bc7enc;

#[cfg(feature = "encoder-etcpak")]
pub mod etcpak;

#[cfg(feature = "encoder-astcenc")]
pub mod astcenc;

#[cfg(feature = "encoder-amd")]
pub mod compressonator;

/// User-facing encoder choice for [`TargetFormat::Compressed`](crate::TargetFormat).
///
/// `Auto` picks the highest-priority compiled-in encoder that supports the
/// target format. The remaining variants pin a specific backend and carry
/// its settings. Each variant is gated by its `encoder-*` feature; backends
/// not enabled at compile time aren't part of the enum at all.
#[derive(Debug, Clone, Default)]
pub enum Encoder {
    #[default]
    Auto,
    #[cfg(feature = "encoder-bc7enc")]
    Bc7enc(bc7enc::Bc7encSettings),
    #[cfg(feature = "encoder-intel")]
    Intel(ispc::IspcSettings),
    #[cfg(feature = "encoder-etcpak")]
    Etcpak(etcpak::EtcpakSettings),
    #[cfg(feature = "encoder-amd")]
    Amd(compressonator::AmdSettings),
    #[cfg(feature = "encoder-astcenc")]
    Astcenc(astcenc::AstcencSettings),
}

/// Static description of one compiled-in encoder backend, used by the CLI's
/// `--list-encoders` table and by `format::parse_format` for prefix matching.
pub struct EncoderInfo {
    pub name: &'static str,
    pub supported_formats: &'static [ktx2::Format],
}

/// All compiled-in encoders, in priority order. The first entry that supports
/// a given format wins for [`Encoder::Auto`].
pub fn compiled_in_encoders() -> Vec<EncoderInfo> {
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
    use backend::Encoder as _;
    vec![
        #[cfg(feature = "encoder-bc7enc")]
        EncoderInfo {
            name: bc7enc::Bc7encEncoder::name(),
            supported_formats: bc7enc::Bc7encEncoder::supported_formats(),
        },
        #[cfg(feature = "encoder-intel")]
        EncoderInfo {
            name: ispc::IspcEncoder::name(),
            supported_formats: ispc::IspcEncoder::supported_formats(),
        },
        #[cfg(feature = "encoder-etcpak")]
        EncoderInfo {
            name: etcpak::EtcpakEncoder::name(),
            supported_formats: etcpak::EtcpakEncoder::supported_formats(),
        },
        #[cfg(feature = "encoder-amd")]
        EncoderInfo {
            name: compressonator::CompressonatorEncoder::name(),
            supported_formats: compressonator::CompressonatorEncoder::supported_formats(),
        },
        #[cfg(feature = "encoder-astcenc")]
        EncoderInfo {
            name: astcenc::AstcencEncoder::name(),
            supported_formats: astcenc::AstcencEncoder::supported_formats(),
        },
    ]
}
