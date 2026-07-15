pub(crate) use crate::quality::Quality;

pub(crate) mod backend;
#[cfg(any(
    feature = "encoder-intel",
    feature = "encoder-bc7enc",
    feature = "encoder-etcpak",
    feature = "encoder-amd",
))]
pub(crate) mod parallel;

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

/// Assert that `encode` produces identical bytes on one- and two-worker
/// Rayon pools.
#[cfg(all(
    test,
    feature = "rayon",
    any(
        feature = "encoder-intel",
        feature = "encoder-bc7enc",
        feature = "encoder-etcpak",
        feature = "encoder-amd",
        feature = "encoder-astcenc",
    )
))]
pub(crate) fn assert_parallel_matches_serial(encode: impl Fn() -> Vec<u8> + Sync) {
    let pool = |threads| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
    };
    let serial = pool(1).install(&encode);
    let parallel = pool(2).install(&encode);
    assert_eq!(parallel, serial);
}

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

/// Resolve [`Encoder::Auto`] to the concrete backend that will actually run
/// for `format`, honoring compiled-in features and the same priority order as
/// the internal auto-pick used during encoding.
///
/// Returns [`None`] when no compiled-in encoder supports `format`; callers can
/// leave [`Encoder::Auto`] in place and let the encode step surface the real
/// "no encoder supports …" error. The encoding path delegates to this resolver;
/// it is public so the CLI can apply the matching
/// `--<encoder>-opts` to the backend Auto will select.
pub fn resolve_auto_encoder(format: ktx2::Format) -> Option<Encoder> {
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

    #[cfg(feature = "encoder-bc7enc")]
    if bc7enc::Bc7encEncoder::supported_formats().contains(&format) {
        return Some(Encoder::Bc7enc(Default::default()));
    }
    #[cfg(feature = "encoder-intel")]
    if ispc::IspcEncoder::supported_formats().contains(&format) {
        return Some(Encoder::Intel(Default::default()));
    }
    #[cfg(feature = "encoder-etcpak")]
    if etcpak::EtcpakEncoder::supported_formats().contains(&format) {
        return Some(Encoder::Etcpak(Default::default()));
    }
    #[cfg(feature = "encoder-amd")]
    if compressonator::CompressonatorEncoder::supported_formats().contains(&format) {
        return Some(Encoder::Amd(Default::default()));
    }
    #[cfg(feature = "encoder-astcenc")]
    if astcenc::AstcencEncoder::supported_formats().contains(&format) {
        return Some(Encoder::Astcenc(Default::default()));
    }

    let _ = format;
    None
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

#[cfg(test)]
mod tests {
    use super::*;

    fn encoder_name(encoder: &Encoder) -> &'static str {
        match encoder {
            Encoder::Auto => "auto",
            #[cfg(feature = "encoder-bc7enc")]
            Encoder::Bc7enc(_) => "bc7e",
            #[cfg(feature = "encoder-intel")]
            Encoder::Intel(_) => "intel",
            #[cfg(feature = "encoder-etcpak")]
            Encoder::Etcpak(_) => "etcpak",
            #[cfg(feature = "encoder-amd")]
            Encoder::Amd(_) => "amd",
            #[cfg(feature = "encoder-astcenc")]
            Encoder::Astcenc(_) => "astcenc",
        }
    }

    #[test]
    fn auto_resolution_matches_compiled_encoder_priority() {
        let encoders = compiled_in_encoders();
        for info in &encoders {
            for &format in info.supported_formats {
                let expected = encoders
                    .iter()
                    .find(|candidate| candidate.supported_formats.contains(&format))
                    .unwrap()
                    .name;
                let resolved = resolve_auto_encoder(format).unwrap();
                assert_eq!(encoder_name(&resolved), expected, "format {format:?}");
            }
        }
    }
}
