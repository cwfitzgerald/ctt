use crate::error::{Status, map_error, set_last_error};
use crate::formats::to_ctt_format;
use crate::image::{Image, take_image};
use crate::output::PipelineOutput;
use crate::types::{
    AlphaMode, ColorSpace, Format, MipmapFilter, OptionalAlphaMode, OptionalColorSpace,
    OptionalSize, OptionalSwizzle, Quality,
};

// ---------------------------------------------------------------------------
// bc7enc settings
// ---------------------------------------------------------------------------

/// Settings for the `bc7enc-rdo` encoder.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Bc7encSettings {
    /// Use perceptual error metrics (default `true`).
    pub perceptual: bool,
}

/// Default settings for the `bc7enc-rdo` encoder.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_bc7enc_settings_default() -> Bc7encSettings {
    Bc7encSettings { perceptual: true }
}

// ---------------------------------------------------------------------------
// Intel ISPC settings
// ---------------------------------------------------------------------------

/// Settings for the Intel ISPC texture compressor.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct IntelSettings {
    /// Encode the alpha channel for BC7. No effect on other formats.
    pub alpha: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_intel_settings_default() -> IntelSettings {
    IntelSettings { alpha: false }
}

// ---------------------------------------------------------------------------
// etcpak settings
// ---------------------------------------------------------------------------

/// Settings for the etcpak encoder.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EtcpakSettings {
    /// Enable dithering for ETC1 / BC1.
    pub dither: bool,
    /// Enable heuristic-based fast mode selection for ETC2 RGB / RGBA.
    pub use_heuristics: bool,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_etcpak_settings_default() -> EtcpakSettings {
    EtcpakSettings {
        dither: false,
        use_heuristics: false,
    }
}

// ---------------------------------------------------------------------------
// AMD Compressonator settings
// ---------------------------------------------------------------------------

/// Settings for the AMD Compressonator encoder. Currently no fields are
/// exposed; this struct exists for ABI uniformity with the other encoders.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AmdSettings {
    pub _reserved: u8,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_amd_settings_default() -> AmdSettings {
    AmdSettings { _reserved: 0 }
}

// ---------------------------------------------------------------------------
// astcenc settings
// ---------------------------------------------------------------------------

/// Settings for the `astcenc` encoder. Currently no fields are exposed.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct AstcencSettings {
    pub _reserved: u8,
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_astcenc_settings_default() -> AstcencSettings {
    AstcencSettings { _reserved: 0 }
}

// ---------------------------------------------------------------------------
// Encoder (tagged union)
// ---------------------------------------------------------------------------

/// User-facing encoder choice for compressed targets.
///
/// `Auto` picks the best compiled-in encoder for the requested format. Other
/// variants pin a specific backend and carry its settings. All variants are
/// always present in the ABI; selecting an encoder whose feature is disabled
/// at compile time returns `CTT_STATUS_ENCODER_NOT_COMPILED_IN`.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum Encoder {
    Auto,
    Bc7enc(Bc7encSettings),
    Intel(IntelSettings),
    Etcpak(EtcpakSettings),
    Amd(AmdSettings),
    Astcenc(AstcencSettings),
}

#[unsafe(no_mangle)]
pub extern "C" fn ctt_encoder_auto() -> Encoder {
    Encoder::Auto
}

impl Encoder {
    fn into_inner(self) -> Result<ctt::Encoder, Status> {
        match self {
            Encoder::Auto => Ok(ctt::Encoder::Auto),
            Encoder::Bc7enc(_settings) => {
                #[cfg(feature = "encoder-bc7enc")]
                {
                    Ok(ctt::Encoder::Bc7enc(
                        ctt::encoders::bc7enc::Bc7encSettings {
                            perceptual: _settings.perceptual,
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-bc7enc"))]
                {
                    set_last_error("encoder 'bc7enc' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Intel(_settings) => {
                #[cfg(feature = "encoder-intel")]
                {
                    Ok(ctt::Encoder::Intel(ctt::encoders::ispc::IspcSettings {
                        alpha: _settings.alpha,
                    }))
                }
                #[cfg(not(feature = "encoder-intel"))]
                {
                    set_last_error("encoder 'intel' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Etcpak(_settings) => {
                #[cfg(feature = "encoder-etcpak")]
                {
                    Ok(ctt::Encoder::Etcpak(
                        ctt::encoders::etcpak::EtcpakSettings {
                            dither: _settings.dither,
                            use_heuristics: _settings.use_heuristics,
                        },
                    ))
                }
                #[cfg(not(feature = "encoder-etcpak"))]
                {
                    set_last_error("encoder 'etcpak' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Amd(_) => {
                #[cfg(feature = "encoder-amd")]
                {
                    Ok(ctt::Encoder::Amd(
                        ctt::encoders::compressonator::AmdSettings,
                    ))
                }
                #[cfg(not(feature = "encoder-amd"))]
                {
                    set_last_error("encoder 'amd' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
            Encoder::Astcenc(_) => {
                #[cfg(feature = "encoder-astcenc")]
                {
                    Ok(ctt::Encoder::Astcenc(
                        ctt::encoders::astcenc::AstcencSettings,
                    ))
                }
                #[cfg(not(feature = "encoder-astcenc"))]
                {
                    set_last_error("encoder 'astcenc' is not compiled into this build");
                    Err(Status::EncoderNotCompiledIn)
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// TargetFormat (tagged union)
// ---------------------------------------------------------------------------

/// Inner payload of the [`TargetFormat::Compressed`] variant.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompressedTargetFormat {
    pub format: Format,
    pub encoder: Encoder,
}

/// The target format for a conversion.
///
/// `None` keeps the input format (no conversion). `Uncompressed` produces a
/// plain pixel format. `Compressed` block-encodes with the chosen
/// [`Encoder`].
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum TargetFormat {
    None,
    Uncompressed(Format),
    Compressed(CompressedTargetFormat),
}

impl TargetFormat {
    fn into_inner(self) -> Result<Option<ctt::TargetFormat>, Status> {
        match self {
            TargetFormat::None => Ok(None),
            TargetFormat::Uncompressed(f) => {
                let Some(fmt) = to_ctt_format(f) else {
                    set_last_error(
                        "TargetFormat::Uncompressed: format must be a non-zero VkFormat",
                    );
                    return Err(Status::InvalidArgument);
                };
                Ok(Some(ctt::TargetFormat::Uncompressed(fmt)))
            }
            TargetFormat::Compressed(body) => {
                let Some(fmt) = to_ctt_format(body.format) else {
                    set_last_error("TargetFormat::Compressed: format must be a non-zero VkFormat");
                    return Err(Status::InvalidArgument);
                };
                let encoder = body.encoder.into_inner()?;
                Ok(Some(ctt::TargetFormat::Compressed {
                    format: fmt,
                    encoder,
                }))
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Container (tagged union)
// ---------------------------------------------------------------------------

/// Output container format.
///
/// `Ktx2` writes a plain KTX2 file. `Ktx2Zstd` / `Ktx2Zlib` apply the
/// corresponding KTX2 supercompression scheme; the carried integer is the
/// compression level passed to the underlying codec. `Raw` returns the
/// processed image without serializing into a file format.
#[repr(C, u8)]
#[derive(Debug, Clone, Copy)]
pub enum Container {
    Dds,
    Ktx2,
    Ktx2Zstd(i32),
    Ktx2Zlib(u8),
    Raw,
}

impl From<Container> for ctt::Container {
    fn from(c: Container) -> Self {
        match c {
            Container::Dds => ctt::Container::Dds,
            Container::Ktx2 => ctt::Container::Ktx2(None),
            Container::Ktx2Zstd(level) => {
                ctt::Container::Ktx2(Some(ctt::Ktx2Supercompression::Zstd { level }))
            }
            Container::Ktx2Zlib(level) => {
                ctt::Container::Ktx2(Some(ctt::Ktx2Supercompression::Zlib { level }))
            }
            Container::Raw => ctt::Container::Raw,
        }
    }
}

// ---------------------------------------------------------------------------
// ConvertSettings
// ---------------------------------------------------------------------------

/// Settings controlling a [`ctt_convert`] call.
///
/// Build via [`ctt_convert_settings_default`] and overwrite fields, or
/// fill the struct directly. Optional fields have a `present` flag — set
/// it to `false` to keep the default behavior.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ConvertSettings {
    pub format: TargetFormat,
    pub container: Container,
    pub quality: Quality,
    pub output_color_space: OptionalColorSpace,
    pub output_alpha: OptionalAlphaMode,
    pub swizzle: OptionalSwizzle,
    pub mipmap: bool,
    pub mipmap_count: OptionalSize,
    pub mipmap_filter: MipmapFilter,
}

/// Default-constructed settings: input format preserved, KTX2 container,
/// `Basic` quality, no swizzle / mipmaps / overrides.
#[unsafe(no_mangle)]
pub extern "C" fn ctt_convert_settings_default() -> ConvertSettings {
    ConvertSettings {
        format: TargetFormat::None,
        container: Container::Ktx2,
        quality: Quality::Basic,
        output_color_space: OptionalColorSpace {
            present: false,
            value: ColorSpace::Linear,
        },
        output_alpha: OptionalAlphaMode {
            present: false,
            value: AlphaMode::Straight,
        },
        swizzle: OptionalSwizzle {
            present: false,
            value: crate::types::Swizzle {
                channels: [
                    crate::types::SwizzleChannel::R,
                    crate::types::SwizzleChannel::G,
                    crate::types::SwizzleChannel::B,
                    crate::types::SwizzleChannel::A,
                ],
            },
        },
        mipmap: false,
        mipmap_count: OptionalSize {
            present: false,
            value: 0,
        },
        mipmap_filter: MipmapFilter::Triangle,
    }
}

// ---------------------------------------------------------------------------
// convert
// ---------------------------------------------------------------------------

/// Run the conversion pipeline.
///
/// **Consumes** `image` on both success and failure — the handle must not be
/// destroyed by the caller after this call. On success, writes a freshly
/// allocated `ctt_pipeline_output_t` handle into `*out` (caller frees with
/// `ctt_pipeline_output_destroy`).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn ctt_convert(
    image: *mut Image,
    settings: *const ConvertSettings,
    out: *mut *mut PipelineOutput,
) -> Status {
    if out.is_null() {
        if !image.is_null() {
            drop(unsafe { Box::from_raw(image) });
        }
        set_last_error("ctt_convert: out is null");
        return Status::NullPointer;
    }
    let Some(settings) = (unsafe { settings.as_ref() }) else {
        if !image.is_null() {
            drop(unsafe { Box::from_raw(image) });
        }
        set_last_error("ctt_convert: settings is null");
        return Status::NullPointer;
    };

    let image = match unsafe { take_image(image) } {
        Ok(i) => i,
        Err(s) => return s,
    };

    let format = match settings.format.into_inner() {
        Ok(f) => f,
        Err(s) => return s,
    };

    let swizzle = if settings.swizzle.present {
        Some(settings.swizzle.value.into())
    } else {
        None
    };

    let inner = ctt::ConvertSettings {
        format,
        container: settings.container.into(),
        quality: settings.quality.into_inner(),
        output_color_space: settings
            .output_color_space
            .present
            .then(|| settings.output_color_space.value.into()),
        output_alpha: settings
            .output_alpha
            .present
            .then(|| settings.output_alpha.value.into()),
        swizzle,
        mipmap: settings.mipmap,
        mipmap_count: settings
            .mipmap_count
            .present
            .then_some(settings.mipmap_count.value),
        mipmap_filter: settings.mipmap_filter.into(),
    };

    match ctt::convert(image, inner) {
        Ok(output) => {
            let boxed = Box::into_raw(Box::new(PipelineOutput(Some(output))));
            unsafe {
                *out = boxed;
            }
            Status::Ok
        }
        Err(e) => map_error(e),
    }
}
