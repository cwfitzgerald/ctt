//! Internal trait that pins the per-encoder API shape.

use crate::error::Result;
use crate::quality::Quality;
use crate::surface::Surface;

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
pub(crate) trait Encoder {
    type Settings: Default;

    fn name() -> &'static str;
    fn supported_formats() -> &'static [ktx2::Format];
    /// Pixel format the encoder expects to receive in `compress`.
    ///
    /// Most encoders only depend on the target format, but some (notably
    /// astcenc, whose input type varies between LDR and HDR) also need the
    /// caller's settings to decide. Implementations that don't care can
    /// ignore `settings`.
    fn required_input_format(format: ktx2::Format, settings: &Self::Settings) -> ktx2::Format;
    fn compress(
        surface: &Surface,
        format: ktx2::Format,
        quality: Quality,
        settings: &Self::Settings,
    ) -> Result<Vec<u8>>;
}
