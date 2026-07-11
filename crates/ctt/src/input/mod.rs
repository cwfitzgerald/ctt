pub mod dds;
pub mod ktx2;

use crate::alpha::AlphaMode;
use crate::error::Result;
use crate::surface::{ColorSpace, Image};

/// The detected container type of input data.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputFormat {
    /// KTX2 container.
    Ktx2,
    /// DDS container.
    Dds,
}

/// Detect whether the data is a known container format (KTX2 or DDS).
///
/// Returns `None` for unrecognized data (e.g. PNG, JPEG, etc.).
pub fn detect_container(data: &[u8]) -> Option<InputFormat> {
    // KTX2: 12-byte magic «KTX 20»\r\n\x1A\n
    if data.starts_with(&::ktx2::MAGIC) {
        return Some(InputFormat::Ktx2);
    }

    // DDS: 4-byte magic "DDS " (0x20534444 little-endian)
    if data.len() >= 4 && data[..4] == [0x44, 0x44, 0x53, 0x20] {
        return Some(InputFormat::Dds);
    }

    None
}

/// Optional overrides for metadata that the container provides.
///
/// When set, these take precedence over the values read from the container.
#[derive(Debug, Clone, Copy, Default)]
pub struct InputOverrides {
    /// Overrides the color space of every surface when set.
    pub color_space: Option<ColorSpace>,
    /// Overrides the alpha mode of every surface when set.
    pub alpha: Option<AlphaMode>,
}

/// Decode a KTX2 or DDS container from raw bytes, auto-detecting which.
///
/// Returns `None` if the data doesn't match a known container magic.
/// Use the returned `Image` directly or feed it into the pipeline.
///
/// Fields in `overrides` take precedence over container metadata when set.
pub fn decode_container(data: &[u8], overrides: InputOverrides) -> Result<Option<Image>> {
    match detect_container(data) {
        Some(format) => decode_container_as(data, format, overrides).map(Some),
        None => Ok(None),
    }
}

/// Decode a container from raw bytes using an explicitly specified format.
///
/// Use this when the user has explicitly chosen the input format, bypassing
/// magic number detection.
pub fn decode_container_as(
    data: &[u8],
    format: InputFormat,
    overrides: InputOverrides,
) -> Result<Image> {
    let mut image = match format {
        InputFormat::Ktx2 => ktx2::decode_ktx2_image(data)?,
        InputFormat::Dds => dds::decode_dds_image(data)?,
    };

    // Apply overrides to all surfaces.
    if overrides.color_space.is_some() || overrides.alpha.is_some() {
        for layer in &mut image.surfaces {
            for surface in layer {
                if let Some(cs) = overrides.color_space {
                    surface.color_space = cs;
                }
                if let Some(alpha) = overrides.alpha {
                    surface.alpha = alpha;
                }
            }
        }
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_ktx2_magic() {
        let mut data = vec![0u8; 80];
        data[..12].copy_from_slice(&::ktx2::MAGIC);
        assert_eq!(detect_container(&data), Some(InputFormat::Ktx2));
    }

    #[test]
    fn detect_dds_magic() {
        let data = [0x44, 0x44, 0x53, 0x20, 0, 0, 0, 0];
        assert_eq!(detect_container(&data), Some(InputFormat::Dds));
    }

    #[test]
    fn detect_unknown_returns_none() {
        let data = [0x89, 0x50, 0x4E, 0x47]; // PNG magic
        assert_eq!(detect_container(&data), None);
    }

    #[test]
    fn detect_empty_returns_none() {
        assert_eq!(detect_container(&[]), None);
    }
}
