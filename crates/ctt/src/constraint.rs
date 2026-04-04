use crate::alpha::AlphaMode;
use crate::format::ColorSpace;

/// Describes the format requirements of a transform node.
///
/// Each field is `None` to accept any value, or `Some(list)` to restrict to specific values.
/// All specified constraints must be satisfied simultaneously.
#[derive(Debug, Clone)]
pub struct FormatConstraint {
    /// Accepted data layout formats. `None` = any format.
    pub formats: Option<Vec<ktx2::Format>>,
    /// Accepted color spaces. `None` = any color space.
    pub color_spaces: Option<Vec<ColorSpace>>,
    /// Accepted alpha modes. `None` = any alpha mode.
    pub alpha_modes: Option<Vec<AlphaMode>>,
}

impl FormatConstraint {
    /// A constraint that accepts anything.
    pub fn any() -> Self {
        Self {
            formats: None,
            color_spaces: None,
            alpha_modes: None,
        }
    }

    /// Check whether a given format state satisfies this constraint.
    pub fn accepts(&self, format: ktx2::Format, cs: ColorSpace, alpha: AlphaMode) -> bool {
        let format_ok = self
            .formats
            .as_ref()
            .is_none_or(|list| list.contains(&format));
        let cs_ok = self
            .color_spaces
            .as_ref()
            .is_none_or(|list| list.contains(&cs));
        let alpha_ok = self
            .alpha_modes
            .as_ref()
            .is_none_or(|list| list.contains(&alpha));
        format_ok && cs_ok && alpha_ok
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn any_accepts_everything() {
        let c = FormatConstraint::any();
        assert!(c.accepts(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Premultiplied,
        ));
    }

    #[test]
    fn format_restriction() {
        let c = FormatConstraint {
            formats: Some(vec![ktx2::Format::R8G8B8A8_UNORM]),
            color_spaces: None,
            alpha_modes: None,
        };
        assert!(c.accepts(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        ));
        assert!(!c.accepts(
            ktx2::Format::R16G16B16A16_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Straight,
        ));
    }

    #[test]
    fn combined_restriction() {
        let c = FormatConstraint {
            formats: Some(vec![
                ktx2::Format::R8G8B8A8_UNORM,
                ktx2::Format::R32G32B32A32_SFLOAT,
            ]),
            color_spaces: Some(vec![ColorSpace::Linear]),
            alpha_modes: None,
        };
        // Right format, wrong color space
        assert!(!c.accepts(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        ));
        // Right format, right color space
        assert!(c.accepts(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        ));
    }
}
