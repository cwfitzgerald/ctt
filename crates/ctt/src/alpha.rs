use std::fmt;

/// How a surface's alpha channel relates to its color channels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum AlphaMode {
    /// Straight (non-premultiplied) alpha: color channels are independent of
    /// alpha. The default.
    #[default]
    Straight,
    /// Premultiplied alpha: color channels have already been multiplied by
    /// alpha.
    Premultiplied,
    /// Fully opaque: the alpha channel carries no transparency information.
    Opaque,
}

impl fmt::Display for AlphaMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Straight => f.write_str("straight"),
            Self::Premultiplied => f.write_str("premultiplied"),
            Self::Opaque => f.write_str("opaque"),
        }
    }
}
