use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum AlphaMode {
    #[default]
    Straight,
    Premultiplied,
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
