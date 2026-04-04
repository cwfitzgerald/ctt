#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Default)]
pub enum AlphaMode {
    #[default]
    Straight,
    Premultiplied,
    Opaque,
}
