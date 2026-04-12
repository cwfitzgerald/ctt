/// Universal quality preset all encoders understand.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Quality {
    UltraFast,
    VeryFast,
    Fast,
    #[default]
    Basic,
    Slow,
    VerySlow,
}
