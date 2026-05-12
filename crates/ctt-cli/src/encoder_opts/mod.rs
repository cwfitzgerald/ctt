//! Per-encoder option parsing for the `--<encoder>-opts key=val;...` CLI flags.
//!
//! Each encoder backend has a small `Opts` shim struct (in a submodule) that
//! is `#[derive(Facet)]`-annotated and flat-shaped for CLI ergonomics. The
//! shim mirrors the library's `Settings` type via `into_settings()` — a full
//! struct literal that turns into a compile error if the library adds a field.
//!
//! Facet is used for `--help-encoder` introspection (name + doc + type hint
//! per field). Value parsing goes through each shim's hand-written
//! [`OptsShim::apply_kv`] method, because facet's generic `parse_from_str`
//! covers only scalars (no enums, Options, or arrays).

use std::collections::BTreeSet;
use std::fmt;

use facet::{Facet, Field, Shape, Type, UserType};

pub mod astcenc;
pub mod bc7enc;
pub mod etcpak;
pub mod intel;

/// Implemented by each per-encoder shim. The `apply_kv` method knows how to
/// route a CLI-spelled key (kebab-case) to one of the shim's fields and parse
/// the value string into that field's type.
pub trait OptsShim: Default {
    /// Apply one `key=val` pair to `self`. Errors should be carried as
    /// [`ParseError::BadValue`] so the caller can prepend context.
    fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError>;
}

/// The result of parsing one `--<encoder>-opts` string.
#[derive(Debug)]
pub struct ParsedOpts<T> {
    /// The parsed value, with unset fields left at `T::default()`.
    pub value: T,
    /// CLI-spelled key names the user explicitly set. Used by the
    /// override-warning logic when a top-level CLI flag and an opts key
    /// both target the same library knob.
    pub touched: BTreeSet<String>,
}

/// Errors surfaced by [`parse_opts`].
#[derive(Debug)]
pub enum ParseError {
    /// `key=val` was malformed: missing `=`.
    MissingEquals { pair: String },
    /// The key did not match any field of the target shim.
    UnknownKey { key: String, valid: Vec<String> },
    /// The value failed to parse as the field's type.
    BadValue { key: String, message: String },
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEquals { pair } => write!(f, "expected key=value, got `{pair}`"),
            Self::UnknownKey { key, valid } => {
                write!(
                    f,
                    "unknown option `{key}`; valid keys: {}",
                    valid.join(", ")
                )
            }
            Self::BadValue { key, message } => write!(f, "while setting `{key}`: {message}"),
        }
    }
}

impl std::error::Error for ParseError {}

/// Parse a `key=val[;key=val...]` string into a shim.
///
/// Pair separator is `;`. Values may contain `,` (used for arrays like
/// `channel-weights=1,1,1,2`). Facet's shape introspection is used to
/// reject unknown keys with a list of valid alternatives; value parsing
/// delegates to [`OptsShim::apply_kv`].
pub fn parse_opts<'facet, T>(input: &str) -> Result<ParsedOpts<T>, ParseError>
where
    T: Facet<'facet> + OptsShim + 'facet,
{
    let fields = struct_fields(T::SHAPE);

    let mut value = T::default();
    let mut touched = BTreeSet::new();

    for raw_pair in split_pairs(input) {
        let pair = raw_pair.trim();
        if pair.is_empty() {
            continue;
        }
        let (key, val) = pair
            .split_once('=')
            .ok_or_else(|| ParseError::MissingEquals {
                pair: pair.to_string(),
            })?;
        let key = key.trim();
        let val = val.trim();

        if let Some(fs) = fields
            && !fs.iter().any(|f| f.effective_name() == key)
        {
            return Err(ParseError::UnknownKey {
                key: key.into(),
                valid: fs.iter().map(|f| f.effective_name().to_string()).collect(),
            });
        }

        value.apply_kv(key, val)?;
        touched.insert(key.to_string());
    }

    Ok(ParsedOpts { value, touched })
}

fn struct_fields(shape: &'static Shape) -> Option<&'static [Field]> {
    if let Type::User(UserType::Struct(s)) = shape.ty {
        Some(s.fields)
    } else {
        None
    }
}

/// Split a `key=val;key=val` string into pairs, keeping `,` inside values
/// intact (so array forms like `channel-weights=1,1,1,2` parse correctly).
fn split_pairs(input: &str) -> impl Iterator<Item = &str> {
    input.split(';')
}

/// Render `--help-encoder NAME` output for any Facet-derived shim.
pub fn print_help_encoder<'facet, T>(encoder_name: &str)
where
    T: Facet<'facet> + 'facet,
{
    let Some(fields) = struct_fields(T::SHAPE) else {
        println!("internal error: {encoder_name} shim is not a struct");
        return;
    };

    println!("{encoder_name} options — pass as `--{encoder_name}-opts key=val[;key=val...]`");
    println!();
    for field in fields {
        let key = field.effective_name();
        let type_hint = render_type_hint(field.shape());
        println!("  {key:<22} <{type_hint}>");
        for line in field.doc {
            let trimmed = line.trim();
            if !trimmed.is_empty() {
                println!("    {trimmed}");
            }
        }
        println!();
    }
}

fn render_type_hint(shape: &'static Shape) -> String {
    use facet::Def;
    match shape.def {
        Def::Option(opt) => format!("optional {}", render_type_hint(opt.t())),
        Def::Array(arr) => format!("[{}; {}]", render_type_hint(arr.t), arr.n),
        _ => {
            // For enums, list the variants as a `|` chain.
            if let Type::User(UserType::Enum(e)) = shape.ty {
                e.variants
                    .iter()
                    .map(|v| v.effective_name())
                    .collect::<Vec<_>>()
                    .join("|")
            } else {
                shape.type_identifier.to_string()
            }
        }
    }
}

/// Common parsing helpers used by the per-encoder `apply_kv` implementations.
pub(crate) mod parse_helpers {
    use super::ParseError;

    pub fn bool(key: &str, value: &str) -> Result<bool, ParseError> {
        match value {
            "true" | "1" | "yes" | "on" => Ok(true),
            "false" | "0" | "no" | "off" | "" => Ok(false),
            other => Err(ParseError::BadValue {
                key: key.into(),
                message: format!("expected bool, got `{other}`"),
            }),
        }
    }

    pub fn f32(key: &str, value: &str) -> Result<f32, ParseError> {
        value
            .parse()
            .map_err(|e: std::num::ParseFloatError| ParseError::BadValue {
                key: key.into(),
                message: e.to_string(),
            })
    }

    pub fn f32_array<const N: usize>(key: &str, value: &str) -> Result<[f32; N], ParseError> {
        let parts: Vec<&str> = value.split(',').map(str::trim).collect();
        if parts.len() != N {
            return Err(ParseError::BadValue {
                key: key.into(),
                message: format!("expected {N} comma-separated floats, got {}", parts.len()),
            });
        }
        let mut out = [0.0; N];
        for (i, p) in parts.iter().enumerate() {
            out[i] = f32(key, p)?;
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use facet::Facet;

    #[derive(Facet, Debug, Default, PartialEq)]
    #[facet(rename_all = "kebab-case")]
    struct Sample {
        /// A boolean knob.
        flag: bool,
        /// A float in 0..=100.
        weight: f32,
        /// Optional channel weights.
        weights: Option<[f32; 4]>,
        /// A subcategory.
        kind: SampleKind,
    }

    #[derive(Facet, Debug, Default, Clone, Copy, PartialEq)]
    #[repr(u8)]
    #[facet(rename_all = "kebab-case")]
    enum SampleKind {
        #[default]
        Plain,
        FancyOne,
    }

    impl OptsShim for Sample {
        fn apply_kv(&mut self, key: &str, value: &str) -> Result<(), ParseError> {
            match key {
                "flag" => self.flag = parse_helpers::bool(key, value)?,
                "weight" => self.weight = parse_helpers::f32(key, value)?,
                "weights" => self.weights = Some(parse_helpers::f32_array::<4>(key, value)?),
                "kind" => {
                    self.kind = match value {
                        "plain" => SampleKind::Plain,
                        "fancy-one" => SampleKind::FancyOne,
                        other => {
                            return Err(ParseError::BadValue {
                                key: key.into(),
                                message: format!("unknown kind `{other}`"),
                            });
                        }
                    }
                }
                _ => unreachable!("parser pre-validates the key against Facet's field list"),
            }
            Ok(())
        }
    }

    #[test]
    fn parses_scalars_and_enum() {
        let parsed = parse_opts::<Sample>("flag=true;kind=fancy-one;weight=3.5").unwrap();
        assert!(parsed.value.flag);
        assert_eq!(parsed.value.kind, SampleKind::FancyOne);
        assert_eq!(parsed.value.weight, 3.5);
        assert_eq!(parsed.value.weights, None);
        let touched: Vec<&str> = parsed.touched.iter().map(String::as_str).collect();
        assert_eq!(touched, vec!["flag", "kind", "weight"]);
    }

    #[test]
    fn parses_array_value() {
        let parsed = parse_opts::<Sample>("weights=1,2,3,4").unwrap();
        assert_eq!(parsed.value.weights, Some([1.0, 2.0, 3.0, 4.0]));
    }

    #[test]
    fn unset_fields_take_defaults() {
        let parsed = parse_opts::<Sample>("flag=true").unwrap();
        assert_eq!(parsed.value.weight, 0.0);
        assert_eq!(parsed.value.kind, SampleKind::Plain);
        assert!(parsed.touched.contains("flag"));
        assert!(!parsed.touched.contains("kind"));
    }

    #[test]
    fn unknown_key_lists_valid_options() {
        let err = parse_opts::<Sample>("bogus=1").unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus"));
        assert!(msg.contains("flag"));
        assert!(msg.contains("kind"));
    }
}
