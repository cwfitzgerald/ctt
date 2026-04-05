use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;

use std::fmt;

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::{ChannelKind, FormatExt};

/// The reason a format conversion is lossy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LossyReason {
    /// Channel count reduced (e.g. RGBA → R).
    ChannelCountReduction { from: usize, to: usize },
    /// Channel kind conversion loses precision (e.g. f32 → u16).
    ChannelKindPrecisionLoss { from: ChannelKind, to: ChannelKind },
    /// Color space changed at the same channel precision (e.g. u8 sRGB → u8 linear).
    ColorSpaceChangeAtSamePrecision {
        from_cs: ColorSpace,
        to_cs: ColorSpace,
        kind: ChannelKind,
    },
}

impl fmt::Display for LossyReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ChannelCountReduction { from, to } => {
                write!(f, "channel count reduced from {from} to {to}")
            }
            Self::ChannelKindPrecisionLoss { from, to } => {
                write!(f, "{from:?} to {to:?} loses precision")
            }
            Self::ColorSpaceChangeAtSamePrecision {
                from_cs,
                to_cs,
                kind,
            } => {
                write!(f, "{from_cs} to {to_cs} at {kind:?} precision is lossy")
            }
        }
    }
}

/// Check whether a conversion from `from` to `to` is lossless.
///
/// Returns `Ok(())` if lossless, or `Err(LossyReason)` explaining why it is lossy.
pub fn check_lossless(from: FormatState, to: FormatState) -> std::result::Result<(), LossyReason> {
    let from_cc = from.format.channel_count().unwrap_or(0);
    let to_cc = to.format.channel_count().unwrap_or(0);

    // Rule 1: channel count reduction.
    if to_cc < from_cc {
        return Err(LossyReason::ChannelCountReduction {
            from: from_cc,
            to: to_cc,
        });
    }

    let from_ck = from.format.channel_kind();
    let to_ck = to.format.channel_kind();

    if let (Some(fk), Some(tk)) = (from_ck, to_ck) {
        // Rule 2: channel kind precision loss.
        if !is_lossless_kind_conversion(fk, tk) {
            return Err(LossyReason::ChannelKindPrecisionLoss { from: fk, to: tk });
        }

        // Rule 3: color space change at same precision.
        if from.color_space != to.color_space && fk == tk {
            return Err(LossyReason::ColorSpaceChangeAtSamePrecision {
                from_cs: from.color_space,
                to_cs: to.color_space,
                kind: fk,
            });
        }
    }

    Ok(())
}

/// Returns `true` if the channel kind conversion preserves all values.
fn is_lossless_kind_conversion(from: ChannelKind, to: ChannelKind) -> bool {
    use ChannelKind::*;
    matches!(
        (from, to),
        (U8, U8 | U16 | U32 | F16 | F32)
            | (U16, U16 | U32 | F32)
            | (F16, F16 | F32)
            | (F32, F32)
            | (U32, U32)
    )
}

/// Type alias for a surface conversion function.
pub type SurfaceConverter = Arc<dyn Fn(&Surface) -> Result<Surface> + Send + Sync>;

/// A format + color space + alpha mode triple representing the full state of an image's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FormatState {
    pub format: ktx2::Format,
    pub color_space: ColorSpace,
    pub alpha: AlphaMode,
}

impl fmt::Display for FormatState {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:?} ({}, {})",
            self.format, self.color_space, self.alpha
        )
    }
}

impl FormatState {
    pub fn new(format: ktx2::Format, color_space: ColorSpace, alpha: AlphaMode) -> Self {
        Self {
            format,
            color_space,
            alpha,
        }
    }

    /// Check if this state satisfies a [`FormatConstraint`].
    pub fn satisfies(&self, constraint: &FormatConstraint) -> bool {
        constraint.accepts(self.format, self.color_space, self.alpha)
    }
}

/// A directed edge in the conversion graph with an explicit target state.
pub struct ExactEdge {
    /// The target format state after conversion.
    pub target: FormatState,
    /// Cost of this conversion (lower is better).
    pub cost: u32,
    /// The function that performs the conversion on a single surface.
    pub converter: SurfaceConverter,
}

/// A directed edge keyed by format only — color space and alpha mode pass through from the source.
pub struct FormatEdge {
    /// The target format (cs/alpha inherited from source).
    pub target_format: ktx2::Format,
    /// Cost of this conversion (lower is better).
    pub cost: u32,
    /// The function that performs the conversion on a single surface.
    pub converter: SurfaceConverter,
}

/// A graph of format conversions with cost-based shortest-path resolution.
///
/// Nodes are [`FormatState`] values. Edges come in two tiers:
/// - **Format edges**: keyed by format only, color space and alpha pass through from the source.
/// - **Exact edges**: keyed by full [`FormatState`], for transitions that change color space or alpha.
pub struct ConversionGraph {
    /// Edges keyed by format only — cs/alpha pass through from source.
    format_edges: HashMap<ktx2::Format, Vec<FormatEdge>>,
    /// Edges keyed by full FormatState — for cs/alpha transitions.
    exact_edges: HashMap<FormatState, Vec<ExactEdge>>,
}

impl Default for ConversionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionGraph {
    pub fn new() -> Self {
        Self {
            format_edges: HashMap::new(),
            exact_edges: HashMap::new(),
        }
    }

    /// Add a format-only edge (cs/alpha pass through from source).
    pub fn add_format_edge(&mut self, from_format: ktx2::Format, edge: FormatEdge) {
        self.format_edges.entry(from_format).or_default().push(edge);
    }

    /// Add an exact edge with a specific source and target state.
    pub fn add_exact_edge(&mut self, from: FormatState, edge: ExactEdge) {
        self.exact_edges.entry(from).or_default().push(edge);
    }

    /// Iterate all outgoing edges from `state`, yielding `(target, cost, converter)`.
    ///
    /// Format edges have their target resolved using the source state's cs/alpha.
    fn edges_from(
        &self,
        state: FormatState,
    ) -> impl Iterator<Item = (FormatState, u32, &SurfaceConverter)> {
        let format_iter = self
            .format_edges
            .get(&state.format)
            .into_iter()
            .flat_map(move |edges| {
                edges.iter().map(move |e| {
                    let target = FormatState::new(e.target_format, state.color_space, state.alpha);
                    (target, e.cost, &e.converter)
                })
            });

        let exact_iter = self
            .exact_edges
            .get(&state)
            .into_iter()
            .flat_map(|edges| edges.iter().map(|e| (e.target, e.cost, &e.converter)));

        format_iter.chain(exact_iter)
    }

    /// Find the shortest path from `from` to `to` using Dijkstra's algorithm.
    ///
    /// Returns the sequence of intermediate states (excluding `from`, including `to`),
    /// or `None` if no path exists. Returns an empty vec if `from == to`.
    pub fn find_path(&self, from: FormatState, to: FormatState) -> Option<Vec<FormatState>> {
        if from == to {
            return Some(Vec::new());
        }

        let mut dist: HashMap<FormatState, u32> = HashMap::new();
        let mut prev: HashMap<FormatState, FormatState> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(u32, FormatState)>> = BinaryHeap::new();

        dist.insert(from, 0);
        heap.push(Reverse((0, from)));

        while let Some(Reverse((cost, state))) = heap.pop() {
            if state == to {
                return Some(Self::reconstruct_path(&prev, from, to));
            }

            if cost > *dist.get(&state).unwrap_or(&u32::MAX) {
                continue;
            }

            for (target, edge_cost, _) in self.edges_from(state) {
                let new_cost = cost + edge_cost;
                if new_cost < *dist.get(&target).unwrap_or(&u32::MAX) {
                    dist.insert(target, new_cost);
                    prev.insert(target, state);
                    heap.push(Reverse((new_cost, target)));
                }
            }
        }

        None
    }

    /// Find the shortest path from `from` to any state satisfying `constraint`.
    pub fn find_path_to_constraint(
        &self,
        from: FormatState,
        constraint: &FormatConstraint,
    ) -> Option<Vec<FormatState>> {
        if from.satisfies(constraint) {
            return Some(Vec::new());
        }

        let mut dist: HashMap<FormatState, u32> = HashMap::new();
        let mut prev: HashMap<FormatState, FormatState> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(u32, FormatState)>> = BinaryHeap::new();

        dist.insert(from, 0);
        heap.push(Reverse((0, from)));

        while let Some(Reverse((cost, state))) = heap.pop() {
            if state != from && state.satisfies(constraint) {
                return Some(Self::reconstruct_path(&prev, from, state));
            }

            if cost > *dist.get(&state).unwrap_or(&u32::MAX) {
                continue;
            }

            for (target, edge_cost, _) in self.edges_from(state) {
                let new_cost = cost + edge_cost;
                if new_cost < *dist.get(&target).unwrap_or(&u32::MAX) {
                    dist.insert(target, new_cost);
                    prev.insert(target, state);
                    heap.push(Reverse((new_cost, target)));
                }
            }
        }

        None
    }

    /// Look up the converter function for a direct single-hop conversion.
    pub fn get_converter(&self, from: FormatState, to: FormatState) -> Option<&SurfaceConverter> {
        // Check format edges first (cs/alpha pass through).
        if from.color_space == to.color_space && from.alpha == to.alpha {
            if let Some(converter) = self.format_edges.get(&from.format).and_then(|edges| {
                edges
                    .iter()
                    .find(|e| e.target_format == to.format)
                    .map(|e| &e.converter)
            }) {
                return Some(converter);
            }
        }

        // Check exact edges.
        self.exact_edges.get(&from)?.iter().find_map(|edge| {
            if edge.target == to {
                Some(&edge.converter)
            } else {
                None
            }
        })
    }

    fn reconstruct_path(
        prev: &HashMap<FormatState, FormatState>,
        from: FormatState,
        to: FormatState,
    ) -> Vec<FormatState> {
        let mut path = Vec::new();
        let mut current = to;
        while current != from {
            path.push(current);
            current = *prev.get(&current).expect("broken path in reconstruct_path");
        }
        path.reverse();
        path
    }
}

// ---- Format conversion logic (merged from transform/convert.rs) ----

/// Convert a surface to a different uncompressed format.
///
/// Supports channel extraction (RGBA->R, RGBA->RG), channel expansion
/// (R->RGBA, RG->RGBA, RGB->RGBA), and bit-depth conversion between
/// U8, U16, F16, and F32.
pub fn convert_surface(surface: &Surface, target: ktx2::Format) -> Result<Surface> {
    if surface.format == target {
        return Ok(surface.clone());
    }

    let src_cc = surface
        .format
        .channel_count()
        .expect("unknown src channel count");
    let src_ck = surface
        .format
        .channel_kind()
        .expect("unknown src channel kind");
    let src_cs = src_ck.byte_size();
    let src_bpp = src_cc * src_cs;

    let dst_cc = target.channel_count().expect("unknown dst channel count");
    let dst_ck = target.channel_kind().expect("unknown dst channel kind");
    let dst_cs = dst_ck.byte_size();
    let dst_bpp = dst_cc * dst_cs;

    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let dst_stride = width * dst_bpp;

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        for x in 0..width {
            let src_off = y * src_stride + x * src_bpp;
            let dst_off = y * dst_stride + x * dst_bpp;

            for dst_ch in 0..dst_cc {
                let val = if dst_ch < src_cc {
                    let ch_off = src_off + dst_ch * src_cs;
                    read_channel(&surface.data, ch_off, src_ck)
                } else {
                    // Expansion: alpha defaults to max, others to 0.
                    if dst_ch == 3 { 1.0 } else { 0.0 }
                };

                let ch_off = dst_off + dst_ch * dst_cs;
                write_channel(&mut out, ch_off, dst_ck, val);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: surface.color_space,
        alpha: surface.alpha,
    })
}

/// Read a single channel value as f64, normalized to [0, 1] for integer types.
fn read_channel(data: &[u8], offset: usize, ck: ChannelKind) -> f64 {
    match ck {
        ChannelKind::U8 => data[offset] as f64 / 255.0,
        ChannelKind::U16 => {
            let v = u16::from_le_bytes([data[offset], data[offset + 1]]);
            v as f64 / 65535.0
        }
        ChannelKind::F16 => {
            let bits = u16::from_le_bytes([data[offset], data[offset + 1]]);
            half::f16::from_bits(bits).to_f64()
        }
        ChannelKind::F32 => {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            f32::from_le_bytes(bytes) as f64
        }
        ChannelKind::U32 => {
            let bytes = [
                data[offset],
                data[offset + 1],
                data[offset + 2],
                data[offset + 3],
            ];
            u32::from_le_bytes(bytes) as f64 / u32::MAX as f64
        }
    }
}

/// Write a single channel value (f64, normalized [0,1] for integer types).
fn write_channel(data: &mut [u8], offset: usize, ck: ChannelKind, val: f64) {
    match ck {
        ChannelKind::U8 => {
            data[offset] = (val.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
        ChannelKind::U16 => {
            let v = (val.clamp(0.0, 1.0) * 65535.0).round() as u16;
            data[offset..offset + 2].copy_from_slice(&v.to_le_bytes());
        }
        ChannelKind::F16 => {
            let h = half::f16::from_f64(val);
            data[offset..offset + 2].copy_from_slice(&h.to_le_bytes());
        }
        ChannelKind::F32 => {
            let v = val as f32;
            data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        }
        ChannelKind::U32 => {
            let v = (val.clamp(0.0, 1.0) * u32::MAX as f64).round() as u32;
            data[offset..offset + 4].copy_from_slice(&v.to_le_bytes());
        }
    }
}

/// Apply the sRGB EOTF (electro-optical transfer function) to convert a single channel from
/// sRGB-encoded to linear.
fn srgb_eotf(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

/// Apply the inverse sRGB EOTF (OETF) to convert a single channel from linear to sRGB-encoded.
fn srgb_oetf(c: f64) -> f64 {
    if c <= 0.0031308 {
        c * 12.92
    } else {
        1.055 * c.powf(1.0 / 2.4) - 0.055
    }
}

/// Convert a surface from sRGB u8 unorm to linear f32.
///
/// RGB channels get the sRGB EOTF applied; alpha (if present) is treated as already linear
/// and is just rescaled.
fn srgb_to_linear(surface: &Surface, target: ktx2::Format, has_alpha: bool) -> Result<Surface> {
    let src_cc = surface
        .format
        .channel_count()
        .expect("unknown src channel count");
    let dst_cc = target.channel_count().expect("unknown dst channel count");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let dst_stride = width * dst_cc * 4; // f32 = 4 bytes

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        for x in 0..width {
            let src_off = y * src_stride + x * src_cc;
            let dst_off = y * dst_stride + x * dst_cc * 4;

            for ch in 0..dst_cc {
                let val = if ch < src_cc {
                    let raw = surface.data[src_off + ch] as f64 / 255.0;
                    if has_alpha && ch == 3 {
                        raw // alpha is linear
                    } else {
                        srgb_eotf(raw)
                    }
                } else if ch == 3 {
                    1.0 // alpha default
                } else {
                    0.0
                };

                let v = val as f32;
                let off = dst_off + ch * 4;
                out[off..off + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: ColorSpace::Linear,
        alpha: surface.alpha,
    })
}

/// Convert a surface from linear f32 to sRGB u8 unorm.
///
/// RGB channels get the inverse sRGB EOTF applied; alpha (if present) is treated as linear
/// and is just rescaled.
fn linear_to_srgb(surface: &Surface, target: ktx2::Format, has_alpha: bool) -> Result<Surface> {
    let src_cc = surface
        .format
        .channel_count()
        .expect("unknown src channel count");
    let dst_cc = target.channel_count().expect("unknown dst channel count");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let src_stride = surface.stride as usize;
    let dst_stride = width * dst_cc; // u8 = 1 byte

    let mut out = vec![0u8; dst_stride * height];

    for y in 0..height {
        for x in 0..width {
            let src_off = y * src_stride + x * src_cc * 4;
            let dst_off = y * dst_stride + x * dst_cc;

            for ch in 0..dst_cc {
                let linear = if ch < src_cc {
                    let off = src_off + ch * 4;
                    let bytes = [
                        surface.data[off],
                        surface.data[off + 1],
                        surface.data[off + 2],
                        surface.data[off + 3],
                    ];
                    f32::from_le_bytes(bytes) as f64
                } else if ch == 3 {
                    1.0
                } else {
                    0.0
                };

                let encoded = if has_alpha && ch == 3 {
                    linear // alpha stays linear
                } else {
                    srgb_oetf(linear)
                };

                out[dst_off + ch] = (encoded.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: dst_stride as u32,
        format: target,
        color_space: ColorSpace::Srgb,
        alpha: surface.alpha,
    })
}

/// Premultiply alpha: RGB *= A. Operates on normalized [0,1] values.
fn premultiply_alpha(surface: &Surface) -> Result<Surface> {
    let cc = surface
        .format
        .channel_count()
        .expect("unknown channel count");
    let ck = surface.format.channel_kind().expect("unknown channel kind");
    let cs = ck.byte_size();
    let bpp = cc * cs;

    assert!(cc == 4, "premultiply_alpha requires 4-channel format");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let stride = surface.stride as usize;

    let mut out = surface.data.clone();

    for y in 0..height {
        for x in 0..width {
            let off = y * stride + x * bpp;
            let alpha = read_channel(&surface.data, off + 3 * cs, ck);

            for ch in 0..3 {
                let val = read_channel(&surface.data, off + ch * cs, ck);
                write_channel(&mut out, off + ch * cs, ck, val * alpha);
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: surface.stride,
        format: surface.format,
        color_space: surface.color_space,
        alpha: AlphaMode::Premultiplied,
    })
}

/// Unpremultiply alpha: RGB /= A. Operates on normalized [0,1] values.
fn unpremultiply_alpha(surface: &Surface) -> Result<Surface> {
    let cc = surface
        .format
        .channel_count()
        .expect("unknown channel count");
    let ck = surface.format.channel_kind().expect("unknown channel kind");
    let cs = ck.byte_size();
    let bpp = cc * cs;

    assert!(cc == 4, "unpremultiply_alpha requires 4-channel format");

    let width = surface.width as usize;
    let height = surface.height as usize;
    let stride = surface.stride as usize;

    let mut out = surface.data.clone();

    for y in 0..height {
        for x in 0..width {
            let off = y * stride + x * bpp;
            let alpha = read_channel(&surface.data, off + 3 * cs, ck);

            if alpha > 0.0 {
                for ch in 0..3 {
                    let val = read_channel(&surface.data, off + ch * cs, ck);
                    write_channel(&mut out, off + ch * cs, ck, val / alpha);
                }
            }
        }
    }

    Ok(Surface {
        data: out,
        width: surface.width,
        height: surface.height,
        stride: surface.stride,
        format: surface.format,
        color_space: surface.color_space,
        alpha: AlphaMode::Straight,
    })
}

/// Build the default conversion graph.
///
/// Format edges: `{R, RG, RGB, RGBA} x {U8, U16, F16, F32}` — work at any color space and alpha mode.
/// Exact edges: sRGB ↔ linear (u8 unorm ↔ f32, alpha stays linear), premultiply/unpremultiply (RGBA, linear only).
pub fn build_default_graph() -> ConversionGraph {
    use ktx2::Format as F;

    let formats = [
        F::R8_UNORM,
        F::R16_UNORM,
        F::R16_SFLOAT,
        F::R32_SFLOAT,
        F::R8G8_UNORM,
        F::R16G16_UNORM,
        F::R16G16_SFLOAT,
        F::R32G32_SFLOAT,
        F::R8G8B8_UNORM,
        F::R16G16B16_UNORM,
        F::R16G16B16_SFLOAT,
        F::R32G32B32_SFLOAT,
        F::R8G8B8A8_UNORM,
        F::R16G16B16A16_UNORM,
        F::R16G16B16A16_SFLOAT,
        F::R32G32B32A32_SFLOAT,
    ];

    let mut graph = ConversionGraph::new();

    // Format-only edges: format conversion, cs/alpha pass through.
    for &src in &formats {
        for &dst in &formats {
            if src == dst {
                continue;
            }

            let cost = conversion_cost(src, dst);
            let converter: SurfaceConverter =
                Arc::new(move |surface: &Surface| convert_surface(surface, dst));

            graph.add_format_edge(
                src,
                FormatEdge {
                    target_format: dst,
                    cost,
                    converter,
                },
            );
        }
    }

    // sRGB ↔ linear exact edges (u8 unorm srgb ↔ f32 linear, alpha stays linear).
    let srgb_pairs = [
        (F::R8_UNORM, F::R32_SFLOAT),
        (F::R8G8_UNORM, F::R32G32_SFLOAT),
        (F::R8G8B8_UNORM, F::R32G32B32_SFLOAT),
        (F::R8G8B8A8_UNORM, F::R32G32B32A32_SFLOAT),
    ];

    for alpha in [
        AlphaMode::Straight,
        AlphaMode::Premultiplied,
        AlphaMode::Opaque,
    ] {
        for &(u8_fmt, f32_fmt) in &srgb_pairs {
            let has_alpha = u8_fmt.channel_count().unwrap_or(0) == 4;

            // sRGB u8 → linear f32
            {
                let from = FormatState::new(u8_fmt, ColorSpace::Srgb, alpha);
                let to = FormatState::new(f32_fmt, ColorSpace::Linear, alpha);
                let converter: SurfaceConverter =
                    Arc::new(move |surface: &Surface| srgb_to_linear(surface, f32_fmt, has_alpha));
                graph.add_exact_edge(
                    from,
                    ExactEdge {
                        target: to,
                        cost: 10,
                        converter,
                    },
                );
            }

            // linear f32 → sRGB u8
            {
                let from = FormatState::new(f32_fmt, ColorSpace::Linear, alpha);
                let to = FormatState::new(u8_fmt, ColorSpace::Srgb, alpha);
                let converter: SurfaceConverter =
                    Arc::new(move |surface: &Surface| linear_to_srgb(surface, u8_fmt, has_alpha));
                graph.add_exact_edge(
                    from,
                    ExactEdge {
                        target: to,
                        cost: 10,
                        converter,
                    },
                );
            }
        }
    }

    // Premultiply/unpremultiply exact edges (RGBA formats, linear only).
    let rgba_formats = [
        F::R8G8B8A8_UNORM,
        F::R16G16B16A16_UNORM,
        F::R16G16B16A16_SFLOAT,
        F::R32G32B32A32_SFLOAT,
    ];

    for &fmt in &rgba_formats {
        // straight → premultiplied
        {
            let from = FormatState::new(fmt, ColorSpace::Linear, AlphaMode::Straight);
            let to = FormatState::new(fmt, ColorSpace::Linear, AlphaMode::Premultiplied);
            let converter: SurfaceConverter =
                Arc::new(move |surface: &Surface| premultiply_alpha(surface));
            graph.add_exact_edge(
                from,
                ExactEdge {
                    target: to,
                    cost: 5,
                    converter,
                },
            );
        }

        // premultiplied → straight
        {
            let from = FormatState::new(fmt, ColorSpace::Linear, AlphaMode::Premultiplied);
            let to = FormatState::new(fmt, ColorSpace::Linear, AlphaMode::Straight);
            let converter: SurfaceConverter =
                Arc::new(move |surface: &Surface| unpremultiply_alpha(surface));
            graph.add_exact_edge(
                from,
                ExactEdge {
                    target: to,
                    cost: 5,
                    converter,
                },
            );
        }
    }

    graph
}

/// Compute the cost of converting between two formats.
fn conversion_cost(from: ktx2::Format, to: ktx2::Format) -> u32 {
    let mut cost = 0u32;

    let from_cc = from.channel_count().unwrap_or(4);
    let to_cc = to.channel_count().unwrap_or(4);
    if from_cc != to_cc {
        cost += 20;
    }

    let from_ck = from.channel_kind();
    let to_ck = to.channel_kind();
    if from_ck != to_ck {
        let type_cost = match (from_ck, to_ck) {
            (Some(ChannelKind::U8), Some(ChannelKind::U16))
            | (Some(ChannelKind::U16), Some(ChannelKind::U8)) => 5,
            (Some(ChannelKind::F16), _) | (_, Some(ChannelKind::F16)) => 15,
            _ => 10,
        };
        cost += type_cost;
    }

    let dst_size = to.bytes_per_pixel().unwrap_or(4);
    let src_size = from.bytes_per_pixel().unwrap_or(4);
    if dst_size > src_size {
        cost += (dst_size - src_size) as u32;
    }

    cost
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rgba8_linear() -> FormatState {
        FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        )
    }

    fn r8_linear() -> FormatState {
        FormatState::new(
            ktx2::Format::R8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        )
    }

    fn rgba32f_linear() -> FormatState {
        FormatState::new(
            ktx2::Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Straight,
        )
    }

    #[test]
    fn identity_path_is_empty() {
        let graph = build_default_graph();
        let path = graph.find_path(rgba8_linear(), rgba8_linear());
        assert_eq!(path, Some(Vec::new()));
    }

    #[test]
    fn direct_conversion_exists() {
        let graph = build_default_graph();
        let path = graph.find_path(rgba8_linear(), rgba32f_linear());
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(!path.is_empty());
        assert_eq!(*path.last().unwrap(), rgba32f_linear());
    }

    #[test]
    fn channel_expansion_path() {
        let graph = build_default_graph();
        let path = graph.find_path(r8_linear(), rgba8_linear());
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(*path.last().unwrap(), rgba8_linear());
    }

    #[test]
    fn find_path_to_constraint_already_satisfied() {
        let graph = build_default_graph();
        let constraint = FormatConstraint {
            formats: Some(vec![ktx2::Format::R8G8B8A8_UNORM]),
            color_spaces: None,
            alpha_modes: None,
        };
        let path = graph.find_path_to_constraint(rgba8_linear(), &constraint);
        assert_eq!(path, Some(Vec::new()));
    }

    #[test]
    fn find_path_to_constraint_needs_conversion() {
        let graph = build_default_graph();
        let constraint = FormatConstraint {
            formats: Some(vec![ktx2::Format::R8G8B8A8_UNORM]),
            color_spaces: None,
            alpha_modes: None,
        };
        let path = graph.find_path_to_constraint(r8_linear(), &constraint);
        assert!(path.is_some());
        let path = path.unwrap();
        assert!(path.last().unwrap().satisfies(&constraint));
    }

    #[test]
    fn find_path_to_constraint_picks_cheapest() {
        let graph = build_default_graph();
        let constraint = FormatConstraint {
            formats: Some(vec![
                ktx2::Format::R8G8B8A8_UNORM,
                ktx2::Format::R32G32B32A32_SFLOAT,
            ]),
            color_spaces: None,
            alpha_modes: None,
        };
        let path = graph.find_path_to_constraint(r8_linear(), &constraint);
        assert!(path.is_some());
        let target = path.unwrap().last().unwrap().format;
        assert_eq!(target, ktx2::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn no_path_for_impossible_constraint() {
        let graph = build_default_graph();
        // Use a constraint that truly can't be satisfied: a format not in the graph.
        let constraint = FormatConstraint {
            formats: Some(vec![ktx2::Format::R4G4_UNORM_PACK8]),
            color_spaces: None,
            alpha_modes: None,
        };
        let path = graph.find_path_to_constraint(rgba8_linear(), &constraint);
        assert!(path.is_none());
    }

    #[test]
    fn srgb_to_same_format_different_depth() {
        let graph = build_default_graph();
        let srgb_u8 = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let srgb_u16 = FormatState::new(
            ktx2::Format::R16G16B16A16_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let path = graph.find_path(srgb_u8, srgb_u16);
        assert!(path.is_some());
        let path = path.unwrap();
        assert_eq!(
            path.last().unwrap().format,
            ktx2::Format::R16G16B16A16_UNORM
        );
        assert_eq!(path.last().unwrap().color_space, ColorSpace::Srgb);
    }

    #[test]
    fn srgb_to_linear_path_exists() {
        let graph = build_default_graph();
        let srgb_u8 = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let linear_f32 = FormatState::new(
            ktx2::Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Straight,
        );
        let path = graph.find_path(srgb_u8, linear_f32);
        assert!(path.is_some());
        assert_eq!(*path.unwrap().last().unwrap(), linear_f32);
    }

    #[test]
    fn linear_to_srgb_path_exists() {
        let graph = build_default_graph();
        let linear_f32 = FormatState::new(
            ktx2::Format::R32G32B32A32_SFLOAT,
            ColorSpace::Linear,
            AlphaMode::Straight,
        );
        let srgb_u8 = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let path = graph.find_path(linear_f32, srgb_u8);
        assert!(path.is_some());
        assert_eq!(*path.unwrap().last().unwrap(), srgb_u8);
    }

    #[test]
    fn premultiply_path_exists() {
        let graph = build_default_graph();
        let straight = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Straight,
        );
        let premul = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Linear,
            AlphaMode::Premultiplied,
        );
        let path = graph.find_path(straight, premul);
        assert!(path.is_some());
    }

    #[test]
    fn srgb_roundtrip_surface() {
        // 1x1 pixel: sRGB(128, 64, 32, 200)
        let surface = Surface {
            data: vec![128, 64, 32, 200],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Srgb,
            alpha: AlphaMode::Straight,
        };

        let linear = srgb_to_linear(&surface, ktx2::Format::R32G32B32A32_SFLOAT, true).unwrap();
        assert_eq!(linear.color_space, ColorSpace::Linear);
        assert_eq!(linear.format, ktx2::Format::R32G32B32A32_SFLOAT);

        // Alpha should pass through linearly: 200/255
        let alpha_bytes = &linear.data[12..16];
        let alpha = f32::from_le_bytes(alpha_bytes.try_into().unwrap());
        assert!((alpha - 200.0 / 255.0).abs() < 1e-5);

        let back = linear_to_srgb(&linear, ktx2::Format::R8G8B8A8_UNORM, true).unwrap();
        assert_eq!(back.color_space, ColorSpace::Srgb);
        // Should round-trip within +-1 due to u8 quantization.
        for i in 0..4 {
            assert!(
                (back.data[i] as i16 - surface.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}: {} vs {}",
                back.data[i],
                surface.data[i],
            );
        }
    }

    #[test]
    fn premultiply_roundtrip_surface() {
        // 1x1 pixel with alpha=0.5 (128/255)
        let surface = Surface {
            data: vec![200, 100, 50, 128],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };

        let premul = premultiply_alpha(&surface).unwrap();
        assert_eq!(premul.alpha, AlphaMode::Premultiplied);
        // Alpha should be unchanged
        assert_eq!(premul.data[3], 128);

        let back = unpremultiply_alpha(&premul).unwrap();
        assert_eq!(back.alpha, AlphaMode::Straight);
        // Should round-trip within +-1
        for i in 0..4 {
            assert!(
                (back.data[i] as i16 - surface.data[i] as i16).unsigned_abs() <= 1,
                "channel {i}: {} vs {}",
                back.data[i],
                surface.data[i],
            );
        }
    }

    #[test]
    fn prefers_non_f16_path() {
        let graph = build_default_graph();
        let path = graph.find_path(rgba8_linear(), rgba32f_linear());
        let path = path.unwrap();
        for state in &path[..path.len().saturating_sub(1)] {
            assert_ne!(
                state.format,
                ktx2::Format::R16G16B16A16_SFLOAT,
                "path should not route through F16"
            );
        }
    }

    #[test]
    fn get_converter_works() {
        let graph = build_default_graph();
        let converter = graph.get_converter(rgba8_linear(), rgba32f_linear());
        assert!(converter.is_some());
    }

    #[test]
    fn get_converter_returns_none_for_missing() {
        let graph = build_default_graph();
        let srgb_state = FormatState::new(
            ktx2::Format::R8G8B8A8_UNORM,
            ColorSpace::Srgb,
            AlphaMode::Straight,
        );
        let converter = graph.get_converter(rgba8_linear(), srgb_state);
        assert!(converter.is_none());
    }

    #[test]
    fn format_state_satisfies_constraint() {
        let state = rgba8_linear();
        let c = FormatConstraint::any();
        assert!(state.satisfies(&c));

        let c = FormatConstraint {
            formats: Some(vec![ktx2::Format::R16_UNORM]),
            color_spaces: None,
            alpha_modes: None,
        };
        assert!(!state.satisfies(&c));
    }

    #[test]
    fn convert_surface_no_op_when_same_format() {
        let surface = Surface {
            data: vec![10, 20, 30, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        assert_eq!(result.data, surface.data);
    }

    #[test]
    fn convert_surface_rgba8_to_r8() {
        let surface = Surface {
            data: vec![100, 150, 200, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8_UNORM).unwrap();
        assert_eq!(result.data, vec![100]);
    }

    #[test]
    fn convert_surface_r8_to_rgba8() {
        let surface = Surface {
            data: vec![100],
            width: 1,
            height: 1,
            stride: 1,
            format: ktx2::Format::R8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let result = convert_surface(&surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        // R=100, G=0, B=0, A=255
        assert_eq!(result.data, vec![100, 0, 0, 255]);
    }

    #[test]
    fn convert_surface_u8_to_u16_roundtrip() {
        let surface = Surface {
            data: vec![128, 0, 0, 255],
            width: 1,
            height: 1,
            stride: 4,
            format: ktx2::Format::R8G8B8A8_UNORM,
            color_space: ColorSpace::Linear,
            alpha: AlphaMode::Straight,
        };
        let u16_surface = convert_surface(&surface, ktx2::Format::R16G16B16A16_UNORM).unwrap();
        assert_eq!(u16_surface.data.len(), 8);

        let back = convert_surface(&u16_surface, ktx2::Format::R8G8B8A8_UNORM).unwrap();
        assert_eq!(back.data, surface.data);
    }

    // ---- check_lossless tests ----

    fn state(format: ktx2::Format, cs: ColorSpace) -> FormatState {
        FormatState::new(format, cs, AlphaMode::Straight)
    }

    #[test]
    fn lossless_u8_to_f32() {
        assert!(
            check_lossless(
                state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
                state(ktx2::Format::R32G32B32A32_SFLOAT, ColorSpace::Linear),
            )
            .is_ok()
        );
    }

    #[test]
    fn lossy_u16_to_f16() {
        let result = check_lossless(
            state(ktx2::Format::R16G16B16A16_UNORM, ColorSpace::Linear),
            state(ktx2::Format::R16G16B16A16_SFLOAT, ColorSpace::Linear),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ChannelKindPrecisionLoss { .. })
        ));
    }

    #[test]
    fn lossy_u32_to_f32() {
        let result = check_lossless(
            state(ktx2::Format::R32_UINT, ColorSpace::Linear),
            state(ktx2::Format::R32_SFLOAT, ColorSpace::Linear),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ChannelKindPrecisionLoss { .. })
        ));
    }

    #[test]
    fn lossy_f32_to_f16() {
        let result = check_lossless(
            state(ktx2::Format::R32G32B32A32_SFLOAT, ColorSpace::Linear),
            state(ktx2::Format::R16G16B16A16_SFLOAT, ColorSpace::Linear),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ChannelKindPrecisionLoss { .. })
        ));
    }

    #[test]
    fn lossy_channel_count_reduction() {
        let result = check_lossless(
            state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
            state(ktx2::Format::R8_UNORM, ColorSpace::Linear),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ChannelCountReduction { from: 4, to: 1 })
        ));
    }

    #[test]
    fn lossless_channel_count_expansion() {
        assert!(
            check_lossless(
                state(ktx2::Format::R8_UNORM, ColorSpace::Linear),
                state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
            )
            .is_ok()
        );
    }

    #[test]
    fn lossy_srgb_to_linear_same_precision() {
        let result = check_lossless(
            state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Srgb),
            state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ColorSpaceChangeAtSamePrecision { .. })
        ));
    }

    #[test]
    fn lossless_srgb_to_linear_higher_precision() {
        assert!(
            check_lossless(
                state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Srgb),
                state(ktx2::Format::R32G32B32A32_SFLOAT, ColorSpace::Linear),
            )
            .is_ok()
        );
    }

    #[test]
    fn lossy_linear_to_srgb_same_precision() {
        let result = check_lossless(
            state(ktx2::Format::R32G32B32A32_SFLOAT, ColorSpace::Linear),
            state(ktx2::Format::R32G32B32A32_SFLOAT, ColorSpace::Srgb),
        );
        assert!(matches!(
            result,
            Err(LossyReason::ColorSpaceChangeAtSamePrecision { .. })
        ));
    }

    #[test]
    fn lossless_identity() {
        assert!(
            check_lossless(
                state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
                state(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear),
            )
            .is_ok()
        );
    }
}
