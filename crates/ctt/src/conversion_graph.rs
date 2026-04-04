use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::cmp::Reverse;

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::{ChannelKind, FormatExt};

/// Type alias for a surface conversion function.
pub type SurfaceConverter = Arc<dyn Fn(&Surface) -> Result<Surface> + Send + Sync>;

/// A format + color space + alpha mode triple representing the full state of an image's format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct FormatState {
    pub format: ktx2::Format,
    pub color_space: ColorSpace,
    pub alpha: AlphaMode,
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

/// A directed edge in the conversion graph.
pub struct ConversionEdge {
    /// The target format state after conversion.
    pub target: FormatState,
    /// Cost of this conversion (lower is better).
    pub cost: u32,
    /// The function that performs the conversion on a single surface.
    pub converter: SurfaceConverter,
}

/// A graph of format conversions with cost-based shortest-path resolution.
///
/// Nodes are [`FormatState`] values. Edges are available conversions with associated costs.
/// The resolver uses Dijkstra's algorithm to find the cheapest conversion path.
pub struct ConversionGraph {
    edges: HashMap<FormatState, Vec<ConversionEdge>>,
}

impl Default for ConversionGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl ConversionGraph {
    pub fn new() -> Self {
        Self {
            edges: HashMap::new(),
        }
    }

    /// Add a directed conversion edge from `from` to `edge.target`.
    pub fn add_edge(&mut self, from: FormatState, edge: ConversionEdge) {
        self.edges.entry(from).or_default().push(edge);
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

            if let Some(edges) = self.edges.get(&state) {
                for edge in edges {
                    let new_cost = cost + edge.cost;
                    if new_cost < *dist.get(&edge.target).unwrap_or(&u32::MAX) {
                        dist.insert(edge.target, new_cost);
                        prev.insert(edge.target, state);
                        heap.push(Reverse((new_cost, edge.target)));
                    }
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

            if let Some(edges) = self.edges.get(&state) {
                for edge in edges {
                    let new_cost = cost + edge.cost;
                    if new_cost < *dist.get(&edge.target).unwrap_or(&u32::MAX) {
                        dist.insert(edge.target, new_cost);
                        prev.insert(edge.target, state);
                        heap.push(Reverse((new_cost, edge.target)));
                    }
                }
            }
        }

        None
    }

    /// Look up the converter function for a direct single-hop conversion.
    pub fn get_converter(
        &self,
        from: FormatState,
        to: FormatState,
    ) -> Option<&SurfaceConverter> {
        self.edges.get(&from)?.iter().find_map(|edge| {
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

    let src_cc = surface.format.channel_count().expect("unknown src channel count");
    let src_ck = surface.format.channel_kind().expect("unknown src channel kind");
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

/// Build the default conversion graph with edges between the 16 basic uncompressed formats:
/// `{R, RG, RGB, RGBA} x {U8, U16, F16, F32}`, all at `Linear` color space and `Straight` alpha.
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

    for &src in &formats {
        for &dst in &formats {
            if src == dst {
                continue;
            }

            let cost = conversion_cost(src, dst);

            let converter: SurfaceConverter = Arc::new(move |surface: &Surface| {
                convert_surface(surface, dst)
            });

            let from = FormatState::new(src, ColorSpace::Linear, AlphaMode::Straight);
            let to = FormatState::new(dst, ColorSpace::Linear, AlphaMode::Straight);

            graph.add_edge(
                from,
                ConversionEdge {
                    target: to,
                    cost,
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
        let constraint = FormatConstraint {
            formats: None,
            color_spaces: Some(vec![ColorSpace::Srgb]),
            alpha_modes: None,
        };
        let path = graph.find_path_to_constraint(rgba8_linear(), &constraint);
        assert!(path.is_none());
    }

    #[test]
    fn prefers_non_f16_path() {
        let graph = build_default_graph();
        let path = graph.find_path(rgba8_linear(), rgba32f_linear());
        let path = path.unwrap();
        for state in &path[..path.len().saturating_sub(1)] {
            assert_ne!(
                state.format, ktx2::Format::R16G16B16A16_SFLOAT,
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
}
