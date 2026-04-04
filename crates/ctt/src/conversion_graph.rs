use std::collections::{BinaryHeap, HashMap};
use std::sync::Arc;
use std::cmp::Reverse;

use crate::alpha::AlphaMode;
use crate::constraint::FormatConstraint;
use crate::error::Result;
use crate::format::ColorSpace;
use crate::surface::Surface;

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
    ///
    /// Returns the sequence of intermediate states (excluding `from`, including the target),
    /// or `None` if no path exists. Returns an empty vec if `from` already satisfies the
    /// constraint.
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

/// Build the default conversion graph with edges ported from the existing `convert_image` logic.
///
/// This covers conversions between the 16 basic uncompressed formats:
/// `{R, RG, RGB, RGBA} × {U8, U16, F16, F32}`, all at `Linear` color space and `Straight` alpha.
///
/// Color space and alpha mode conversions can be added later as additional edges.
pub fn build_default_graph() -> ConversionGraph {
    use crate::format::{ChannelType, PixelComponents, PixelFormat};
    use crate::transform::convert::convert_image;
    use crate::vk_format::FormatExt;

    let components = [
        PixelComponents::R,
        PixelComponents::Rg,
        PixelComponents::Rgb,
        PixelComponents::Rgba,
    ];
    let channel_types = [ChannelType::U8, ChannelType::U16, ChannelType::F16, ChannelType::F32];

    // Build all 16 (format, pixel_format) pairs
    let mut formats: Vec<(ktx2::Format, PixelFormat)> = Vec::with_capacity(16);
    for &comp in &components {
        for &ct in &channel_types {
            let pf = PixelFormat {
                components: comp,
                channel_type: ct,
                color_space: ColorSpace::Linear,
            };
            let vk = ktx2::Format::from_pixel_format(pf);
            let (vk, _) = vk.normalize();
            formats.push((vk, pf));
        }
    }

    let mut graph = ConversionGraph::new();

    for &(src_vk, src_pf) in &formats {
        for &(dst_vk, dst_pf) in &formats {
            if src_vk == dst_vk {
                continue;
            }

            let cost = conversion_cost(src_pf, dst_pf);

            let converter: SurfaceConverter = {
                Arc::new(move |surface: &Surface| {
                    let raw = surface.to_raw_image()?;
                    let converted = convert_image(&raw, dst_pf)?;
                    let mut result = Surface::from_raw_image(converted);
                    // Preserve the original color space and alpha
                    result.color_space = surface.color_space;
                    result.alpha = surface.alpha;
                    Ok(result)
                })
            };

            let from = FormatState::new(src_vk, ColorSpace::Linear, AlphaMode::Straight);
            let to = FormatState::new(dst_vk, ColorSpace::Linear, AlphaMode::Straight);

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

/// Compute the cost of converting between two pixel formats.
///
/// Heuristics:
/// - Same components, different bit depth (U8 <-> U16): 5
/// - Same components, to/from F32: 10
/// - Involving F16: +15 penalty (CPU math is slow)
/// - Channel expansion/reduction: 20
/// - Combined changes: sum of costs
fn conversion_cost(from: PixelFormat, to: PixelFormat) -> u32 {
    use crate::format::ChannelType;

    let mut cost = 0u32;

    // Channel count change
    if from.components != to.components {
        cost += 20;
    }

    // Bit depth change
    if from.channel_type != to.channel_type {
        let type_cost = match (from.channel_type, to.channel_type) {
            (ChannelType::U8, ChannelType::U16) | (ChannelType::U16, ChannelType::U8) => 5,
            (ChannelType::F16, _) | (_, ChannelType::F16) => 15,
            _ => 10,
        };
        cost += type_cost;
    }

    // Prefer smaller output formats for memory bandwidth
    let dst_size = to.bytes_per_pixel();
    let src_size = from.bytes_per_pixel();
    if dst_size > src_size {
        cost += (dst_size - src_size) as u32;
    }

    cost
}

use crate::format::PixelFormat;

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
        // Should be a direct hop (or short path)
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
        // Constraint accepts both RGBA8 and RGBA32F — should pick RGBA8 (cheaper from R8)
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
        // RGBA8 should be cheaper than RGBA32F from R8
        assert_eq!(target, ktx2::Format::R8G8B8A8_UNORM);
    }

    #[test]
    fn no_path_for_impossible_constraint() {
        let graph = build_default_graph();
        // sRGB color space constraint — no edges convert color space yet
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
        // Going from RGBA8 to RGBA32F should not route through F16 due to cost penalty
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
        // No direct edge from linear to sRGB
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
}
