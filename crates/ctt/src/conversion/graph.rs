use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};
use std::fmt;
use std::sync::Arc;

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
        // A goal state must satisfy the constraint AND preserve any properties
        // that the constraint leaves unconstrained (color space, alpha mode).
        let is_goal = |state: &FormatState| -> bool {
            if !state.satisfies(constraint) {
                return false;
            }
            if constraint.color_spaces.is_none() && state.color_space != from.color_space {
                return false;
            }
            if constraint.alpha_modes.is_none() && state.alpha != from.alpha {
                return false;
            }
            true
        };

        if is_goal(&from) {
            return Some(Vec::new());
        }

        let mut dist: HashMap<FormatState, u32> = HashMap::new();
        let mut prev: HashMap<FormatState, FormatState> = HashMap::new();
        let mut heap: BinaryHeap<Reverse<(u32, FormatState)>> = BinaryHeap::new();

        dist.insert(from, 0);
        heap.push(Reverse((0, from)));

        while let Some(Reverse((cost, state))) = heap.pop() {
            if state != from && is_goal(&state) {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversion::build_default_graph;

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
