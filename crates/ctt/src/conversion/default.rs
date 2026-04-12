use std::sync::Arc;

use crate::alpha::AlphaMode;
use crate::surface::{ColorSpace, Surface};
use crate::vk_format::{ChannelKind, FormatExt};

use super::convert_surface;
use super::graph::{ConversionGraph, ExactEdge, FormatEdge, FormatState, SurfaceConverter};
use super::premultiplication::{premultiply_alpha, unpremultiply_alpha};
use super::srgb::{linear_to_srgb, srgb_to_linear};

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

    // sRGB ↔ linear exact edges.
    //
    // All edges go through F32: (any_fmt, sRGB) → (f32_fmt, Linear) and
    // (f32_fmt, Linear) → (any_fmt, sRGB). This keeps the edge count manageable
    // while supporting all format combinations via the graph.
    let srgb_groups: &[(&[ktx2::Format], ktx2::Format)] = &[
        (
            &[F::R8_UNORM, F::R16_UNORM, F::R16_SFLOAT, F::R32_SFLOAT],
            F::R32_SFLOAT,
        ),
        (
            &[
                F::R8G8_UNORM,
                F::R16G16_UNORM,
                F::R16G16_SFLOAT,
                F::R32G32_SFLOAT,
            ],
            F::R32G32_SFLOAT,
        ),
        (
            &[
                F::R8G8B8_UNORM,
                F::R16G16B16_UNORM,
                F::R16G16B16_SFLOAT,
                F::R32G32B32_SFLOAT,
            ],
            F::R32G32B32_SFLOAT,
        ),
        (
            &[
                F::R8G8B8A8_UNORM,
                F::R16G16B16A16_UNORM,
                F::R16G16B16A16_SFLOAT,
                F::R32G32B32A32_SFLOAT,
            ],
            F::R32G32B32A32_SFLOAT,
        ),
    ];

    for alpha in [
        AlphaMode::Straight,
        AlphaMode::Premultiplied,
        AlphaMode::Opaque,
    ] {
        for (fmts, f32_fmt) in srgb_groups {
            let has_alpha = f32_fmt.channel_count().unwrap_or(0) == 4;
            let f32_fmt = *f32_fmt;

            for &src_fmt in *fmts {
                let cost = conversion_cost(src_fmt, f32_fmt).saturating_sub(5);

                // sRGB src → linear f32
                {
                    let from = FormatState::new(src_fmt, ColorSpace::Srgb, alpha);
                    let to = FormatState::new(f32_fmt, ColorSpace::Linear, alpha);
                    let converter: SurfaceConverter = Arc::new(move |surface: &Surface| {
                        srgb_to_linear(surface, f32_fmt, has_alpha)
                    });
                    graph.add_exact_edge(
                        from,
                        ExactEdge {
                            target: to,
                            cost,
                            converter,
                        },
                    );
                }

                // linear f32 → sRGB src
                {
                    let cost = conversion_cost(f32_fmt, src_fmt).saturating_sub(5);
                    let from = FormatState::new(f32_fmt, ColorSpace::Linear, alpha);
                    let to = FormatState::new(src_fmt, ColorSpace::Srgb, alpha);
                    let converter: SurfaceConverter = Arc::new(move |surface: &Surface| {
                        linear_to_srgb(surface, src_fmt, has_alpha)
                    });
                    graph.add_exact_edge(
                        from,
                        ExactEdge {
                            target: to,
                            cost,
                            converter,
                        },
                    );
                }
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
