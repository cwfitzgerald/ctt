use crate::conversion_graph::{ConversionGraph, FormatState, build_default_graph, check_lossless};
use crate::error::{Error, Result};
use crate::surface::Image;
use crate::transforms::Transform;
use crate::transforms::format_convert::FormatConvertTransform;

/// An input source for the pipeline.
pub enum InputNode {
    /// A pre-decoded image (from the CLI or library consumer).
    Raw(Image),
}

/// How to assemble multiple inputs into a single image.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AssemblyNode {
    /// Single input passthrough (or first input if multiple).
    Identity,
    /// Combine N single-layer inputs into a 6-face cubemap.
    Cubemap,
    /// Combine N single-layer inputs into an array texture.
    Array,
}

/// Output format.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputNode {
    Dds,
    Ktx2,
    /// Return the Image directly (for library consumers / JIT compression).
    Raw,
}

/// A single input branch with its own pre-assembly transform chain.
pub struct InputBranch {
    pub input: InputNode,
    pub transforms: Vec<Box<dyn Transform>>,
}

/// The user-facing pipeline definition.
pub struct Pipeline {
    pub inputs: Vec<InputBranch>,
    pub assembly: AssemblyNode,
    pub transforms: Vec<Box<dyn Transform>>,
    pub output: OutputNode,
    /// If `false` (the default), auto-inserted format conversions that lose
    /// precision will produce errors during [`Pipeline::resolve`]. Set to `true`
    /// to suppress these errors.
    pub allow_lossy_intermediates: bool,
}

/// A fully resolved pipeline ready for execution.
///
/// All format conversions have been inserted and validated.
pub struct ResolvedPipeline {
    inputs: Vec<ResolvedBranch>,
    assembly: AssemblyNode,
    transforms: Vec<Box<dyn Transform>>,
    output: OutputNode,
}

struct ResolvedBranch {
    input: InputNode,
    transforms: Vec<Box<dyn Transform>>,
}

/// The output of pipeline execution.
pub enum PipelineOutput {
    /// Encoded file bytes (DDS or KTX2).
    Encoded(Vec<u8>),
    /// Raw image (for OutputNode::Raw).
    Raw(Image),
}

impl Pipeline {
    /// Resolve formats, insert conversions, and validate the pipeline.
    ///
    /// Returns all errors at once rather than failing on the first.
    pub fn resolve(self) -> std::result::Result<ResolvedPipeline, Vec<Error>> {
        let graph = build_default_graph();
        self.resolve_with_graph(&graph)
    }

    /// Resolve with a custom conversion graph (for testing or extended conversions).
    pub fn resolve_with_graph(
        self,
        graph: &ConversionGraph,
    ) -> std::result::Result<ResolvedPipeline, Vec<Error>> {
        profiling::scope!("Pipeline::resolve");

        let allow_lossy = self.allow_lossy_intermediates;
        let mut errors = Vec::new();

        // Resolve each input branch.
        let mut resolved_inputs = Vec::with_capacity(self.inputs.len());
        for (branch_idx, branch) in self.inputs.into_iter().enumerate() {
            match resolve_branch(branch, graph, &format!("input[{branch_idx}]"), allow_lossy) {
                Ok(resolved) => resolved_inputs.push(resolved),
                Err(mut errs) => errors.append(&mut errs),
            }
        }

        // Determine the format state after assembly.
        let post_assembly_state = if resolved_inputs.is_empty() {
            if errors.is_empty() {
                errors.push(Error::UnsupportedFormat("pipeline has no inputs".into()));
            }
            return Err(errors);
        } else {
            let first_state = branch_output_state(&resolved_inputs[0]);
            // Validate all branches produce the same format state.
            for (i, branch) in resolved_inputs.iter().enumerate().skip(1) {
                let state = branch_output_state(branch);
                if state != first_state {
                    errors.push(Error::UnsupportedFormat(format!(
                        "input[{i}] produces {state} but input[0] produces {first_state}; \
                         all branches must produce the same format for assembly",
                    )));
                }
            }
            first_state
        };

        // Resolve post-assembly transforms.
        let resolved_transforms = match resolve_transform_chain(
            self.transforms,
            post_assembly_state,
            graph,
            "post-assembly",
            allow_lossy,
        ) {
            Ok((transforms, _final_state)) => transforms,
            Err(mut errs) => {
                errors.append(&mut errs);
                Vec::new()
            }
        };

        if !errors.is_empty() {
            return Err(errors);
        }

        Ok(ResolvedPipeline {
            inputs: resolved_inputs,
            assembly: self.assembly,
            transforms: resolved_transforms,
            output: self.output,
        })
    }
}

impl ResolvedPipeline {
    /// Execute the resolved pipeline.
    pub fn execute(self) -> Result<PipelineOutput> {
        profiling::scope!("ResolvedPipeline::execute");

        // Execute each input branch.
        let mut images: Vec<Image> = Vec::with_capacity(self.inputs.len());
        for branch in self.inputs {
            let image = execute_branch(branch)?;
            images.push(image);
        }

        // Assembly.
        let mut image = match self.assembly {
            AssemblyNode::Identity => images.into_iter().next().expect("validated during resolve"),
            AssemblyNode::Cubemap => {
                if images.len() != 6 {
                    return Err(Error::CubemapFaceCount(images.len()));
                }
                let mut surfaces = Vec::with_capacity(6);
                for img in images {
                    if img.surfaces.len() != 1 {
                        return Err(Error::UnsupportedFormat(
                            "cubemap assembly requires single-layer inputs".into(),
                        ));
                    }
                    surfaces.extend(img.surfaces);
                }
                Image {
                    surfaces,
                    is_cubemap: true,
                }
            }
            AssemblyNode::Array => {
                let mut surfaces = Vec::new();
                for img in images {
                    surfaces.extend(img.surfaces);
                }
                Image {
                    surfaces,
                    is_cubemap: false,
                }
            }
        };

        // Execute post-assembly transforms.
        for transform in &self.transforms {
            profiling::scope!("transform", transform.name());
            image = transform.execute(image)?;
        }

        // Output.
        match self.output {
            OutputNode::Dds => {
                profiling::scope!("encode_dds");
                let bytes = crate::output::dds::encode_dds_image(&image)?;
                Ok(PipelineOutput::Encoded(bytes))
            }
            OutputNode::Ktx2 => {
                profiling::scope!("encode_ktx2");
                let bytes = crate::output::ktx2::encode_ktx2_image(&image)?;
                Ok(PipelineOutput::Encoded(bytes))
            }
            OutputNode::Raw => Ok(PipelineOutput::Raw(image)),
        }
    }
}

/// Resolve a single input branch, inserting format conversions as needed.
fn resolve_branch(
    branch: InputBranch,
    graph: &ConversionGraph,
    label: &str,
    allow_lossy: bool,
) -> std::result::Result<ResolvedBranch, Vec<Error>> {
    profiling::scope!("resolve input branch", label);
    let input_state = input_format_state(&branch.input);

    let (transforms, _final_state) =
        resolve_transform_chain(branch.transforms, input_state, graph, label, allow_lossy)?;

    Ok(ResolvedBranch {
        input: branch.input,
        transforms,
    })
}

/// Resolve a chain of transforms, inserting format conversions between steps.
///
/// Takes ownership of the transforms and returns a new chain with conversion steps
/// interleaved as needed. Returns the resolved transform list and the final format state.
type ResolveResult = std::result::Result<(Vec<Box<dyn Transform>>, FormatState), Vec<Error>>;

fn resolve_transform_chain(
    transforms: Vec<Box<dyn Transform>>,
    initial_state: FormatState,
    graph: &ConversionGraph,
    label: &str,
    allow_lossy: bool,
) -> ResolveResult {
    profiling::scope!("resolve transform chain", label);
    let mut errors = Vec::new();
    let mut resolved: Vec<Box<dyn Transform>> = Vec::new();
    let mut current_state = initial_state;

    for transform in transforms {
        let constraint = transform.constraint();

        if !current_state.satisfies(&constraint) {
            // Try to find a conversion path.
            match graph.find_path_to_constraint(current_state, &constraint) {
                Some(path) => {
                    // Insert conversion transforms for each hop.
                    let mut hop_from = current_state;
                    for (hop_idx, hop_to) in path.iter().enumerate() {
                        log::debug!(
                            "{label}: auto-convert {hop_from} -> {hop_to} for '{}'",
                            transform.name(),
                        );
                        let converter = graph.get_converter(hop_from, *hop_to).cloned();
                        match converter {
                            Some(conv) => {
                                // Check for spurious precision loss in intermediate
                                // hops. The final hop is the intentional target, so
                                // it is exempt — the user chose that precision.
                                let is_last = hop_idx == path.len() - 1;
                                if !allow_lossy && !is_last {
                                    if let Err(reason) = check_lossless(initial_state, *hop_to) {
                                        errors.push(Error::LossyConversion {
                                            from: initial_state.to_string(),
                                            to: hop_to.to_string(),
                                            transform: transform.name().to_string(),
                                            reason: reason.to_string(),
                                        });
                                    }
                                }
                                resolved.push(Box::new(FormatConvertTransform::new(
                                    hop_to.format,
                                    hop_to.color_space,
                                    hop_to.alpha,
                                    conv,
                                )));
                            }
                            None => {
                                errors.push(Error::UnsupportedConversion(format!(
                                    "{label}: no converter for {hop_from} -> {hop_to}",
                                )));
                            }
                        }
                        hop_from = *hop_to;
                    }
                    current_state = hop_from;
                }
                None => {
                    errors.push(Error::UnsupportedConversion(format!(
                        "{label}: no conversion path from {current_state} to satisfy constraint of '{}'",
                        transform.name()
                    )));
                }
            }
        }

        // Track format through this transform.
        let (fmt, cs, alpha) = transform.output_format(
            current_state.format,
            current_state.color_space,
            current_state.alpha,
        );
        current_state = FormatState::new(fmt, cs, alpha);
        log::debug!("{label}: '{}' -> {current_state}", transform.name());

        // Add the original transform after any conversions.
        resolved.push(transform);
    }

    if !errors.is_empty() {
        return Err(errors);
    }

    log::debug!("{label}: resolved {n} transform(s)", n = resolved.len());

    Ok((resolved, current_state))
}

/// Get the format state of an input node.
fn input_format_state(input: &InputNode) -> FormatState {
    match input {
        InputNode::Raw(image) => {
            let first = &image.surfaces[0][0];
            FormatState::new(first.format, first.color_space, first.alpha)
        }
    }
}

/// Get the output format state of a resolved branch.
fn branch_output_state(branch: &ResolvedBranch) -> FormatState {
    profiling::scope!("branch_output_state");
    let mut state = input_format_state(&branch.input);
    for transform in &branch.transforms {
        let (fmt, cs, alpha) =
            transform.output_format(state.format, state.color_space, state.alpha);
        state = FormatState::new(fmt, cs, alpha);
    }
    state
}

/// Execute a resolved branch.
fn execute_branch(branch: ResolvedBranch) -> Result<Image> {
    profiling::scope!("input_branch");
    let InputNode::Raw(mut image) = branch.input;

    for transform in &branch.transforms {
        profiling::scope!("transform", transform.name());
        image = transform.execute(image)?;
    }

    Ok(image)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::alpha::AlphaMode;
    use crate::surface::ColorSpace;

    fn make_test_image(format: ktx2::Format, cs: ColorSpace) -> Image {
        use crate::surface::Surface;
        use crate::vk_format::FormatExt as _;
        // 4x4 pixel image
        let bpp = format.bytes_per_pixel().unwrap_or(4);
        let stride = 4 * bpp as u32;
        Image {
            surfaces: vec![vec![Surface {
                data: vec![128u8; (stride * 4) as usize],
                width: 4,
                height: 4,
                stride,
                format,
                color_space: cs,
                alpha: AlphaMode::Straight,
            }]],
            is_cubemap: false,
        }
    }

    #[test]
    fn passthrough_no_transforms() {
        let image = make_test_image(ktx2::Format::R8G8B8A8_UNORM, ColorSpace::Linear);
        let pipeline = Pipeline {
            inputs: vec![InputBranch {
                input: InputNode::Raw(image),
                transforms: Vec::new(),
            }],
            assembly: AssemblyNode::Identity,
            transforms: Vec::new(),
            output: OutputNode::Raw,
            allow_lossy_intermediates: false,
        };

        let resolved = pipeline.resolve().unwrap();
        let output = resolved.execute().unwrap();
        match output {
            PipelineOutput::Raw(img) => {
                assert_eq!(img.surfaces.len(), 1);
                assert_eq!(img.surfaces[0][0].format, ktx2::Format::R8G8B8A8_UNORM);
            }
            _ => panic!("expected Raw output"),
        }
    }

    #[test]
    fn empty_pipeline_errors() {
        let pipeline = Pipeline {
            inputs: Vec::new(),
            assembly: AssemblyNode::Identity,
            transforms: Vec::new(),
            output: OutputNode::Raw,
            allow_lossy_intermediates: false,
        };

        let result = pipeline.resolve();
        assert!(result.is_err());
    }
}
