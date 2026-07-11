//! End-to-end tests for the ctt CLI.
//!
//! All test files are submodules of this single test root so the shared
//! `common` helpers compile once.

mod common;

mod arrays;
mod color_alpha;
mod compression;
mod containers;
mod cubemap;
mod cubemap_array;
mod edge_cases;
mod encode_matrix;
mod encoder_opts;
mod encoder_select;
mod errors;
mod image_inputs;
mod mipmap;
mod passthrough;
mod supercompression;
mod swizzle;
mod volume;
