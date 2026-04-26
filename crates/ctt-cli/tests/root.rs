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
mod edge_cases;
mod encoder_select;
mod errors;
mod mipmap;
mod passthrough;
mod supercompression;
mod swizzle;
