//! End-to-end tests for the ctt CLI.
//!
//! All test files are submodules of this single test root so the shared
//! `common` helpers compile once. See `plans/e2e-tests.md`.

mod common;

mod compression;
mod containers;
mod cubemap;
mod encoder_select;
mod errors;
mod mipmap;
mod passthrough;
