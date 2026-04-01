#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/../../../astc-encoder/Source}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: source directory not found: $SRC_DIR" >&2
    exit 1
fi

DST_DIR="$SCRIPT_DIR/cpp"
rm -rf "$DST_DIR"
mkdir "$DST_DIR"

# Core library source files (22 files, NO CLI files)
for f in \
    astcenc_averages_and_directions.cpp \
    astcenc_block_sizes.cpp \
    astcenc_color_quantize.cpp \
    astcenc_color_unquantize.cpp \
    astcenc_compress_symbolic.cpp \
    astcenc_compute_variance.cpp \
    astcenc_decompress_symbolic.cpp \
    astcenc_diagnostic_trace.cpp \
    astcenc_entry.cpp \
    astcenc_find_best_partitioning.cpp \
    astcenc_ideal_endpoints_and_weights.cpp \
    astcenc_image.cpp \
    astcenc_integer_sequence.cpp \
    astcenc_mathlib.cpp \
    astcenc_mathlib_softfloat.cpp \
    astcenc_partition_tables.cpp \
    astcenc_percentile_tables.cpp \
    astcenc_pick_best_endpoint_format.cpp \
    astcenc_quantization.cpp \
    astcenc_symbolic_physical.cpp \
    astcenc_weight_align.cpp \
    astcenc_weight_quant_xfer_tables.cpp
do
    cp "$SRC_DIR/$f" "$DST_DIR/$f"
done

# Public API header
cp "$SRC_DIR/astcenc.h" "$DST_DIR/astcenc.h"

# Internal headers
cp "$SRC_DIR/astcenc_internal.h" "$DST_DIR/astcenc_internal.h"
cp "$SRC_DIR/astcenc_internal_entry.h" "$DST_DIR/astcenc_internal_entry.h"
cp "$SRC_DIR/astcenc_mathlib.h" "$DST_DIR/astcenc_mathlib.h"
cp "$SRC_DIR/astcenc_diagnostic_trace.h" "$DST_DIR/astcenc_diagnostic_trace.h"

# SIMD vector math headers
cp "$SRC_DIR/astcenc_vecmathlib.h" "$DST_DIR/astcenc_vecmathlib.h"
cp "$SRC_DIR/astcenc_vecmathlib_sse_4.h" "$DST_DIR/astcenc_vecmathlib_sse_4.h"
cp "$SRC_DIR/astcenc_vecmathlib_avx2_8.h" "$DST_DIR/astcenc_vecmathlib_avx2_8.h"
cp "$SRC_DIR/astcenc_vecmathlib_neon_4.h" "$DST_DIR/astcenc_vecmathlib_neon_4.h"
cp "$SRC_DIR/astcenc_vecmathlib_sve_8.h" "$DST_DIR/astcenc_vecmathlib_sve_8.h"
cp "$SRC_DIR/astcenc_vecmathlib_none_4.h" "$DST_DIR/astcenc_vecmathlib_none_4.h"
cp "$SRC_DIR/astcenc_vecmathlib_common_4.h" "$DST_DIR/astcenc_vecmathlib_common_4.h"

echo "Vendored astcenc from $SRC_DIR into $DST_DIR"
