#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/../../../compressonator}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: source directory not found: $SRC_DIR" >&2
    exit 1
fi

DST_DIR="$SCRIPT_DIR/cpp"
rm -rf "$DST_DIR"
mkdir -p "$DST_DIR/source" "$DST_DIR/shaders" "$DST_DIR/cmp_math"

# cmp_core/source/ — core API, SIMD variants, and math headers
for f in \
    cmp_core.h \
    cmp_core.cpp \
    core_simd.h \
    core_simd_sse.cpp \
    core_simd_avx.cpp \
    core_simd_avx512.cpp \
    cmp_math_vec4.h \
    cmp_math_func.h
do
    cp "$SRC_DIR/cmp_core/source/$f" "$DST_DIR/source/$f"
done

# cmp_core/shaders/ — encoder kernels and shared headers
for f in \
    bc1_encode_kernel.cpp bc1_encode_kernel.h \
    bc2_encode_kernel.cpp bc2_encode_kernel.h \
    bc3_encode_kernel.cpp bc3_encode_kernel.h \
    bc4_encode_kernel.cpp bc4_encode_kernel.h \
    bc5_encode_kernel.cpp bc5_encode_kernel.h \
    bc6_encode_kernel.cpp bc6_encode_kernel.h \
    bc7_encode_kernel.cpp bc7_encode_kernel.h \
    common_def.h \
    bcn_common_kernel.h \
    bcn_common_api.h \
    bc1_cmp.h \
    bc1_common_kernel.h \
    bc6_common_encoder.h \
    bc7_common_encoder.h \
    bc7_cmpmsc.h
do
    cp "$SRC_DIR/cmp_core/shaders/$f" "$DST_DIR/shaders/$f"
done

# applications/_libs/cmp_math/ — CPU feature detection and math utilities
for f in \
    cpu_extensions.cpp cpu_extensions.h \
    cmp_math_common.cpp cmp_math_common.h
do
    cp "$SRC_DIR/applications/_libs/cmp_math/$f" "$DST_DIR/cmp_math/$f"
done

# License
cp "$SRC_DIR/license/corelicense.txt" "$SCRIPT_DIR/LICENSE-MIT-COMPRESSONATOR.md"

# ============================================================================
# Patches — apply after copying so re-vendoring is trivial.
# ============================================================================

# Use python for complex multi-line patches, sed for simple ones.

# --- cmp_core.h: extern "C" wrapping + default-arg fix ---

sed -i \
    -e '/^\/\/ API Definitions/i\
#ifdef __cplusplus\
extern "C" {\
#endif' \
    -e '/^#endif  \/\/ CMP_CORE/i\
#ifdef __cplusplus\
}\
#endif' \
    "$DST_DIR/source/cmp_core.h"

sed -i 's/const void\* options = NULL)/const void* options CMP_DEFAULTNULL)/' \
    "$DST_DIR/source/cmp_core.h"

# --- All .cpp files: extern "C" on public API functions ---

for f in "$DST_DIR"/source/*.cpp "$DST_DIR"/shaders/*.cpp; do
    sed -i 's/^int CMP_CDECL /extern "C" int CMP_CDECL /g' "$f"
    sed -i 's/^void CMP_CDECL /extern "C" void CMP_CDECL /g' "$f"
done

# --- cpu_extensions.cpp: cross-platform CPUID ---
# --- core_simd.h: x86-only SIMD declarations ---
# --- bc1_cmp.h: x86-only SIMD dispatch ---

DST_DIR_WIN="$(cygpath -w "$DST_DIR" 2>/dev/null || echo "$DST_DIR")"

python3 -c "
import re, sys, os

base = r'$DST_DIR_WIN'.replace('\\\\', '/')

# --- cpu_extensions.cpp ---
f = base + '/cmp_math/cpu_extensions.cpp'
t = open(f).read()

# Fix include: use cpuid.h on GCC/Clang x86, intrin.h on MSVC x86
t = t.replace(
    '#ifdef _WIN32\n#include <intrin.h>\n#endif',
    '#if defined(_M_X64) || defined(_M_IX86)\n#include <intrin.h>\n#elif defined(__x86_64__) || defined(__i386__)\n#include <cpuid.h>\n#endif'
)

# Fix GetCPUID: cross-platform implementation
t = t.replace(
    '''#ifdef _WIN32
    __cpuidex(outInfo, functionID, 0);  // defined in intrin.h
#else
    // To Do
    //__cpuid_count(0, function_id, outInfo[0], outInfo[1], outInfo[2], outInfo[3]);
#endif''',
    '''#if defined(_M_X64) || defined(_M_IX86)
    __cpuidex(outInfo, functionID, 0);
#elif defined(__x86_64__) || defined(__i386__)
    __cpuid_count(functionID, 0, outInfo[0], outInfo[1], outInfo[2], outInfo[3]);
#else
    (void)functionID;
    outInfo[0] = outInfo[1] = outInfo[2] = outInfo[3] = 0;
#endif'''
)

# Fix GetCPUExtensions: enable on Linux x86 too (was #ifndef __linux__)
t = t.replace(
    '#ifndef __linux__',
    '#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)'
)

open(f, 'w', newline='\n').write(t)

# --- core_simd.h ---
f = base + '/source/core_simd.h'
t = open(f).read()

t = t.replace(
    'float sse_bc1ComputeBestEndpoints',
    '#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)\nfloat sse_bc1ComputeBestEndpoints'
)
t = t.replace(
    'float avx512_bc1ComputeBestEndpoints(float*, float*, float*, float*, float*, int, int);',
    'float avx512_bc1ComputeBestEndpoints(float*, float*, float*, float*, float*, int, int);\n#endif // x86'
)

open(f, 'w', newline='\n').write(t)

# --- bc1_cmp.h ---
f = base + '/shaders/bc1_cmp.h'
t = open(f).read()

# Guard the SIMD function pointer assignments in bc1ToggleSIMD.
# On non-x86 the SIMD symbols don't exist, so always use scalar.
t = t.replace(
    '    if (useAVX512 && IsAvailableAVX512(extensions))\n    {\n        cpu_bc1ComputeBestEndpoints = avx512_bc1ComputeBestEndpoints;',
    '#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)\n    if (useAVX512 && IsAvailableAVX512(extensions))\n    {\n        cpu_bc1ComputeBestEndpoints = avx512_bc1ComputeBestEndpoints;'
)
t = t.replace(
    '        cpu_bc1ComputeBestEndpoints = sse_bc1ComputeBestEndpoints;\n    }\n    else\n    {\n        cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n    }',
    '        cpu_bc1ComputeBestEndpoints = sse_bc1ComputeBestEndpoints;\n    }\n    else\n    {\n        cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n    }\n#else\n    cpu_bc1ComputeBestEndpoints = _cpu_bc1ComputeBestEndpoints;\n#endif // x86'
)

open(f, 'w', newline='\n').write(t)
"

echo "Vendored compressonator CMP_Core from $SRC_DIR into $DST_DIR"
