#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="${1:-$SCRIPT_DIR/../../../bc7enc_rdo}"

if [[ ! -d "$SRC_DIR" ]]; then
    echo "error: source directory not found: $SRC_DIR" >&2
    exit 1
fi

DST_DIR="$SCRIPT_DIR/ispc"
rm -rf "$DST_DIR"
mkdir "$DST_DIR"

cp "$SRC_DIR/bc7e.ispc" "$DST_DIR/bc7e.ispc"
