#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
mkdir -p "${BIN_DIR}"

ARCH="${ARCH:-sm_120}"
NVCC_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -Xcompiler
  -Wall
  -arch="${ARCH}"
)

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/cosine_backward_bench.cu" \
  -o "${BIN_DIR}/cosine_backward_bench"

"${BIN_DIR}/cosine_backward_bench" "$@"
