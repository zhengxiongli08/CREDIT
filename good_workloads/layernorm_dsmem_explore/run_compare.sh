#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TORCH_ENV="${TORCH_ENV:-cluster}"

conda run --no-capture-output -n "${TORCH_ENV}" \
  python "${SCRIPT_DIR}/compare_layernorm_torch.py" "$@"
