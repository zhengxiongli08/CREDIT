#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

conda run -n cluster --no-capture-output \
  python3 "${SCRIPT_DIR}/compare_rmsnorm_gated_full_backward_torch.py" "$@"
