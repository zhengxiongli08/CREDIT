#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make
./ffn_gemm_rmsnorm_push_bench --mode all "$@"
