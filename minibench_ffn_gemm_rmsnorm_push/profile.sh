#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"

if ! command -v ncu >/dev/null 2>&1; then
  echo "ncu not found" >&2
  exit 1
fi

make
out_dir="$script_dir/results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$out_dir"
common_args=(--groups 256 --input-k 128 --hidden 12288 --out-rows 8 --warmup 3 --iters 10 --no-verify)
./ffn_gemm_rmsnorm_push_bench --mode all "${common_args[@]}" | tee "$out_dir/runtime_summary.txt"
for mode in baseline fused_local fused_cluster; do
  ncu -f -o "$out_dir/$mode" ./ffn_gemm_rmsnorm_push_bench --mode "$mode" "${common_args[@]}" > "$out_dir/${mode}_stdout.txt" 2>&1
  echo "Saved $out_dir/$mode.ncu-rep"
done
