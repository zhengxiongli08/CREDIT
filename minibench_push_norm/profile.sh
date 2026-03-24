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

common_args=(--groups 512 --tile-k 12288 --rows 4 --producer-iters 32 --warmup 3 --iters 10 --no-verify)
./push_norm_bench --mode all "${common_args[@]}" | tee "$out_dir/runtime_summary.txt"
for mode in baseline fused_local fused_cluster; do
  ncu -f -o "$out_dir/$mode" ./push_norm_bench --mode "$mode" "${common_args[@]}" > "$out_dir/${mode}_stdout.txt" 2>&1
  echo "Saved $out_dir/$mode.ncu-rep"
done
