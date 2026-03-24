#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make >/dev/null

printf 'groups,hidden,out_rows,producer_iters,baseline_ms,fused_local_ms,fused_cluster_ms,cluster_vs_baseline,cluster_vs_local\n'
for out_rows in 1 2 4 8; do
  for piters in 8 16 32 64; do
    out=$(./ffn_push_bench --mode all --groups 512 --hidden 12288 --out-rows "$out_rows" --producer-iters "$piters" --warmup 1 --iters 5 --no-verify)
    baseline=$(printf '%s\n' "$out" | awk -F= '/MODE=baseline/{getline; print $2}')
    local=$(printf '%s\n' "$out" | awk -F= '/MODE=fused_local/{getline; print $2}')
    cluster=$(printf '%s\n' "$out" | awk -F= '/MODE=fused_cluster/{getline; print $2}')
    s1=$(printf '%s\n' "$out" | awk -F= '/SPEEDUP_FUSED_CLUSTER_VS_BASELINE/{print $2}')
    s2=$(printf '%s\n' "$out" | awk -F= '/SPEEDUP_FUSED_CLUSTER_VS_FUSED_LOCAL/{print $2}')
    printf '512,12288,%s,%s,%s,%s,%s,%s,%s\n' "$out_rows" "$piters" "$baseline" "$local" "$cluster" "$s1" "$s2"
  done
done
