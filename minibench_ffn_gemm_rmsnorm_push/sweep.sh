#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make >/dev/null

printf 'groups,input_k,hidden,out_rows,baseline_ms,fused_local_ms,fused_cluster_ms,cluster_vs_baseline,cluster_vs_local\n'
for input_k in 32 64 128 256; do
  for out_rows in 1 2 4 8; do
    out=$(./ffn_gemm_rmsnorm_push_bench --mode all --groups 256 --input-k "$input_k" --hidden 12288 --out-rows "$out_rows" --warmup 1 --iters 5 --no-verify)
    baseline=$(printf '%s\n' "$out" | awk -F= '/MODE=baseline/{getline; print $2}')
    local=$(printf '%s\n' "$out" | awk -F= '/MODE=fused_local/{getline; print $2}')
    cluster=$(printf '%s\n' "$out" | awk -F= '/MODE=fused_cluster/{getline; print $2}')
    s1=$(printf '%s\n' "$out" | awk -F= '/SPEEDUP_FUSED_CLUSTER_VS_BASELINE/{print $2}')
    s2=$(printf '%s\n' "$out" | awk -F= '/SPEEDUP_FUSED_CLUSTER_VS_FUSED_LOCAL/{print $2}')
    printf '256,%s,12288,%s,%s,%s,%s,%s,%s\n' "$input_k" "$out_rows" "$baseline" "$local" "$cluster" "$s1" "$s2"
  done
done
