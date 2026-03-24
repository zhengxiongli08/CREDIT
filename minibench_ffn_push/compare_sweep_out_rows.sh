#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make >/dev/null
py=/home/zli2793/tools/miniforge3/envs/tile/bin/python

groups=512
hidden=12288
producer_iters=32
warmup=5
iters=20

printf 'groups,hidden,out_rows,producer_iters,cuda_cluster_ms,pytorch_ms,tensorrt_ms,pytorch_rel_to_cuda,tensorrt_rel_to_cuda\n'
for out_rows in 8 16 32 64; do
  cuda_out=$(./ffn_push_bench --mode fused_cluster --groups "$groups" --hidden "$hidden" --out-rows "$out_rows" --producer-iters "$producer_iters" --warmup "$warmup" --iters "$iters" --no-verify)
  pt_out=$($py ./ffn_push_pytorch_compile.py --groups "$groups" --hidden "$hidden" --out-rows "$out_rows" --producer-iters "$producer_iters" --warmup "$warmup" --iters "$iters")
  trt_out=$($py ./ffn_push_tensorrt.py --groups "$groups" --hidden "$hidden" --out-rows "$out_rows" --producer-iters "$producer_iters" --warmup "$warmup" --iters "$iters")

  cuda_ms=$(printf '%s\n' "$cuda_out" | awk -F= '/AVG_MS/{print $2}')
  pt_ms=$(printf '%s\n' "$pt_out" | awk -F= '/AVG_MS/{print $2}')
  trt_ms=$(printf '%s\n' "$trt_out" | awk -F= '/AVG_MS/{print $2}')

  printf '%s,%s,%s,%s,%s,%s,%s,' "$groups" "$hidden" "$out_rows" "$producer_iters" "$cuda_ms" "$pt_ms" "$trt_ms"
  awk -v pt="$pt_ms" -v cu="$cuda_ms" 'BEGIN { printf "%.3f,", pt / cu }'
  awk -v trt="$trt_ms" -v cu="$cuda_ms" 'BEGIN { printf "%.3f\n", trt / cu }'
done
