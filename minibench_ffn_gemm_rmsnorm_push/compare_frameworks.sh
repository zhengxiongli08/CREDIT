#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make >/dev/null

PYTHON_BIN='/home/zli2793/tools/miniforge3/envs/tile/bin/python'

args=(--groups 128 --input-k 128 --hidden 12288 --out-rows 8 --warmup 10 --iters 50)

cuda_out=$(./ffn_gemm_rmsnorm_push_bench --mode fused_cluster "${args[@]}")
pytorch_out=$($PYTHON_BIN ./ffn_pytorch_compile.py "${args[@]}")
trt_out=$($PYTHON_BIN ./ffn_tensorrt.py "${args[@]}")

cuda_ms=$(printf '%s\n' "$cuda_out" | awk -F= '/MODE=fused_cluster/{getline; print $2}')
pytorch_ms=$(printf '%s\n' "$pytorch_out" | awk -F= '/AVG_MS/{print $2}')
trt_ms=$(printf '%s\n' "$trt_out" | awk -F= '/AVG_MS/{print $2}')

printf 'IMPLEMENTATION,AVG_MS,RUNTIME_RELATIVE_TO_CUDA_CLUSTER\n'
printf 'cuda_cluster,%s,1.000\n' "$cuda_ms"
printf 'pytorch_compile,%s,' "$pytorch_ms"
awk -v a="$pytorch_ms" -v b="$cuda_ms" 'BEGIN { printf "%.3f\n", a / b }'
printf 'tensorrt,%s,' "$trt_ms"
awk -v a="$trt_ms" -v b="$cuda_ms" 'BEGIN { printf "%.3f\n", a / b }'
