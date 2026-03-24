#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
cd "$script_dir"
make >/dev/null
py=/home/zli2793/tools/miniforge3/envs/tile/bin/python
args=(--groups 512 --hidden 12288 --out-rows 8 --producer-iters 32 --warmup 10 --iters 50)
if [[ $# -gt 0 ]]; then
	args=("$@")
fi

cuda_out=$(./ffn_push_bench --mode fused_cluster "${args[@]}" --no-verify)
pt_out=$($py ./ffn_push_pytorch_compile.py "${args[@]}")
trt_out=$($py ./ffn_push_tensorrt.py "${args[@]}")

cuda_ms=$(printf '%s\n' "$cuda_out" | awk -F= '/MODE=fused_cluster/{getline; print $2}')
pt_ms=$(printf '%s\n' "$pt_out" | awk -F= '/AVG_MS/{print $2}')
trt_ms=$(printf '%s\n' "$trt_out" | awk -F= '/AVG_MS/{print $2}')

printf 'IMPLEMENTATION,AVG_MS,RUNTIME_RELATIVE_TO_CUDA_CLUSTER\n'
printf 'cuda_cluster,%s,1.000\n' "$cuda_ms"
printf 'pytorch_compile,%s,' "$pt_ms"
awk -v pt="$pt_ms" -v cu="$cuda_ms" 'BEGIN { printf "%.3f\n", pt / cu }'
printf 'tensorrt,%s,' "$trt_ms"
awk -v trt="$trt_ms" -v cu="$cuda_ms" 'BEGIN { printf "%.3f\n", trt / cu }'
