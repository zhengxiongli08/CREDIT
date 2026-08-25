#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BIN_DIR}" "${RESULTS_DIR}"

ARCH="${ARCH:-sm_120}"
NVCC_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -Xcompiler
  -Wall
  -arch="${ARCH}"
)

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/markdown_workload_bench.cu" \
  -o "${BIN_DIR}/markdown_workload_bench"

COMMON_ARGS=(
  --csv
  --batch-size 8192
  --n-values 65536
  --warmup 1
  --iters 1
  --cluster-size 8
  --no-verify
)

METRICS=gpu__time_duration.sum,lts__t_bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum

NCU_COMMON=(
  --target-processes all
  --csv
  --page raw
  --print-units base
  --cache-control none
  --clock-control none
  --print-kernel-base function
  --metrics "${METRICS}"
)

run_case() {
  local workload="$1"
  local variant="$2"
  local tpr="$3"
  local kernel_regex="$4"
  local out="${RESULTS_DIR}/ncu_${workload}_${variant}.csv"

  ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
    --kernel-name "regex:${kernel_regex}" \
    "${BIN_DIR}/markdown_workload_bench" \
    "${COMMON_ARGS[@]}" \
    --workload "${workload}" \
    --thread-per-row-values "${tpr}" \
    > "${out}"
  echo "wrote ${out}"
}

run_case softmax block 16 ".*block_softmax_kernel.*"
run_case softmax cluster 256 ".*cluster_softmax_staged_kernel.*"
run_case rmsnorm block 16 ".*block_rmsnorm_kernel.*"
run_case rmsnorm cluster 256 ".*cluster_rmsnorm_staged_kernel.*"
run_case cross_entropy block 16 ".*block_cross_entropy_kernel.*"
run_case cross_entropy cluster 256 ".*cluster_cross_entropy_staged_kernel.*"
