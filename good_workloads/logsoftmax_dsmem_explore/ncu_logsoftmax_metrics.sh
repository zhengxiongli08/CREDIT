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

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/logsoftmax_bench.cu" \
  -o "${BIN_DIR}/logsoftmax_bench"

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

ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
  --kernel-name regex:.*block_logsoftmax_kernel.* \
  "${BIN_DIR}/logsoftmax_bench" \
  "${COMMON_ARGS[@]}" \
  --thread-per-row-values 16 \
  > "${RESULTS_DIR}/ncu_logsoftmax_block.csv"
echo "wrote ${RESULTS_DIR}/ncu_logsoftmax_block.csv"

ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
  --kernel-name regex:.*cluster_logsoftmax_staged_kernel.* \
  "${BIN_DIR}/logsoftmax_bench" \
  "${COMMON_ARGS[@]}" \
  --thread-per-row-values 256 \
  > "${RESULTS_DIR}/ncu_logsoftmax_cluster.csv"
echo "wrote ${RESULTS_DIR}/ncu_logsoftmax_cluster.csv"
