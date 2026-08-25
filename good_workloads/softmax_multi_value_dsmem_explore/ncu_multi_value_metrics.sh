#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BIN_DIR}" "${RESULTS_DIR}"

ARCH="${ARCH:-sm_120}"
ROWS="${ROWS:-2048}"
COLS="${COLS:-65536}"
CHANNELS="${CHANNELS:-8}"
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"

NVCC_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -Xcompiler
  -Wall
  -arch="${ARCH}"
)

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/multi_value_bench.cu" \
  -o "${BIN_DIR}/multi_value_bench"

COMMON_ARGS=(
  --csv
  --rows "${ROWS}"
  --n-values "${COLS}"
  --channels-values "${CHANNELS}"
  --warmup 1
  --iters 1
  --cluster-size "${CLUSTER_SIZE}"
  --no-verify
)

METRICS=gpu__time_duration.sum,lts__t_bytes.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum
BLOCK_OUT="${RESULTS_DIR}/ncu_multi_value_d${CHANNELS}_block.csv"
CLUSTER_OUT="${RESULTS_DIR}/ncu_multi_value_d${CHANNELS}_cluster.csv"

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
  --kernel-name regex:.*block_multi_kernel.* \
  "${BIN_DIR}/multi_value_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant block \
  > "${BLOCK_OUT}"
echo "wrote ${BLOCK_OUT}"

ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
  --kernel-name regex:.*cluster_multi_staged_kernel.* \
  "${BIN_DIR}/multi_value_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant cluster \
  > "${CLUSTER_OUT}"
echo "wrote ${CLUSTER_OUT}"
