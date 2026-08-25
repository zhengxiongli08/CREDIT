#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BIN_DIR}" "${RESULTS_DIR}"

ARCH="${ARCH:-sm_120}"
ROWS="${ROWS:-4096}"
COLS="${COLS:-32768}"
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"

NVCC_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -Xcompiler
  -Wall
  -arch="${ARCH}"
)

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/lamb_trust_bench.cu" \
  -o "${BIN_DIR}/lamb_trust_bench"

COMMON_ARGS=(
  --csv
  --rows "${ROWS}"
  --n-values "${COLS}"
  --warmup 1
  --iters 1
  --cluster-size "${CLUSTER_SIZE}"
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
  --kernel-name regex:.*block_lamb_read_kernel.* \
  "${BIN_DIR}/lamb_trust_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant block \
  > "${RESULTS_DIR}/ncu_lamb_trust_block.csv"
echo "wrote ${RESULTS_DIR}/ncu_lamb_trust_block.csv"

ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
  --kernel-name regex:.*cluster_lamb_staged_kernel.* \
  "${BIN_DIR}/lamb_trust_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant cluster \
  > "${RESULTS_DIR}/ncu_lamb_trust_cluster.csv"
echo "wrote ${RESULTS_DIR}/ncu_lamb_trust_cluster.csv"
