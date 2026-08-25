#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BIN_DIR="${SCRIPT_DIR}/bin"
RESULTS_DIR="${SCRIPT_DIR}/results"
mkdir -p "${BIN_DIR}" "${RESULTS_DIR}"

ARCH="${ARCH:-sm_120}"
BATCH_SIZE="${BATCH_SIZE:-32}"
CHANNELS="${CHANNELS:-256}"
ELEMS_PER_CHANNEL="${ELEMS_PER_CHANNEL:-65536}"
CLUSTER_SIZE="${CLUSTER_SIZE:-8}"

NVCC_FLAGS=(
  -std=c++17
  -O3
  -lineinfo
  -Xcompiler
  -Wall
  -arch="${ARCH}"
)

nvcc "${NVCC_FLAGS[@]}" "${SCRIPT_DIR}/batchnorm_bench.cu" \
  -o "${BIN_DIR}/batchnorm_bench"

COMMON_ARGS=(
  --csv
  --batch-size "${BATCH_SIZE}"
  --channels "${CHANNELS}"
  --n-values "${ELEMS_PER_CHANNEL}"
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
  --kernel-name regex:.*block_bn_read_kernel.* \
  "${BIN_DIR}/batchnorm_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant block \
  > "${RESULTS_DIR}/ncu_batchnorm_block.csv"
echo "wrote ${RESULTS_DIR}/ncu_batchnorm_block.csv"

ncu "${NCU_COMMON[@]}" -s 1 -c 1 \
  --kernel-name regex:.*cluster_bn_staged_kernel.* \
  "${BIN_DIR}/batchnorm_bench" \
  "${COMMON_ARGS[@]}" \
  --profile-variant cluster \
  > "${RESULTS_DIR}/ncu_batchnorm_cluster.csv"
echo "wrote ${RESULTS_DIR}/ncu_batchnorm_cluster.csv"
