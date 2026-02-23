#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/results/autotune_$STAMP"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/tuning.csv"
LOG="$OUT_DIR/tuning.log"

echo "latency_ms,config" > "$CSV"

# Candidate space constrained to valid/meaningful combinations for current kernel.
configs=(
  "-DFF_BM=32 -DFF_BK_IN=64 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=16 -DFF_PAD=8 -DFF_WARPS_M=1 -DFF_WARPS_N=8"
  "-DFF_BM=32 -DFF_BK_IN=64 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=16 -DFF_PAD=8 -DFF_WARPS_M=1 -DFF_WARPS_N=4"
  "-DFF_BM=32 -DFF_BK_IN=32 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=16 -DFF_PAD=8 -DFF_WARPS_M=1 -DFF_WARPS_N=8"
  "-DFF_BM=32 -DFF_BK_IN=64 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=8  -DFF_PAD=8 -DFF_WARPS_M=1 -DFF_WARPS_N=8"
  "-DFF_BM=64 -DFF_BK_IN=32 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=16 -DFF_PAD=8 -DFF_WARPS_M=2 -DFF_WARPS_N=4"
  "-DFF_BM=16 -DFF_BK_IN=64 -DFF_BK_OUT=128 -DFF_CLUSTER_SIZE=16 -DFF_PAD=8 -DFF_WARPS_M=1 -DFF_WARPS_N=4"
)

best_lat=""
best_cfg=""

for cfg in "${configs[@]}"; do
  echo "=== TRY: $cfg ===" | tee -a "$LOG"

  if ! timeout 180 nvcc -O3 -arch=native $cfg gemm_fuser_inter_sm.cu -o /tmp/fuser_autotune_bin >>"$LOG" 2>&1; then
    echo "compile_failed" | tee -a "$LOG"
    continue
  fi

  out="$(timeout 120 /tmp/fuser_autotune_bin 2 40 2>&1 || true)"
  echo "$out" >> "$LOG"

  if echo "$out" | grep -q "CUDA error"; then
    echo "runtime_failed" | tee -a "$LOG"
    continue
  fi

  lat="$(echo "$out" | sed -nE 's/.*LATENCY_MS=([0-9]+(\.[0-9]+)?).*/\1/p' | tail -n1)"
  if [[ -z "$lat" ]]; then
    echo "no_latency" | tee -a "$LOG"
    continue
  fi

  echo "$lat,$cfg" >> "$CSV"

  if [[ -z "$best_lat" ]] || awk -v a="$lat" -v b="$best_lat" 'BEGIN{exit !(a<b)}'; then
    best_lat="$lat"
    best_cfg="$cfg"
  fi
done

echo "Best latency: ${best_lat:-N/A}" | tee -a "$LOG"
echo "Best config: ${best_cfg:-N/A}" | tee -a "$LOG"
echo "Output dir: $OUT_DIR"
