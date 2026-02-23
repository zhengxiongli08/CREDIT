#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

RUNS="${1:-5}"
PY_WARMUP="${PY_WARMUP:-50}"
PY_ITERS="${PY_ITERS:-20}"
TRT_WARMUP="${TRT_WARMUP:-100}"
TRT_ITERS="${TRT_ITERS:-500}"
CUSTOM_WARMUP="${CUSTOM_WARMUP:-20}"
CUSTOM_ITERS="${CUSTOM_ITERS:-100}"
CUSTOM_SOURCE="${CUSTOM_SOURCE:-gemm_fuser_cublaslt.cu}"
CUSTOM_BIN="${CUSTOM_BIN:-gemm_fuser_cublaslt}"
CUSTOM_NVCC_FLAGS="${CUSTOM_NVCC_FLAGS:--O3}"
CUSTOM_NVCC_POST_FLAGS="${CUSTOM_NVCC_POST_FLAGS:--lcublas -lcublasLt}"
# workspace_mb fp16_accum
CUSTOM_ARGS="${CUSTOM_ARGS:-16 1}"
TIMEOUT_PYTORCH_SEC="${TIMEOUT_PYTORCH_SEC:-300}"
TIMEOUT_TRT_SEC="${TIMEOUT_TRT_SEC:-600}"
TIMEOUT_CUSTOM_SEC="${TIMEOUT_CUSTOM_SEC:-300}"
RUN_PYTORCH="${RUN_PYTORCH:-1}"
RUN_TENSORRT="${RUN_TENSORRT:-1}"
RUN_CUSTOM="${RUN_CUSTOM:-1}"

STAMP="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/results/$STAMP"
mkdir -p "$OUT_DIR"

CSV_FILE="$OUT_DIR/latency_runs.csv"
LOG_FILE="$OUT_DIR/console.log"

echo "backend,run,latency_ms" > "$CSV_FILE"

echo "[1/2] Compiling inter-SM fused kernel..." | tee -a "$LOG_FILE"
if ! nvcc $CUSTOM_NVCC_FLAGS -arch=native "$CUSTOM_SOURCE" -o "$CUSTOM_BIN" $CUSTOM_NVCC_POST_FLAGS 2>>"$LOG_FILE"; then
  echo "native arch compile failed, retrying with sm_90a" | tee -a "$LOG_FILE"
  nvcc $CUSTOM_NVCC_FLAGS -arch=sm_90a "$CUSTOM_SOURCE" -o "$CUSTOM_BIN" $CUSTOM_NVCC_POST_FLAGS 2>>"$LOG_FILE"
fi

echo "[2/2] Running benchmarks for $RUNS rounds..." | tee -a "$LOG_FILE"

extract_latency() {
  local text="$1"
  echo "$text" | sed -nE 's/.*LATENCY_MS=([0-9]+(\.[0-9]+)?).*/\1/p' | tail -n1
}

for ((r=1; r<=RUNS; r++)); do
  echo "=== Round $r/$RUNS ===" | tee -a "$LOG_FILE"

  if [[ "$RUN_PYTORCH" == "1" ]]; then
    echo "[PyTorch]" | tee -a "$LOG_FILE"
    pyt_out="$(timeout "$TIMEOUT_PYTORCH_SEC" python3 gemm_pytorch.py --warmup "$PY_WARMUP" --iters "$PY_ITERS" 2>&1 | tee -a "$LOG_FILE" || true)"
    pyt_lat="$(extract_latency "$pyt_out")"
    [[ -z "$pyt_lat" ]] && pyt_lat="nan"
    echo "pytorch,$r,$pyt_lat" >> "$CSV_FILE"
  fi

  if [[ "$RUN_TENSORRT" == "1" ]]; then
    echo "[TensorRT]" | tee -a "$LOG_FILE"
    trt_out="$(timeout "$TIMEOUT_TRT_SEC" python3 gemm_tensorrt.py --warmup "$TRT_WARMUP" --iters "$TRT_ITERS" 2>&1 | tee -a "$LOG_FILE" || true)"
    trt_lat="$(extract_latency "$trt_out")"
    [[ -z "$trt_lat" ]] && trt_lat="nan"
    echo "tensorrt,$r,$trt_lat" >> "$CSV_FILE"
  fi

  if [[ "$RUN_CUSTOM" == "1" ]]; then
    echo "[Inter-SM Fuser]" | tee -a "$LOG_FILE"
    cus_out="$(timeout "$TIMEOUT_CUSTOM_SEC" ./"$CUSTOM_BIN" "$CUSTOM_WARMUP" "$CUSTOM_ITERS" $CUSTOM_ARGS 2>&1 | tee -a "$LOG_FILE" || true)"
    cus_lat="$(extract_latency "$cus_out")"
    [[ -z "$cus_lat" ]] && cus_lat="nan"
    echo "inter_sm_fuser,$r,$cus_lat" >> "$CSV_FILE"
  fi
done

python3 analyze_ffn_inter_sm.py --csv "$CSV_FILE" --outdir "$OUT_DIR"

echo "Done. Results in: $OUT_DIR"
