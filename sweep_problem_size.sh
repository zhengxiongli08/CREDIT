#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-/home/zli2793/tools/miniforge3/envs/tile/bin/python3.11}"
M_LIST="${M_LIST:-128 256 512}"
K="${K:-4096}"
N="${N:-16384}"
PY_WARMUP="${PY_WARMUP:-20}"
PY_ITERS="${PY_ITERS:-20}"
TRT_WARMUP="${TRT_WARMUP:-20}"
TRT_ITERS="${TRT_ITERS:-100}"
CUS_WARMUP="${CUS_WARMUP:-20}"
CUS_ITERS="${CUS_ITERS:-100}"
TIMEOUT_SEC="${TIMEOUT_SEC:-900}"
CUSTOM_SOURCE="${CUSTOM_SOURCE:-gemm_fuser_cublaslt.cu}"
CUSTOM_BIN="${CUSTOM_BIN:-gemm_fuser_cublaslt_sweep}"
CUSTOM_NVCC_FLAGS="${CUSTOM_NVCC_FLAGS:--O3}"
CUSTOM_NVCC_POST_FLAGS="${CUSTOM_NVCC_POST_FLAGS:--lcublas -lcublasLt}"
CUSTOM_ARGS="${CUSTOM_ARGS:-16 1}"

OUT_DIR="results/size_sweep_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$OUT_DIR"
CSV="$OUT_DIR/size_sweep.csv"
LOG="$OUT_DIR/console.log"

echo "m,k,n,pytorch_ms,tensorrt_ms,inter_sm_ms" > "$CSV"

a(){
  sed -nE 's/.*LATENCY_MS=([0-9]+(\.[0-9]+)?).*/\1/p' | tail -n1
}

for M in $M_LIST; do
  echo "=== M=$M K=$K N=$N ===" | tee -a "$LOG"

  pyt="nan"
  trt="nan"
  cus="nan"

  out=$(timeout "$TIMEOUT_SEC" "$PYTHON_BIN" gemm_pytorch.py --m "$M" --k "$K" --n "$N" --warmup "$PY_WARMUP" --iters "$PY_ITERS" 2>&1 | tee -a "$LOG" || true)
  val=$(echo "$out" | a)
  [[ -n "$val" ]] && pyt="$val"

  out=$(timeout "$TIMEOUT_SEC" "$PYTHON_BIN" gemm_tensorrt.py --m "$M" --k "$K" --n "$N" --warmup "$TRT_WARMUP" --iters "$TRT_ITERS" 2>&1 | tee -a "$LOG" || true)
  val=$(echo "$out" | a)
  [[ -n "$val" ]] && trt="$val"

  NVCC_FLAGS="$CUSTOM_NVCC_FLAGS -DFF_M_GLOBAL=${M} -DFF_K_GLOBAL=${K} -DFF_N_GLOBAL=${N}"
  if ! nvcc $NVCC_FLAGS -arch=native "$CUSTOM_SOURCE" -o "$CUSTOM_BIN" $CUSTOM_NVCC_POST_FLAGS 2>>"$LOG"; then
    nvcc $NVCC_FLAGS -arch=sm_90a "$CUSTOM_SOURCE" -o "$CUSTOM_BIN" $CUSTOM_NVCC_POST_FLAGS 2>>"$LOG"
  fi

  out=$(timeout "$TIMEOUT_SEC" ./$CUSTOM_BIN "$CUS_WARMUP" "$CUS_ITERS" $CUSTOM_ARGS 2>&1 | tee -a "$LOG" || true)
  val=$(echo "$out" | a)
  [[ -n "$val" ]] && cus="$val"

  echo "$M,$K,$N,$pyt,$trt,$cus" | tee -a "$CSV"
done

"$PYTHON_BIN" - "$CSV" <<'PY'
import csv, sys, os
csv_path = sys.argv[1]
rows = list(csv.DictReader(open(csv_path)))
print("\nSummary:")
for r in rows:
    m=r['m']
    p=float(r['pytorch_ms']) if r['pytorch_ms']!='nan' else float('nan')
    t=float(r['tensorrt_ms']) if r['tensorrt_ms']!='nan' else float('nan')
    c=float(r['inter_sm_ms']) if r['inter_sm_ms']!='nan' else float('nan')
    spt=(p/c) if c==c and p==p and c>0 else float('nan')
    stt=(t/c) if c==c and t==t and c>0 else float('nan')
    print(f"M={m}: pytorch={p:.6f} ms, tensorrt={t:.6f} ms, inter_sm={c:.6f} ms, speedup_vs_pt={spt:.3f}, speedup_vs_trt={stt:.3f}")
PY

echo "Saved: $CSV"
