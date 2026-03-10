#!/bin/bash

# Run gemm_pytorch.py with different matrix dimensions
# M is always 128, sweep through different [k, n] pairs

M=128

# Array of [k, n] pairs
K_N_PAIRS=(
    "32 512"
    "512 256"
    "416 512"
    "768 3072"
    "4096 16384"
    "1024 4096"
    "768 768"
    "2048 8192"
    "512 2048"
    "384 1536"
)

echo "Starting GEMM PyTorch sweep with M=$M"
echo "=========================================="

for pair in "${K_N_PAIRS[@]}"; do
    read -r k n <<< "$pair"
    echo ""
    echo "Running with M=$M, K=$k, N=$n"
    python gemm_pytorch.py --m $M --k $k --n $n
    echo "=========================================="
done

echo "Sweep completed!"
