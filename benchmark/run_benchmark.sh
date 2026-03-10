# Exit when error
set -e

# Compile
nvcc -arch=native -std=c++20 benchmark.cu -o ../bin/benchmark
# Run
../bin/benchmark
