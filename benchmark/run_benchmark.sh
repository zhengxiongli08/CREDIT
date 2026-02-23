# Exit on error
set -e

# Define filenames
SOURCE_FILE="benchmark.cu"
OUTPUT_BIN="benchmark_run"
RESULTS_FILE="raw_results.txt"
ITERATIONS=100

# 1. Compile the code
echo "Compiling..."
nvcc -arch=native -O3 $SOURCE_FILE -o $OUTPUT_BIN

if [ $? -ne 0 ]; then
    echo "Compilation failed!"
    exit 1
fi

# 2. Clear previous results
if [ -f $RESULTS_FILE ]; then
    rm $RESULTS_FILE
fi

echo "Starting $ITERATIONS iterations..."
echo "Output will be saved to $RESULTS_FILE"

# 3. Loop 100 times
for ((i=1; i<=ITERATIONS; i++))
do
   echo "Run #$i"
   echo "--- RUN $i ---" >> $RESULTS_FILE
   ./$OUTPUT_BIN >> $RESULTS_FILE
done

echo "Done! Data collected in $RESULTS_FILE"