NVCC ?= nvcc
CUDA_ARCH ?= native
NVCCFLAGS ?= -std=c++17 -O3
BUILD_DIR ?= build
BENCHMARK_NAMES := primitive_benchmark cluster_control
PRIMITIVE_BENCHMARK := $(BUILD_DIR)/primitive_benchmark
CLUSTER_CONTROL := $(BUILD_DIR)/cluster_control
BENCHMARKS := $(PRIMITIVE_BENCHMARK) $(CLUSTER_CONTROL)
WORKLOAD_NAMES := \
	layernorm_backward \
	weighted_var_backward \
	pearson_backward \
	softmax_logits_backward \
	lars_momentum \
	rowwise_quant
WORKLOAD_BINARIES := $(addprefix $(BUILD_DIR)/,$(WORKLOAD_NAMES))

.PHONY: all benchmark workloads clean $(BENCHMARK_NAMES) $(WORKLOAD_NAMES)

all: benchmark workloads

benchmark: $(BENCHMARKS)

workloads: $(WORKLOAD_BINARIES)

primitive_benchmark: $(PRIMITIVE_BENCHMARK)

cluster_control: $(CLUSTER_CONTROL)

$(WORKLOAD_NAMES): %: $(BUILD_DIR)/%

$(PRIMITIVE_BENCHMARK): src/primitive_benchmark.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall -arch=$(CUDA_ARCH) $< -o $@

$(CLUSTER_CONTROL): src/cluster_control.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall -arch=$(CUDA_ARCH) $< -o $@

$(BUILD_DIR)/%: src/workloads/%.cu | $(BUILD_DIR)
	$(NVCC) $(NVCCFLAGS) -Xcompiler -Wall -arch=$(CUDA_ARCH) $< -o $@

$(BUILD_DIR):
	mkdir -p $@

clean:
	rm -rf $(BUILD_DIR)
