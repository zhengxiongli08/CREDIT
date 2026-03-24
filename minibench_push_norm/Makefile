CUDA_ARCH ?= sm_120
NVCC ?= nvcc
NVCCFLAGS ?= -O3 --use_fast_math -std=c++17 -lineinfo -Xcompiler=-Wno-psabi
TARGET := push_norm_bench

all: $(TARGET)

$(TARGET): push_norm_bench.cu
	$(NVCC) $(NVCCFLAGS) -arch=$(CUDA_ARCH) $< -o $@

clean:
	rm -f $(TARGET)
