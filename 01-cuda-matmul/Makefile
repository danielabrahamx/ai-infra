# Build: make [ARCH=sm_86]   (RTX 2050 = sm_86, A100 = sm_80, H100 = sm_90)
ARCH ?= sm_86
NVCC ?= nvcc

bench: src/bench.cu src/kernels.cuh
	$(NVCC) -O3 -arch=$(ARCH) -o bench src/bench.cu -lcublas

test:
	python -m pytest -q

clean:
	rm -f bench bench.exe

.PHONY: test clean
