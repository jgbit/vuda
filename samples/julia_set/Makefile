include ../config.mk

EXECUTABLE=julia_set
SOURCES=$(EXECUTABLE).cpp miniz.c
CUDA_SRC=$(EXECUTABLE)_cuda.cu

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)
	glslangValidator -V julia_kernel.comp -o julia_kernel.spv

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda