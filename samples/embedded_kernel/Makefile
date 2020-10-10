include ../config.mk

EXECUTABLE=embedded_kernel
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE)_cuda.cu

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)
	glslangValidator -V kernel.comp -o kernel.spv

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda