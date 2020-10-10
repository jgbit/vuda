include ../config.mk

EXECUTABLE=specializedkernels
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE)_cuda.cu

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)
	glslangValidator -V threadid.comp -o threadid.spv

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda