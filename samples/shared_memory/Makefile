include ../config.mk

EXECUTABLE=shared_memory
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE).cu

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)
	glslangValidator -V glslDynamicReverse.comp -o glslDynamicReverse.spv
	glslangValidator -V glslStaticReverse.comp -o glslStaticReverse.spv

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda