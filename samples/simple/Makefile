include ../config.mk

EXECUTABLE=simple
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE).cpp

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)
	glslangValidator -V add.comp -o add.spv

cuda: $(CUDA_SRC)
	$(NVCC) $^ -x cu -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda