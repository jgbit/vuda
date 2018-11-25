include ../config.mk

EXECUTABLE=threads_and_streams
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE).cu

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda