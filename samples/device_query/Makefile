include ../config.mk

EXECUTABLE=device_query
SOURCES=$(EXECUTABLE).cpp
CUDA_SRC=$(EXECUTABLE).cpp

$(EXECUTABLE): $(SOURCES)
	$(CC) $(CFLAGS) $^ -o $@ $(INCLUDE) $(LDFLAGS)

cuda: $(CUDA_SRC)
	$(NVCC) $^ -o $(EXECUTABLE)_$@

.PHONY: clean

clean:
	rm -f $(EXECUTABLE) $(EXECUTABLE)_cuda