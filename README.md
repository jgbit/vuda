## VUDA

VUDA is a header-only library based on Vulkan that provides a CUDA Runtime API interface for writing GPU-accelerated applications.

## Documentation

VUDA is based on the [Vulkan API](https://www.khronos.org/vulkan/). The functionality of VUDA conforms (as much as possible) to the specification of the CUDA runtime. For normal usage consult the reference guide for the [NVIDIA CUDA Runtime API](https://docs.nvidia.com/cuda/cuda-runtime-api/index.html), otherwise check the VUDA wiki:

- [Change List](https://github.com/jgbit/vuda/wiki/Change-List)
- [Setup and Compilation](https://github.com/jgbit/vuda/wiki/Setup-and-Compilation)
- [Deviations from CUDA](https://github.com/jgbit/vuda/wiki/Deviations-from-CUDA)
- [Implementation Details](https://github.com/jgbit/vuda/wiki/Implementation-Details)

## Usage

All VUDA functionality can be accessed by including `vuda.hpp` and using its namespace `vuda::`.
Alternatively, one can utilize `vuda_runtime.hpp` which wraps and redirect all CUDA functionality.

```c++
#if defined(__NVCC__)
    #include <cuda_runtime.h>
#else
    #include <vuda_runtime.hpp>
#endif

int main(void)
{
    // assign a device to the thread
    cudaSetDevice(0);
    // allocate memory on the device
    const int N = 5000;
    int a[N], b[N], c[N];
    for(int i = 0; i < N; ++i)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    int *dev_a, *dev_b, *dev_c;
    cudaMalloc((void**)&dev_a, N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * sizeof(int));
    // copy the arrays a and b to the device
    cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * sizeof(int), cudaMemcpyHostToDevice);
    // run kernel (vulkan shader module)
    const int blocks = 128;
    const int threads = 128;
#if defined(__NVCC__)
    add<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);
#else
    const int stream_id = 0;
    vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
#endif
    // copy result to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // do something useful with the result in array c ...        

    // free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}
```