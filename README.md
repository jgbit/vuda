## VUDA

VUDA is a header-only library based on Vulkan that provides a CUDA Runtime API interface for writing GPU-accelerated applications.

## Setup

The only requirements for developing with the VUDA library is to have access to a Vulkan compatible system and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).

To compile a c++17 program (x64) using the VUDA library it is necessary to specify:

* the path to the VUDA header file
* the path to the Vulkan SDK header files
* the path to the Vulkan SDK library files and add the additional library dependency

## Running a sample

Each sample accompanying the VUDA library includes a simple Makefile for compilation with g++.
Simply set the path to the sample folder and run ```make```.
Additionally, some of the samples includes an equivalent cuda version (for quick comparison).
To compile cuda source code, install the [cuda toolkit](https://developer.nvidia.com/cuda-toolkit).

## Usage

```c++
#include <vuda.hpp>

int main(void)
{
    // assign a device to the thread
    vuda::setDevice(0);
    // allocate memory on the device
    const int N = 5000;
    int a[N], b[N], c[N];
    for(int i = 0; i < N; ++i)
    {
        a[i] = -i;
        b[i] = i * i;
    }
    int *dev_a, *dev_b, *dev_c;
    vuda::malloc((void**)&dev_a, N * sizeof(int));
    vuda::malloc((void**)&dev_b, N * sizeof(int));
    vuda::malloc((void**)&dev_c, N * sizeof(int));
    // copy the arrays a and b to the device
    vuda::memcpy(dev_a, a, N * sizeof(int), vuda::memcpyHostToDevice);
    vuda::memcpy(dev_b, b, N * sizeof(int), vuda::memcpyHostToDevice);
    // run kernel (vulkan shader module)
    const int stream_id = 0;
    const int blocks = 128;
    const int threads = 128;
    vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
    // copy result to host
    vuda::memcpy(c, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost);

    // do something useful with the result in array c ...    
    
    // free memory on device
    vuda::free(dev_a);
    vuda::free(dev_b);
    vuda::free(dev_c);
}
```

## Compile flags

| Flag | Comment |
| :--- | :------ |
| `VUDA_STD_LAYER_ENABLED` | Enables the std vulkan layer |
| `VUDA_DEBUG_ENABLED`     | Enables run-time exceptions  |

## Change Log

| Date | Changes |
| :--- | :------ |
| 23/11/2019 | support for embedded kernels, embedded kernel sample |
| 15/10/2019 | internal state and memory allocation improvements, shared memory sample |
| 20/12/2018 | data structure for managing descriptor set and command buffer pool allocations |
| 25/11/2018 | Makefiles and gcc conformity |
| 13/11/2018 | virtual alloc for local device mem, one buffer per mem alloc, vuda::events, sync memcpy conformity, bandwidthtest, julia set |
| 25/10/2018 | memory allocator introduced: mallocHost, hostAlloc, optimized memory transfers, (comparable speeds with cuda in simple vector addition) |
| 17/10/2018 | kernel interface updated: kernel specialization, arbitrary arguments |
| 06/10/2018 | initial commit |
