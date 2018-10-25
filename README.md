## VUDA

VUDA is a header-only library based on Vulkan that provides a CUDA Runtime API interface for writing GPU-accelerated applications.

## Setup

The only requirements for developing with the VUDA library is to have access to a vulkan compatible system and install the [Vulkan SDK](https://vulkan.lunarg.com/sdk/home).
To compile a c++11 program using the VUDA library it is necessary to specify:
* the include path to the Vulkan SDK header files
* the include path to the Vulkan SDK library files
* the additional dependency to the vulkan-1.lib
* the include path to the VUDA header file
It is recommended to compile towards x64.

## Usage

```c++
#include <vuda.hpp>
int main(void)
{
    // assign a device to thread
    vuda::SetDevice(0);
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
    vuda::kernelLaunch("add.spv", "main", blocks, stream_id, threads, dev_a, dev_b, dev_c, N);
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
| 25/10/2018 | memory allocator introduced: mallocHost, hostAlloc, optimized memory transfers, (comparable speeds with cuda in simple vector addition) |
| 17/10/2018 | kernel interface updated: kernel specialization, arbitrary arguments |
| 06/10/2018 | initial commit |
