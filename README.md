# VUDA

VUDA is a header-only lib based on Vulkan that provides a CUDA Runtime API interface for writing GPU-accelerated applications.

# Compile flags

| Flag | Comment |
| :--- | :------ |
| `VUDA_STD_LAYER_ENABLED` | Enables the std vulkan layer |
| `VUDA_DEBUG_ENABLED`     | Enables run-time exceptions  |

# Usage

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

# Change Log

| Date | Changes |
| :--- | :------ |
| 25/10/2018 | memory allocator introduced: mallocHost, hostAlloc, optimized memory transfers, (comparable speeds with cuda in simple vector addition) |
| 17/10/2018 | kernel interface updated: kernel specialization, arbitrary arguments |
| 06/10/2018 | initial commit |
