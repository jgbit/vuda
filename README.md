# VUDA

VUDA is a header-only lib based on Vulkan that provides a CUDA Runtime API interface for writing GPU-accelerated applications.

# Usage

Minimal example:

```c++
#include <vuda.hpp>
int main()
{
// allocate memory on the device
int *dev_a = 0, *dev_b, *dev_c;
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
vuda::kernelLaunch("add.spv", "main", blocks, threads, stream_id, dev_a, dev_b, dev_c, N);
// copy result to host
vuda::memcpy(c, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost);

// do something useful with the result in array c ...

// free memory on device
vuda::free(dev_a);
vuda::free(dev_b);
vuda::free(dev_c);
}
```
