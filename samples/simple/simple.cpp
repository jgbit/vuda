#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include <iostream>

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
    vuda::kernelLaunch("add.spv", "main", blocks, threads, stream_id, dev_a, dev_b, dev_c, N);
    // copy result to host
    vuda::memcpy(c, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost);

    // do something useful with the result in array c ...    
    for(int i = 0; i < N; ++i)
        if(a[i] + b[i] != c[i])
        {
            std::cout << "wrong result at index " << i << std::endl;
            break;
        }

    // free memory on device
    vuda::free(dev_a);
    vuda::free(dev_b);
    vuda::free(dev_c);
}