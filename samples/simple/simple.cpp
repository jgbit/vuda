
#if defined(__NVCC__)
#include <cuda_runtime.h>
#include <iostream>
#else
#if !defined(NDEBUG)
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif
#include <vuda_runtime.hpp>
#endif

#if defined(__NVCC__)

__global__ void add(const int* dev_a, const int* dev_b, int* dev_c, const int N)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    while(tid < N)
    {
        dev_c[tid] = dev_a[tid] + dev_b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

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
    //void *args[] = { (void*)&dev_a, (void*)&dev_b, (void*)&dev_c, (void*)&N };
    //cudaLaunchKernel(add, blocks, threads, args, 0, stream_id);
    add<<<blocks, threads>>>(dev_a, dev_b, dev_c, N);
#else
    const int stream_id = 0;
    vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
#endif
    // copy result to host
    cudaMemcpy(c, dev_c, N * sizeof(int), cudaMemcpyDeviceToHost);

    // do something useful with the result in array c ...
    for(int i = 0; i < N; ++i)
        if(a[i] + b[i] != c[i])
        {
            std::cout << "wrong result at index " << i << std::endl;
            break;
        }

    // free memory on device
    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
}