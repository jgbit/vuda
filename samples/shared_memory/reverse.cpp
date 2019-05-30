//#include "shared_memory.h"
#define VUDA_DEBUG_KERNEL

#ifdef VUDA_DEBUG_KERNEL
#include <vuda.hpp>

vuda::dim3 gridDim;
vuda::dim3 blockDim;
vuda::dim3 blockIdx;
vuda::debug::threadIdx_def threadIdx;// thread identifier access

#endif

//
// static shared memory
#ifdef VUDA_DEBUG_KERNEL
#define __shared__ static
#endif

__global__ void staticReverse(int *d, int n)
{
    __shared__ int s[64];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

//
// dynamic shared memory
//
#ifdef VUDA_DEBUG_KERNEL
#undef __shared__
#define __shared__
// externally defined shared memory
int s[64];
#endif

__global__ void dynamicReverse(int *d, int n)
{
    extern __shared__ int s[];
    int t = threadIdx.x;
    int tr = n - t - 1;
    s[t] = d[t];
    __syncthreads();
    d[t] = s[tr];
}

void VudaDebugKernel(const int n, int* a, int* r)
{
    //
    // the cuda kernel can be emulated on the host using vuda (compiles with c++)
    // NOTE: the passed arguments must reside on the host

    int* d_host = new int [n];
    for(int i = 0; i < n; i++)
        d_host[i] = a[i];

    vuda::hostKernel(staticReverse, 1, n, d_host, n);

    // check result
    for(int i = 0; i < n; i++)
    {
        if(d_host[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d_host[i], r[i]);
    }

    delete[] d_host;





    //vuda::launchKernel(staticReverse, dim3(1), dim3(n), args, 0, 0);

    /*//
    // run version with static shared memory
    vuda::memcpy(d_d, a, n * sizeof(int), vuda::memcpyHostToDevice);
    //staticReverse << <1, n >> > (d_d, n);
    vuda::launchKernel("staticReverse.spv", "main", 0, 1, n, d_d);
    vuda::memcpy(d, d_d, n * sizeof(int), vuda::memcpyDeviceToHost);*/
}