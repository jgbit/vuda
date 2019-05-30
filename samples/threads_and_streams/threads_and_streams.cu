
#include <stdio.h>
#include <random>
#include <thread>
#include <iostream>
#include <sstream>

#include <cuda_runtime.h>
#include "../tools/timer.hpp"

__global__ void vectorAdd(const int *a, const int *b, int *c, int N)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;

    while(tid < N)
    {
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c)
{
    // Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;

    // Print the vector length to be used, and compute its size
    int numElements = N;    
    int size = N * sizeof(int);

    // Allocate the device input vector A
    int* d_A = NULL;
    int* d_B = NULL;
    int *d_C = NULL;

    err = cudaMalloc((void **)&d_A, size);
    err = cudaMalloc((void **)&d_B, size);
    err = cudaMalloc((void **)&d_C, size);

    err = cudaMemcpy(d_A, a, size, cudaMemcpyHostToDevice);
    err = cudaMemcpy(d_B, b, size, cudaMemcpyHostToDevice);

    // Launch the Vector Add CUDA Kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = 128;    
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);

    err = cudaMemcpy(c, d_C, size, cudaMemcpyDeviceToHost);

    // Free device global memory
    err = cudaFree(d_A);
    err = cudaFree(d_B);
    err = cudaFree(d_C);

    return err;
}

void CheckResult(const unsigned int N, int* a, int* b, int* c)
{
    std::ostringstream ostr;
    bool resultok = true;
    for(unsigned int i = 0; i < N; ++i)
    {
        if(a[i] + b[i] != c[i])
        {
            ostr << "ERROR: kernel execution did not provide the right result at index: " << i << ", ";
            ostr << a[i] << " + " << b[i] << " != " << c[i] << std::endl;
            resultok = false;
            break;
        }
    }
    if(resultok == true)
        ostr << "kernel result verified!" << std::endl;

    std::cout << ostr.str();
}

int main(void)
{
    
    const unsigned int N = 1000000;
    const int num_threads = 1;
    //
    // fill arrays on the Host
    std::cout << "Generating random host data arrays of size " << N << " (int)" << std::endl;

    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, 100);
    for(unsigned int i = 0; i < N; ++i)
    {
        a[i] = dist(mt); // -i + tid;
        b[i] = dist(mt); // i*i;
    }
    std::cout << "done." << std::endl << std::endl;

    //
    // threads        
    Timer timer;
    std::thread* t = new std::thread[num_threads];
    const int totalRuns = 10;
    double totalElapsedTime;

    cudaSetDevice(0);
    totalElapsedTime = 0.0;
    double elapsed = 0.0;
    for(int run = 0; run < totalRuns; ++run)
    {
        // reset test
        for(unsigned int i = 0; i < N; ++i)
            c[i] = 0;

        timer.tic();
        {
            for(int i = 0; i < num_threads; ++i)
                t[i] = std::thread(&SingleThreadSingleStreamExample, i, num_threads, N, a, b, c);
            for(int i = 0; i < num_threads; ++i)
                t[i].join();
        }
        elapsed = timer.toc();
        std::cout << "run: " << run << ", runtime: " << elapsed << "s" << std::endl;

        if(run > 0)
            totalElapsedTime += elapsed;

        // check result
        CheckResult(N, a, b, c);
    }
    if(totalRuns > 1)
        std::cout << "avg. runtime: " << totalElapsedTime / (totalRuns - 1) << "s" << std::endl;

    //
    // clean up
    delete[] t;
    delete[] a;
    delete[] b;
    delete[] c;    
    
    return 0;
}