

#if defined(__NVCC__)
#include <cuda_runtime.h>
#else
#if !defined(NDEBUG)
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif
#include <vuda_runtime.hpp>
#endif

#include <thread>
#include <iostream>
#include <sstream>
#include <random>
#include "../tools/timer.hpp"

class Test
{
public:
    //
    // Tests
    typedef int(*funcptr2)(const int, const int, const unsigned int, int*, int*, int*);

    static int SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c);
    static int SingleThreadMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);
    static int MultipleThreadsMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);

    static void CheckResult(const unsigned int N, int* a, int* b, int* c);
    static void Launch(std::string name, const int num_threads, const unsigned int N, funcptr2 fptr);
};

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

void HandleException(std::string msg)
{
    try 
    {
        throw;
    }
#if !defined(__NVCC__)
    catch(vk::SystemError& err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
#endif
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }
    catch(...)
    {
        std::cout << "unknown error\n";
        std::exit(EXIT_FAILURE);
    }
}

int Test::SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c)
{
    try
    {
        //
        // settings
        const int deviceID = 0;
        cudaSetDevice(deviceID);

        const int blocks = 128;
        const int threads = 128;
        const int bytesize = N * sizeof(int);

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        int *dev_a = 0, *dev_b = 0, *dev_c = 0;

        //
        // allocate memory on the device
        cudaMalloc((void**)&dev_a, bytesize);
        cudaMalloc((void**)&dev_b, bytesize);
        cudaMalloc((void**)&dev_c, bytesize);

        //
        // copy the arrays a and b to the device
        cudaMemcpy(dev_a, a, bytesize, cudaMemcpyHostToDevice);
        cudaMemcpy(dev_b, b, bytesize, cudaMemcpyHostToDevice);

        //
        // run kernel
#if defined(__NVCC__)
        add<<< blocks, threads >>>(dev_a, dev_b, dev_c, N);
#else
        const int stream_id = 0;
        vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
#endif

        //
        // copy result to host
        cudaMemcpy(c, dev_c, bytesize, cudaMemcpyDeviceToHost);

        //
        // display results
#ifdef THREAD_VERBOSE_OUTPUT
        ostr << "thread id: " << tid << ", device id: " << deviceID << ", stream id: " << stream_id << ": ";
        std::cout << ostr.str();
#endif

        //
        // free memory on device
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }    
    catch(...)
    {
        HandleException("SingleThreadSingleStreamExample");
    }

    return 0;
}

/*
    Multiple CUDA streams example taken from the book CUDA BY EXAMPLE
*/
int Test::SingleThreadMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c)
{
    try
    {
        //
        // hardcore device 0
        const int deviceID = 0;
        cudaSetDevice(deviceID);

        //
        // kernel params
        const int nstreams = 2;

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        if(FULL_DATA_SIZE % 2 != 0)
            return EXIT_FAILURE;

        const int N = FULL_DATA_SIZE / 2;
        int *dev_a0, *dev_b0, *dev_c0;
        int *dev_a1, *dev_b1, *dev_c1;

        //
        // create streams
    #if defined(__NVCC__)
        cudaStream_t stream0_id, stream1_id;
        cudaStreamCreate(&stream0_id);
        cudaStreamCreate(&stream1_id);
    #else
        // for now, vuda does not create streams, but operates with a maximum number of compute queues internally
        const int stream0_id = 0;
        const int stream1_id = 1;
    #endif

        //
        // allocate memory on the device
        cudaMalloc((void**)&dev_a0, N * sizeof(int));
        cudaMalloc((void**)&dev_b0, N * sizeof(int));
        cudaMalloc((void**)&dev_c0, N * sizeof(int));
        cudaMalloc((void**)&dev_a1, N * sizeof(int));
        cudaMalloc((void**)&dev_b1, N * sizeof(int));
        cudaMalloc((void**)&dev_c1, N * sizeof(int));

        //
        // hardcode stream submission
        for(unsigned int i = 0; i < FULL_DATA_SIZE; i += nstreams * N)
        {
            //
            // copy the arrays a and b to the device
            cudaMemcpyAsync(dev_a0, a + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0_id);
            cudaMemcpyAsync(dev_a1, a + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1_id);

            cudaMemcpyAsync(dev_b0, b + i, N * sizeof(int), cudaMemcpyHostToDevice, stream0_id);
            cudaMemcpyAsync(dev_b1, b + i + N, N * sizeof(int), cudaMemcpyHostToDevice, stream1_id);

            //
            // run kernel
            const int blocks = 128;
            const int threads = 128;
        #if defined(__NVCC__)
            add<<< blocks, threads, 0, stream0_id >>> (dev_a0, dev_b0, dev_c0, N);
            add<<< blocks, threads, 0, stream1_id >>> (dev_a1, dev_b1, dev_c1, N);
        #else
            vuda::launchKernel("add.spv", "main", stream0_id, blocks, threads, dev_a0, dev_b0, dev_c0, N);
            vuda::launchKernel("add.spv", "main", stream1_id, blocks, threads, dev_a1, dev_b1, dev_c1, N);
        #endif

            //
            // copy result to host
            cudaMemcpyAsync(c + i, dev_c0, N * sizeof(int), cudaMemcpyDeviceToHost, stream0_id);
            cudaMemcpyAsync(c + i + N, dev_c1, N * sizeof(int), cudaMemcpyDeviceToHost, stream1_id);
        }

    #if defined(__NVCC__)
        cudaStreamSynchronize(stream0_id);
        cudaStreamSynchronize(stream1_id);
        cudaStreamDestroy(stream0_id);
        cudaStreamDestroy(stream1_id);
    #endif

        //
        // display results
#ifdef THREAD_VERBOSE_OUTPUT
        ostr << "thread id: " << tid << ", device id: " << deviceID << ", stream id: " << stream0_id << ", " << stream1_id << ": ";
        std::cout << ostr.str();
#endif  

        //
        // free memory on device
        cudaFree(dev_a0);
        cudaFree(dev_b0);
        cudaFree(dev_c0);
        cudaFree(dev_a1);
        cudaFree(dev_b1);
        cudaFree(dev_c1);
    }
    catch(...)
    {
        HandleException("SingleThreadMultipleStreamsExample");
    }

    return 0;
}

int Test::MultipleThreadsMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c)
{
    try
    {
        //
        // hardcode device 0
        const int deviceID = 0;
        cudaSetDevice(deviceID);

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << ", stream: " << stream_id << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        if(FULL_DATA_SIZE % nthreads != 0)
        {
            std::cout << "FULL_DATA_SIZE should be divisible by " << nthreads << std::endl;
            return EXIT_FAILURE;
        }

        const int N = FULL_DATA_SIZE / nthreads;
        int *dev_a, *dev_b, *dev_c;

        //
        // allocate memory on the device
        const int bytesize = N * sizeof(int);
        cudaMalloc((void**)&dev_a, bytesize);
        cudaMalloc((void**)&dev_b, bytesize);
        cudaMalloc((void**)&dev_c, bytesize);

        //
        // create streams
    #if defined(__NVCC__)
        cudaStream_t stream_id;
        cudaStreamCreate(&stream_id);
    #else
        // for now, vuda does not create streams, but operates with a maximum number of compute queues internally
        const int stream_id = tid;
    #endif

        //
        // kernel params
        const int blocks = 128;
        const int threads = 128;

        // copy the arrays a and b to the device
        cudaMemcpyAsync(dev_a, a + tid * N, bytesize, cudaMemcpyHostToDevice, stream_id);
        cudaMemcpyAsync(dev_b, b + tid * N, bytesize, cudaMemcpyHostToDevice, stream_id);

        //
        // run kernel
        #if defined(__NVCC__)
            add <<< blocks, threads, 0, stream_id >>> (dev_a, dev_b, dev_c, N);
        #else
            vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);
        #endif

        //
        // copy result to host
        cudaMemcpyAsync(c + tid * N, dev_c, bytesize, cudaMemcpyDeviceToHost, stream_id);

    #if defined(__NVCC__)
        cudaStreamSynchronize(stream_id);
        cudaStreamDestroy(stream_id);
    #endif

        //
        // free memory on device
        cudaFree(dev_a);
        cudaFree(dev_b);
        cudaFree(dev_c);
    }
    catch(...)
    {
        HandleException("MultipleThreadsMultipleStreamsExample");
    }

    return 0;
}

void Test::CheckResult(const unsigned int N, int* a, int* b, int* c)
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

void Test::Launch(std::string name, const int num_threads, const unsigned int N, funcptr2 fptr)
{
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
    const int totalRuns = 12;
    double totalElapsedTime;

    std::cout << name << std::endl;
    std::cout << std::string(name.length(), '=') << std::endl;

    // 
    // warm up, pre-create the logical devices
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    for(int deviceID = 0; deviceID < deviceCount; ++deviceID)
        cudaSetDevice(deviceID);
    cudaSetDevice(0);

    totalElapsedTime = 0.0;
    for(int run = 0; run < totalRuns; ++run)
    {
        // reset test
        for(unsigned int i = 0; i < N; ++i)
            c[i] = 0;

        timer.tic();
        {
            for(int i = 0; i < num_threads; ++i)
                t[i] = std::thread(fptr, i, num_threads, N, a, b, c);
            for(int i = 0; i < num_threads; ++i)
                t[i].join();
        }
        double elapsed = timer.toc();
        std::cout << "run: " << run << ", runtime: " << elapsed << "s" << std::endl;

        // exclude the first run (as kernel and memory will be allocated)
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
}

int main(int argc, char *argv[])
{
    try
        {
        //
        // arguments
        std::vector<const char*> args;
        for(size_t i = 0; i < (size_t)argc; i++)
            args.push_back(argv[i]);

        //
        // default parameters
        // {problem size, run all tests }

        unsigned int N = 1000000;
        //unsigned int test = 0;
        std::vector<bool> test_runs(3, true);
        /*if(HandleArguments(N, test) == EXIT_SUCCESS)
            return EXIT_SUCCESS;*/

        //
        // Examples

        std::vector<std::string> test_names = {
        "Single device, single thread, single stream",
        "Single device, single thread, multiple streams",
        //"Single device, multiple threads, single stream",
        "Single device, multiple threads, multiple streams"
        };

        int testid;

        //
        // single device, single thread, single stream
        testid = 0;
        if(test_runs[testid])
            Test::Launch(test_names[testid], 1, N, &Test::SingleThreadSingleStreamExample);

        //
        // single device, single thread, multiple streams
        testid = 1;
        if(test_runs[testid])
            Test::Launch(test_names[testid], 1, N, &Test::SingleThreadMultipleStreamsExample);

        //
        // single device, multiple threads, multiple streams
        testid = 2;
        if(test_runs[testid])
            Test::Launch(test_names[testid], 8, N, &Test::MultipleThreadsMultipleStreamsExample);

    }
    catch(...)
    {
        HandleException("main");
    }

#ifndef NDEBUG
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return EXIT_SUCCESS;
}