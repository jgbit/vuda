
/*

EXAMPLE BUILD ON
How to Optimize Data Transfers in CUDA C/C++
https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/

*/

#include <iostream>
#include <iomanip>
#include <sstream>
#include <vector>
#include <cstring>

#if defined(__NVCC__)
#include <cuda_runtime.h>
#else
#if !defined(NDEBUG)
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif
#include <vuda_runtime.hpp>
#endif

#include "../tools/timer.hpp"

double profileTransfer(float *dst, const float *src, unsigned int n, cudaMemcpyKind kind, std::string desc, bool verbose=false)
{
    std::cout << std::endl << desc << ", ";
    if(kind == cudaMemcpyHostToHost)
        std::cout << "HtH";
    else if(kind == cudaMemcpyHostToDevice)
        std::cout << "HtD";
    else if(kind == cudaMemcpyDeviceToHost)
        std::cout << "DtH";
    else if(kind == cudaMemcpyDeviceToDevice)
        std::cout << "DtD";
    std::cout << " transfer" << std::endl;

    Timer timer;
    double time, total = 0.0f;
    const unsigned int bytes = n * sizeof(float);
    const int runs = 10;

    /*cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);*/
    

    for(int run=0; run<=runs; ++run)
    {
        //cudaEventRecord(startEvent, 0);
        timer.tic();
        cudaMemcpy(dst, src, bytes, kind);
        time = timer.toc();
        //cudaEventRecord(stopEvent, 0);
        //cudaEventSynchronize(stopEvent);

        //cudaEventElapsedTime(&time, startEvent, stopEvent);
        if(run != 0)
            total += time;
        if(verbose)
            std::cout << "  iter: " << run << ", transfer time (s)   : " << time << std::endl;
    }

    // clean up events
    //cudaEventDestroy(startEvent);
    //cudaEventDestroy(stopEvent);

    time = total / (double)runs;
    double bw = bytes * 1e-9 / time;
    std::cout << "  avg. transfer time (s)   : " << time << std::endl;
    std::cout << "  avg. bandwidth     (GB/s): " << bw << std::endl;
    return bw;
}

void title(std::string title)
{
    std::cout << std::endl;
    std::cout << std::string(title.size(), '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(title.size(), '=') << std::endl;
}

void print_bwresults(std::vector<std::string> memtype, std::vector<double> bwprofile)
{
    std::stringstream ostr;
    const unsigned int nmemtypes = (unsigned int)memtype.size();
    const int wid = 12;

    title("Bandwidth transfer table");

    ostr << std::setw(wid) << std::left << "src\\dst";
    for(unsigned int src = 0; src < nmemtypes; ++src)
        ostr << std::setw(wid) << std::left << memtype[src];

    int iter = 0;
    ostr << std::endl;
    for(unsigned int src = 0; src < nmemtypes; ++src)
    {
        ostr << std::setw(wid) << std::left << memtype[src];
        for(unsigned int dst = 0; dst < nmemtypes; ++dst)
        {
            ostr << std::setiosflags(std::ios::fixed)
                << std::setprecision(3)
                << std::setw(wid)
                << std::left
                << bwprofile[iter++];
        }
        ostr << std::endl;
    }

    std::cout << ostr.str();
}

void run(void)
{
    int count = 0;
    cudaGetDeviceCount(&count);
    if(count == 0)
    {
        std::cout << "no devices found" << std::endl;
        return;
    }
    cudaSetDevice(0);

    int deviceID = -1;
    cudaGetDevice(&deviceID);

    //
    // problem size
    unsigned int nElements = 32 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);
    // host arrays
    float *h_aPageable, *h_bPageable;
    float *h_aPinned, *h_bPinned;
    float *h_aCached, *h_bCached;
    // device array
    float *d_a, *d_b;

    //
    // allocate
    h_aPageable = (float*)malloc(bytes);                                        // host pageable
    h_bPageable = (float*)malloc(bytes);                                        // host pageable
    cudaMalloc((void**)&d_a, bytes);                                          // device
    cudaMalloc((void**)&d_b, bytes);                                          // device
    cudaMallocHost((void**)&h_aCached, bytes);                                // host cached
    cudaMallocHost((void**)&h_bCached, bytes);                                // host cached
    cudaHostAlloc((void**)&h_aPinned, bytes, cudaHostAllocWriteCombined);   // host pinned
    cudaHostAlloc((void**)&h_bPinned, bytes, cudaHostAllocWriteCombined);   // host pinned

    //
    // initialize data
    for(unsigned int i = 0; i < nElements; ++i)
        h_aPageable[i] = (float)i;

    std::memcpy(h_aPinned, h_aPageable, bytes);
    std::memcpy(h_aCached, h_aPageable, bytes);
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);
    memset(h_bCached, 0, bytes);

    //
    // output device info and transfer size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << std::endl << "Device: " << prop.name << std::endl;
    std::cout << "Transfer size (MB): " << bytes / (1024 * 1024) << std::endl;

    //
    // table
    /* TYPICAL MEMORY TYPES AND TRANSFERS    

                 | local | pageable | pinned | cached |
        ----------------------------------------------
        local    |  DtD  |             DtH
        -----------------=============================
        pageable |       ||
        pinned   |  HtD  ||            HtH
        cached   |       ||
        
        [ there also exist hw with device local and host visible memory types ]

        local   : device local
        pageable: malloc
        pinned  : host visible, coherent
        cached  : host visible, coherent, cached

        fast transfers:
        HtH : cached->pageable (there are many types of pagable->pagable, memcpy, SSE, AVX, threads, ...)
        HtD : pinned->device, pageable->device (staged)
        DtD : local->local
        HTD : local->cached
    */

    int iter = 0;
    std::vector<double> bwprofile(16, 0.0);
    std::vector<std::string> memtype = { "local", "pageable", "pinned", "cached" };

    //
    // perform copies and report bandwidth
    bool verbose = false;

    // 1. row/col
    bwprofile[iter++] = profileTransfer(d_b, d_a, nElements, cudaMemcpyDeviceToDevice, "local to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, d_a, nElements, cudaMemcpyDeviceToHost, "local to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, d_a, nElements, cudaMemcpyDeviceToHost, "local to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, d_a, nElements, cudaMemcpyDeviceToHost, "local to cached", verbose);

    // 2. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aPageable, nElements, cudaMemcpyHostToDevice, "pageable to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aPageable, nElements, cudaMemcpyHostToHost, "pageable to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aPageable, nElements, cudaMemcpyHostToHost, "pageable to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aPageable, nElements, cudaMemcpyHostToHost, "pageable to cached", verbose);

    // 3. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aPinned, nElements, cudaMemcpyHostToDevice, "pinned to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aPinned, nElements, cudaMemcpyHostToHost, "pinned to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aPinned, nElements, cudaMemcpyHostToHost, "pinned to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aPinned, nElements, cudaMemcpyHostToHost, "pinned to cached", verbose);

    // 4. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aCached, nElements, cudaMemcpyHostToDevice, "cached to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aCached, nElements, cudaMemcpyHostToHost, "cached to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aCached, nElements, cudaMemcpyHostToHost, "cached to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aCached, nElements, cudaMemcpyHostToHost, "cached to cached", verbose);

    //
    // print bandwidth resultss
    print_bwresults(memtype, bwprofile);

    std::cout << std::endl;

    //
    // cleanup
    free(h_aPageable);
    free(h_bPageable);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(h_aCached);
    cudaFree(h_bCached);
}

int main()
{
    try
    {
        run();
    }
#if !defined(__NVCC__)
    catch(vk::SystemError& err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
#endif
    catch(const std::exception& e)
    {
        std::cerr << "vuda::Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cout << "unknown error" << std::endl;
        return EXIT_FAILURE;
    }    

#if !defined(NDEBUG)
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return 0;
}