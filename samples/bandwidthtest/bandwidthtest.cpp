#include <iostream>
#include <string>
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

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n, std::string desc)
{
    std::cout << std::endl << desc << " transfers" << std::endl;

    const unsigned int bytes = n * sizeof(float);

    // events for timing
    cudaEvent_t startEvent, stopEvent;
    cudaEventCreate(&startEvent);
    cudaEventCreate(&stopEvent);

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(d, h_a, bytes, cudaMemcpyHostToDevice);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    float time;
    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //std::cout << "  Host to Device event-time (s) : " << time * 1e-3 << std::endl;
    std::cout << "  Host to Device bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    cudaEventRecord(startEvent, 0);
    cudaMemcpy(h_b, d, bytes, cudaMemcpyDeviceToHost);
    cudaEventRecord(stopEvent, 0);
    cudaEventSynchronize(stopEvent);

    cudaEventElapsedTime(&time, startEvent, stopEvent);
    //std::cout << "  Device to Host event-time (ms) : " << time * 1e-3 << std::endl;
    std::cout << "  Device to Host bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {    
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }

    // clean up events
    cudaEventDestroy(startEvent);
    cudaEventDestroy(stopEvent);
}

void profileCopiesHost(float *h_a, float *h_b, float *d_a, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    double time;
    const unsigned int bytes = n * sizeof(float);

    timer.tic();
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
    cudaStreamSynchronize(0);
    time = timer.toc();
    //std::cout << "  Host to Device time      (s)   : " << time << std::endl;
    std::cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;

    timer.tic();
    cudaMemcpy(h_b, d_a, bytes, cudaMemcpyDeviceToHost);
    cudaStreamSynchronize(0);
    time = timer.toc();
    //std::cout << "  Device to Host time      (s)   : " << time << std::endl;
    std::cout << "  Device to Host bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;

    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {   
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }    
}

void run(void)
{
    // assign device to thread
    cudaSetDevice(0);    

    // problem size
    const unsigned int nElements = 16 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);
    
    // host arrays
    float *h_aPageable, *h_bPageable;
    float *h_aPinned, *h_bPinned;
    float *h_aWC, *h_bWC;

    // device array
    float *d_a;

    // allocate    
    h_aPageable = (float*)malloc(bytes);                                // host pageable
    h_bPageable = (float*)malloc(bytes);                                // host pageable
    cudaMalloc((void**)&d_a, bytes);                                    // device
    cudaMallocHost((void**)&h_aPinned, bytes);                          // host pinned
    cudaMallocHost((void**)&h_bPinned, bytes);                          // host pinned
    cudaHostAlloc((void**)&h_aWC, bytes, cudaHostAllocWriteCombined);   // write-combined
    cudaHostAlloc((void**)&h_bWC, bytes, cudaHostAllocWriteCombined);   // write-combined

    // initialize data
    for(unsigned int i = 0; i < nElements; ++i)
        h_aPageable[i] = (float)i;
    
    std::memcpy(h_aPinned, h_aPageable, bytes);

    // output device info and transfer size
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << std::endl << "Device: " << prop.name << std::endl;
    std::cout << "Transfer size (MB): " << bytes / (1024 * 1024) << std::endl;
    
    // perform copies and report bandwidth    
    std::memset(h_bPageable, 0, bytes);
    std::memset(h_bPinned, 0, bytes);
    std::memset(h_bWC, 0, bytes);
    std::cout << std::endl << "TIMING USING EVENTS:";
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aWC, h_bWC, d_a, nElements, "Write-Combined");
    // NOTE:
    // Reading pinned memory (persistently mapped) from the host tends to be slow
    // as it is not cached by the CPU, the GPU sees the memory as it is.
    // This also means that CPU writes and reads tend to be slow.
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");

    /*// host timing
    std::memset(h_bPageable, 0, bytes);
    std::memset(h_bPinned, 0, bytes);
    std::memset(h_bWC, 0, bytes);
    std::cout << std::endl << "TIMING USING HOST:";
    profileCopiesHost(h_aPageable, h_bPageable, d_a, nElements, "Pageable");    
    profileCopiesHost(h_aWC, h_bWC, d_a, nElements, "Write-Combined");
    // reading pinned memory from the host is very slow
    profileCopiesHost(h_aPinned, h_bPinned, d_a, nElements, "Pinned");*/
    
    /*
    // query timing
    std::cout << std::endl << "TIMING USING QUERIES:";
    std::memset(h_bPageable, 0, bytes);
    std::memset(h_bPinned, 0, bytes);
    std::memset(h_bWC, 0, bytes);
    profileCopiesQuery(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopiesQuery(h_aPinned, h_bPinned, d_a, nElements, "Pinned");*/

    //std::cout << std::endl;

    //
    // cleanup
    cudaFree(d_a);
    cudaFreeHost(h_aPinned);
    cudaFreeHost(h_bPinned);
    cudaFreeHost(h_aWC);
    cudaFreeHost(h_bWC);
    free(h_aPageable);
    free(h_bPageable);
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
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cout << "unknown error\n";
        return EXIT_FAILURE;
    }

#if !defined(NDEBUG)
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return 0;
}

/*
void profileCopiesQuery(float *h_a, float *h_b, float *d, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    const unsigned int bytes = n * sizeof(float);

    uint32_t startQueryID, stopQueryID;
    std::thread::id tid = std::this_thread::get_id();
    const vuda::detail::thread_info* tinfo = vuda::detail::interface_thread_info::GetThreadInfo(tid);
    tinfo->GetLogicalDevice()->GetQueryID(tid, &startQueryID);
    tinfo->GetLogicalDevice()->GetQueryID(tid, &stopQueryID);

    tinfo->GetLogicalDevice()->WriteTimeStamp(tid, startQueryID, 0);
    vuda::memcpy(d, h_a, bytes, vuda::memcpyHostToDevice);
    tinfo->GetLogicalDevice()->WriteTimeStamp(tid, stopQueryID, 0);
    vuda::streamSynchronize(0);
    //tinfo.GetLogicalDevice()->FlushQuery(tid, stopQueryID);

    float time;
    time = tinfo->GetLogicalDevice()->GetQueryPoolResults(tid, startQueryID, stopQueryID);
    std::cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-6 / time << std::endl;

    tinfo->GetLogicalDevice()->WriteTimeStamp(tid, startQueryID, 0);
    vuda::memcpy(h_b, d, bytes, vuda::memcpyDeviceToHost);
    tinfo->GetLogicalDevice()->WriteTimeStamp(tid, stopQueryID, 0);
    vuda::streamSynchronize(0);
    //tinfo.GetLogicalDevice()->FlushQuery(tid, stopQueryID);

    time = tinfo->GetLogicalDevice()->GetQueryPoolResults(tid, startQueryID, stopQueryID);
    std::cout << "  Device to Host bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }
}*/