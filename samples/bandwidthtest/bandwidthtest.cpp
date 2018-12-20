
/*

VUDA version of bandwidthtest_cuda.cu

*/

#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include "../tools/timer.hpp"
#include <iomanip>

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    const unsigned int bytes = n * sizeof(float);

    // events for timing
    vuda::event_t startEvent, stopEvent;
    vuda::eventCreate(&startEvent);
    vuda::eventCreate(&stopEvent);

    vuda::eventRecord(startEvent, 0);
    vuda::memcpy(d, h_a, bytes, vuda::memcpyHostToDevice);
    vuda::eventRecord(stopEvent, 0);
    vuda::eventSynchronize(stopEvent);

    float time;
    vuda::eventElapsedTime(&time, startEvent, stopEvent);
    //std::cout << "  Host to Device event-time (s) : " << time * 1e-3 << std::endl;
    std::cout << "  Host to Device bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    vuda::eventRecord(startEvent, 0);
    vuda::memcpy(h_b, d, bytes, vuda::memcpyDeviceToHost);
    vuda::eventRecord(stopEvent, 0);
    vuda::eventSynchronize(stopEvent);

    vuda::eventElapsedTime(&time, startEvent, stopEvent);
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
    vuda::eventDestroy(startEvent);
    vuda::eventDestroy(stopEvent);
}

void profileCopiesQuery(float *h_a, float *h_b, float *d, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    const unsigned int bytes = n * sizeof(float);

    uint32_t startQueryID, stopQueryID;
    std::thread::id tid = std::this_thread::get_id();
    const vuda::detail::thread_info tinfo = vuda::detail::interface_thread_info::GetThreadInfo(tid);
    tinfo.GetLogicalDevice()->GetQueryID(tid, &startQueryID);
    tinfo.GetLogicalDevice()->GetQueryID(tid, &stopQueryID);
        
    tinfo.GetLogicalDevice()->WriteTimeStamp(tid, startQueryID, 0);
    vuda::memcpy(d, h_a, bytes, vuda::memcpyHostToDevice);
    tinfo.GetLogicalDevice()->WriteTimeStamp(tid, stopQueryID, 0);    
    vuda::streamSynchronize(0);
    //tinfo.GetLogicalDevice()->FlushQuery(tid, stopQueryID);

    float time;
    time = tinfo.GetLogicalDevice()->GetQueryPoolResults(tid, startQueryID, stopQueryID);
    std::cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-6 / time << std::endl;
    
    tinfo.GetLogicalDevice()->WriteTimeStamp(tid, startQueryID, 0);
    vuda::memcpy(h_b, d, bytes, vuda::memcpyDeviceToHost);
    tinfo.GetLogicalDevice()->WriteTimeStamp(tid, stopQueryID, 0);
    vuda::streamSynchronize(0);
    //tinfo.GetLogicalDevice()->FlushQuery(tid, stopQueryID);
    
    time = tinfo.GetLogicalDevice()->GetQueryPoolResults(tid, startQueryID, stopQueryID);
    std::cout << "  Device to Host bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }
}

void profileCopiesHost(float *h_a, float *h_b, float *d_a, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    double time;
    const unsigned int bytes = n * sizeof(float);

    timer.tic();
    vuda::memcpy(d_a, h_a, bytes, vuda::memcpyHostToDevice);
    vuda::streamSynchronize(0);
    time = timer.toc();
    //std::cout << "  Host to Device time      (s)   : " << time << std::endl;
    std::cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;

    timer.tic();
    vuda::memcpy(h_b, d_a, bytes, vuda::memcpyDeviceToHost);
    vuda::streamSynchronize(0);
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
    vuda::setDevice(0);

    // problem size
    unsigned int nElements = 16 * 1024 * 1024;
    const unsigned int bytes = nElements * sizeof(float);
    
    // host arrays
    float *h_aPageable, *h_bPageable;
    float *h_aPinned, *h_bPinned;

    // device array
    float *d_a;
        
    // allocate    
    h_aPageable = (float*)malloc(bytes);            // host pageable
    h_bPageable = (float*)malloc(bytes);            // host pageable
    vuda::malloc((void**)&d_a, bytes);              // device
    //vuda::mallocHost((void**)&h_aPinned, bytes);    // host pinned
    //vuda::mallocHost((void**)&h_bPinned, bytes);    // host pinned  
    vuda::hostAlloc((void**)&h_aPinned, bytes, vuda::hostAllocWriteCombined);    // host pinned
    vuda::hostAlloc((void**)&h_bPinned, bytes, vuda::hostAllocWriteCombined);    // host pinned

    // initialize data
    for(unsigned int i = 0; i < nElements; ++i)
        h_aPageable[i] = (float)i;
    
    memcpy(h_aPinned, h_aPageable, bytes);
    memset(h_bPinned, 0, bytes);
    memset(h_bPageable, 0, bytes);

    // output device info and transfer size
    vuda::deviceProp prop;
    vuda::getDeviceProperties(&prop, 0);
    std::cout << std::endl << "Device: " << prop.name << std::endl;
    std::cout << "Transfer size (MB): " << bytes / (1024 * 1024) << std::endl;
    
    // host timing
    std::cout << std::endl << "TIMING USING HOST:";
    //profileCopiesHost(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopiesHost(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopiesHost(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
        
    memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    // perform copies and report bandwidth
    std::cout << std::endl << "TIMING USING EVENTS:";
    profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopies(h_aPinned, h_bPinned, d_a, nElements, "Pinned");
        
    /*memset(h_bPageable, 0, bytes);
    memset(h_bPinned, 0, bytes);

    // query timing
    std::cout << std::endl << "TIMING USING QUERIES:";
    profileCopiesQuery(h_aPageable, h_bPageable, d_a, nElements, "Pageable");
    profileCopiesQuery(h_aPinned, h_bPinned, d_a, nElements, "Pinned");*/

    std::cout << std::endl;

    //
    // cleanup
    vuda::free(d_a);
    vuda::freeHost(h_aPinned);
    vuda::freeHost(h_bPinned);
    free(h_aPageable);
    free(h_bPageable);
}

int main()
{
    try
    {
        run();
    }
    catch(vk::SystemError err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
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

#ifndef NDEBUG
    system("pause");
#endif
    return 0;
}