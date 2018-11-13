
/*

EXAMPLE BUILD ON
How to Optimize Data Transfers in CUDA C/C++
https://devblogs.nvidia.com/how-optimize-data-transfers-cuda-cc/

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

double profileTransfer(float *dst, const float *src, unsigned int n, vuda::memcpyKind kind, std::string desc, bool verbose=false)
{
    std::cout << std::endl << desc << ", ";
    if(kind == vuda::memcpyHostToHost)
        std::cout << "HtH";
    else if(kind == vuda::memcpyHostToDevice)
        std::cout << "HtD";
    else if(kind == vuda::memcpyDeviceToHost)
        std::cout << "DtH";
    else if(kind == vuda::memcpyDeviceToDevice)
        std::cout << "DtD";    
    std::cout << " transfer" << std::endl;

    Timer timer;
    double time, total=0.0;
    const unsigned int bytes = n * sizeof(float);
    const int runs = 5;

    for(int run=0; run<=runs; ++run)
    {
        timer.tic();
        vuda::memcpy(dst, src, bytes, kind, 0);
        vuda::streamSynchronize(0);
        time = timer.toc();
        if(run != 0)
            total += time;
        if(verbose)
            std::cout << "  iter: " << run << ", transfer time (s)   : " << time << std::endl;
    }

    time = total / (double)runs;
    double bw = bytes * 1e-9 / time;
    std::cout << "  avg. transfer time (s)   : " << time << std::endl;
    std::cout << "  avg. bandwidth     (GB/s): " << bw << std::endl;
    return bw;
}

void profileCopies(float *h_a, float *h_b, float *d, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    const unsigned int bytes = n * sizeof(float);

    // events for timing
    vuda::event_t startEvent, stopEvent;
    vuda::eventCreate(&startEvent);
    vuda::eventCreate(&stopEvent);

    /*//
    //
    vuda::eventRecord(startEvent, 0);
    vuda::memcpy(d, h_a, bytes, vuda::memcpyHostToDevice, 0);        
    vuda::eventRecord(stopEvent, 0);
    vuda::eventSynchronize(stopEvent);

    float time;    
    vuda::eventElapsedTime(&time, startEvent, stopEvent);    
    std::cout << "  Host to Device event-time (ms) : " << time * 1e-3 << std::endl;
    std::cout << "  Host to Device bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;

    vuda::eventRecord(startEvent, 0);
    vuda::memcpy(h_b, d, bytes, vuda::memcpyDeviceToHost);
    vuda::eventRecord(stopEvent, 0);
    vuda::eventSynchronize(stopEvent);
    
    vuda::eventElapsedTime(&time, startEvent, stopEvent);
    std::cout << "  Device to Host event-time (ms) : " << time * 1e-3 << std::endl;
    std::cout << "  Device to Host bandwidth  (GB/s): " << bytes * 1e-6 / time << std::endl;    

    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }*/

    // clean up events
    //vuda::eventDestroy(startEvent));
    //vuda::eventDestroy(stopEvent));
}

void profileCopiesHost(float *h_a, float *h_b, float *d_a, float *d_b, unsigned int n, std::string desc)
{
    Timer timer;
    std::cout << std::endl << desc << " transfers" << std::endl;

    double time;
    const unsigned int bytes = n * sizeof(float);
        
    timer.tic();
    vuda::memcpy(d_a, h_a, bytes, vuda::memcpyHostToDevice, 0);
    vuda::streamSynchronize(0);
    time = timer.toc();
    std::cout << "  Host to Device time      (s)   : " << time << std::endl;    
    std::cout << "  Host to Device bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;
    
    // intermediate internal copy
    timer.tic();
    vuda::memcpy(d_b, d_a, bytes, vuda::memcpyDeviceToDevice, 0);
    vuda::streamSynchronize(0);
    time = timer.toc();
    std::cout << "  Device to Device time    (s)   : " << time << std::endl;
    std::cout << "  Device to Device bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;

    timer.tic();    
    vuda::memcpy(h_b, d_b, bytes, vuda::memcpyDeviceToHost);
    vuda::streamSynchronize(0);
    time = timer.toc();
    std::cout << "  Device to Host time      (s)   : " << time << std::endl;
    std::cout << "  Device to Host bandwidth (GB/s): " << bytes * 1e-9 / time << std::endl;
    
    bool test = true;
    for(unsigned int i = 0; i < n; ++i)
    {
        if(h_a[i] != h_b[i])
        {
            test = false;
            std::cout << "*** " << desc << " transfers failed ***" << std::endl;
            break;
        }
    }
    if(test)
        std::cout << "  transfers completed successfully." << std::endl;
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
    vuda::getDeviceCount(&count);
    if(count == 0)
    {
        std::cout << "no devices found" << std::endl;
        return;
    }
    vuda::setDevice(0);

    int deviceID = -1;
    vuda::getDevice(&deviceID);

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
    vuda::malloc((void**)&d_a, bytes);                                          // device
    vuda::malloc((void**)&d_b, bytes);                                          // device
    vuda::mallocHost((void**)&h_aPinned, bytes);                                // host pinned
    vuda::mallocHost((void**)&h_bPinned, bytes);                                // host pinned
    vuda::hostAlloc((void**)&h_aCached, bytes, vuda::hostAllocWriteCombined);   // host cached
    vuda::hostAlloc((void**)&h_bCached, bytes, vuda::hostAllocWriteCombined);   // host cached

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
    vuda::deviceProp prop;
    vuda::getDeviceProperties(&prop, 0);
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
    bwprofile[iter++] = profileTransfer(d_b, d_a, nElements, vuda::memcpyDeviceToDevice, "local to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, d_a, nElements, vuda::memcpyDeviceToHost, "local to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, d_a, nElements, vuda::memcpyDeviceToHost, "local to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, d_a, nElements, vuda::memcpyDeviceToHost, "local to cached", verbose);

    // 2. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aPageable, nElements, vuda::memcpyHostToDevice, "pageable to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aPageable, nElements, vuda::memcpyHostToHost, "pageable to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aPageable, nElements, vuda::memcpyHostToHost, "pageable to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aPageable, nElements, vuda::memcpyHostToHost, "pageable to cached", verbose);

    // 3. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aPinned, nElements, vuda::memcpyHostToDevice, "pinned to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aPinned, nElements, vuda::memcpyHostToHost, "pinned to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aPinned, nElements, vuda::memcpyHostToHost, "pinned to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aPinned, nElements, vuda::memcpyHostToHost, "pinned to cached", verbose);

    // 4. row/col
    bwprofile[iter++] = profileTransfer(d_b, h_aCached, nElements, vuda::memcpyHostToDevice, "cached to local", verbose);
    bwprofile[iter++] = profileTransfer(h_bPageable, h_aCached, nElements, vuda::memcpyHostToHost, "cached to pageable", verbose);
    bwprofile[iter++] = profileTransfer(h_bPinned, h_aCached, nElements, vuda::memcpyHostToHost, "cached to pinned", verbose);
    bwprofile[iter++] = profileTransfer(h_bCached, h_aCached, nElements, vuda::memcpyHostToHost, "cached to cached", verbose);

    //
    // print bandwidth resultss
    print_bwresults(memtype, bwprofile);

    //
    // classical bandwidthtest (timing using host)
    title("Classical bandwidth test (timing using host)");
    {
        memset(h_bPageable, 0, bytes);
        profileCopiesHost(h_aPageable, h_bPageable, d_a, d_b, nElements, "Host timing, Pageable");
        memset(h_bPinned, 0, bytes);
        profileCopiesHost(h_aPinned, h_bPinned, d_a, d_b, nElements, "Host timing, Pinned");
    }

    // timing using events
    /*title("Classical bandwidth test (timing using events)");
    {
        memset(h_bPageable, 0, bytes);
        profileCopies(h_aPageable, h_bPageable, d_a, nElements, "Event timing, Pageable");
        //memset(h_bPinned, 0, bytes);
        //profileCopies(h_aPinned, h_bPinned, d_a, d_b, nElements, "Event timing, Pinned");
    }*/

    std::cout << std::endl;

    //
    // cleanup    
    free(h_aPageable);
    free(h_bPageable);
    vuda::freeHost(h_aPinned);
    vuda::freeHost(h_bPinned);    
    vuda::free(d_a);
    vuda::free(d_b);
    vuda::free(h_aCached);
    vuda::free(h_bCached);
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

    /*std::cout << "done" << std::endl;
    std::cin.get();*/
    //system("pause");
    return 0;
}