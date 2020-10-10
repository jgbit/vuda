
//
// vudac

#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include "../tools/timer.hpp"

/*
    compute kernels are provided as spir-v.

    1. kernels are loaded from spv files on first use. (gives a warmup overhead)
    2. kernels are embedded at compile-time into the binary.

    in vuda a pre-load shader functionality is provided
    as the kernels usually are stored in spv files.
    there will be no performance-wise gain in using this functionality
    except the load can be diverted elsewhere (e.g. start of program)
*/
#include "kernel.spv.h"

const int N = 1 << 20;

void launch_kernel(float* data)
{
    //
    // invoke kernel
#if defined(__NVCC__)
    kernel <<<1, 64 >>> (data, N);
#else
    //
    // launch kernel using the embedded binary
    vuda::launchKernel(kernel_spv, "main", 0, 1, 64, data, N);

    // from file this would look like
    //vuda::launchKernel("kernel.spv", "main", 0, 1, 64, data, N);
#endif

    //
    // debugging the kernel
#ifndef NDEBUG
    float *host = new float[N];
    vuda::memcpy(host, data, N * sizeof(float), vuda::memcpyDeviceToHost);
    for(uint32_t i = 0; i < 10; ++i)
        std::cout << host[i] << " ";
    std::cout << std::endl;
    delete[] host;
#endif
}

void launch()
{
    //
    // assign a device to the thread
    vuda::setDevice(0);

    //
    // tools
    Timer timer;
    double elapsed;
    unsigned int totalRuns = 20;

    //
    // allocate a chunk of memory
    float *data;
    vuda::malloc((void**)&data, N * sizeof(float));

    //
    std::cout << "timing kernel launches" << std::endl;
    for(unsigned int run = 0; run < totalRuns; ++run)
    {
        timer.tic();
        launch_kernel(data);
        elapsed = timer.toc();
        std::cout << "run: " << run << ", elapsed time: " << elapsed << "s" << std::endl;
    }

    //
    // cleanup
    vuda::free(data);
#if defined(__NVCC__)
    cudaDeviceReset();
#endif
}

int main()
{
    try
    {
        launch();
    }
    catch(vk::SystemError& err)
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
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return EXIT_SUCCESS;
}