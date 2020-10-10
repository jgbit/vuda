
#if defined(__NVCC__)
    #include <cuda_runtime.h>
    const char* g_API = "cuda";
#else
    #if !defined(NDEBUG)
    #define VUDA_STD_LAYER_ENABLED
    #define VUDA_DEBUG_ENABLED
    #endif

    #include <vuda_runtime.hpp>
    const char* g_API = "vuda";
#endif

#include "../tools/safecall.hpp"
#include <iostream>

void query_device(void)
{   
    //
    //  When querying the number of devices and their properties it is not neccessary to call SetDevice in VUDA
    //

    int deviceCount;
    SafeCall(cudaGetDeviceCount(&deviceCount));
    std::cout << "number of " << g_API << " capable devices: " << deviceCount << std::endl << std::endl;

    for(int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp deviceProp;
        SafeCall(cudaGetDeviceProperties(&deviceProp, dev));

        std::cout << "Device " << dev << ": " << deviceProp.name << std::endl;
        std::cout << "    Total amount of global memory:                    " << (uint32_t)(deviceProp.totalGlobalMem / 1048576.0) << "MBytes (" << deviceProp.totalGlobalMem << " bytes)" << std::endl;
        std::cout << "    Total amount of shared memory per block:          " << deviceProp.sharedMemPerBlock << " bytes" << std::endl;
        
        std::cout << "    Maximum number of threads per block:              " << deviceProp.maxThreadsPerBlock << std::endl;
        std::cout << "    Max dimension size of a thread block (x,y,z):    (" << deviceProp.maxThreadsDim[0] << ", " << deviceProp.maxThreadsDim[1] << ", " << deviceProp.maxThreadsDim[2] << ")" << std::endl;
        std::cout << "    Max dimension size of a grid size    (x,y,z):    (" << deviceProp.maxGridSize[0] << ", " << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2] << ")" << std::endl;
        std::cout << "    Integrated GPU sharing Host Memory:               " << (deviceProp.integrated ? "Yes" : "No") << std::endl;
        std::cout << "    Support host page-locked memory mapping:          " << (deviceProp.canMapHostMemory ? "Yes" : "No") << std::endl;

        std::cout << "    Maximum Texture Dimension Size (x,y,z):           1D=(" << deviceProp.maxTexture1D << "), 2D=(" << deviceProp.maxTexture2D[0] << ", " << deviceProp.maxTexture2D[1] << "), 3D=(" << deviceProp.maxTexture3D[0] << ", " << deviceProp.maxTexture3D[1] << ", " << deviceProp.maxTexture3D[2] << ")" << std::endl;
        std::cout << "    Maximum Layered 1D Texture Size, (num) layers:    1D=(" << deviceProp.maxTexture1DLayered[0] << "), " << deviceProp.maxTexture1DLayered[1] << " layers" << std::endl;
        std::cout << "    Maximum Layered 2D Texture Size, (num) layers:    2D=(" << deviceProp.maxTexture2DLayered[0] << ", " << deviceProp.maxTexture2DLayered[1] << "), " << deviceProp.maxTexture2DLayered[2] << " layers" << std::endl;
                
#if defined(__NVCC__)

        std::cout << "    CUDA SPECIFICS:" << std::endl;

        const char *sComputeMode[] = {
            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
            "Unknown", nullptr };

        int driverVersion = 0, runtimeVersion = 0;
        SafeCall(cudaDriverGetVersion(&driverVersion));
        SafeCall(cudaRuntimeGetVersion(&runtimeVersion));
        std::cout << "    -- Device PCI Domain ID / Bus ID / location ID:   " << deviceProp.pciDomainID << " / " << deviceProp.pciBusID << " / " << deviceProp.pciDeviceID << std::endl;
        std::cout << "    -- CUDA Driver Version / Runtime Version:         " << driverVersion / 1000 << "." << (driverVersion % 100) / 10 << " / " << runtimeVersion / 1000 << "." << (runtimeVersion % 100) / 10 << std::endl;
        std::cout << "    -- CUDA Capability Major/Minor version number:    " << deviceProp.major << "." << deviceProp.minor << std::endl;
        #if defined(WIN32) || defined(_WIN32) || defined(WIN64) || defined(_WIN64)
        std::cout << "    -- CUDA Device Driver Mode (TCC or WDDM):         " << (deviceProp.tccDriver ? "TCC (Tesla Compute Cluster Driver)" : "WDDM (Windows Display Driver Model)") << std::endl;
        #endif        
        
        std::cout << "    -- Multiprocessors:                               " << deviceProp.multiProcessorCount << std::endl;
        std::cout << "    -- GPU Max Clock rate:                            " << deviceProp.clockRate * 1e-3f << "MHz" << "(" << deviceProp.clockRate * 1e-6f << " GHz)" << std::endl;
        std::cout << "    -- Total number of registers available per block: " << deviceProp.regsPerBlock << std::endl;
        std::cout << "    -- Maximum number of threads per multiprocessor:  " << deviceProp.maxThreadsPerMultiProcessor << std::endl;
        std::cout << "    -- Warp size:                                     " << deviceProp.warpSize << std::endl;
        std::cout << "    -- Memory Clock rate:                             " << deviceProp.memoryClockRate * 1e-3f << " Mhz" << std::endl;
        std::cout << "    -- Memory Bus Width:                              " << deviceProp.memoryBusWidth << "-bit" << std::endl;
        std::cout << "    -- Total amount of constant memory:               " << deviceProp.totalConstMem << " bytes" << std::endl;
        std::cout << "    -- Maximum memory pitch:                          " << deviceProp.memPitch << " bytes" << std::endl;
        std::cout << "    -- L2 Cache Size:                                 " << deviceProp.l2CacheSize << " bytes" << std::endl;
        std::cout << "    -- Texture alignment:                             " << deviceProp.textureAlignment << " bytes" << std::endl;        
        std::cout << "    -- Concurrent copy and kernel execution:          " << (deviceProp.deviceOverlap ? "Yes" : "No") << " with " << deviceProp.asyncEngineCount << " copy engine(s)" << std::endl; 
        std::cout << "    -- Run time limit on kernels:                     " << (deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No") << std::endl;
        std::cout << "    -- Alignment requirement for Surfaces:            " << (deviceProp.surfaceAlignment ? "Yes" : "No") << std::endl;
        std::cout << "    -- Device has ECC support:                        " << (deviceProp.ECCEnabled ? "Enabled" : "Disabled") << std::endl;
        std::cout << "    -- Device supports Unified Addressing (UVA):      " << (deviceProp.unifiedAddressing ? "Yes" : "No") << std::endl;
        std::cout << "    -- Device supports Compute Preemption:            " << (deviceProp.computePreemptionSupported ? "Yes" : "No") << std::endl;
        std::cout << "    -- Supports Cooperative Kernel Launch:            " << (deviceProp.cooperativeLaunch ? "Yes" : "No") << std::endl;
        std::cout << "    -- Supports MultiDevice Co-op Kernel Launch:      " << (deviceProp.cooperativeMultiDeviceLaunch ? "Yes" : "No") << std::endl;        
        std::cout << "    -- Compute Mode:                                  " << sComputeMode[deviceProp.computeMode] << std::endl;

#endif

        std::cout << std::endl;
    }
}

int main()
{
    try
    {
        query_device();
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