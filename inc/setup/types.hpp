#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // vuda enums

        enum commandBufferStateFlags {
            cbReset = 0,
            cbRecording = 1,
            cbSubmitted = 2
        };

        /*enum bufferUsageFlags {
            eDeviceUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,

            eHostUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            //eHostInternalUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
            eHostInternalUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,

            eCachedUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
            //eCachedInternalUsage = VK_BUFFER_USAGE_TRANSFER_DST_BIT
            eCachedInternalUsage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        };*/

        enum bufferUsageFlags {
            eDefault = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT
        };

        enum vudaMemoryTypes {
            eDeviceLocal,
            ePinned,
            ePinned_Internal,
            eCached,
            eCached_Internal,
            eLast
        };

    } //namespace detail

    //
    // CUDA equivalents

    enum enumvudaError { vudaSuccess = 0, vudaErrorInvalidDevice = 101 };
    enum memcpyKind { memcpyHostToHost = 0, memcpyHostToDevice = 1, memcpyDeviceToHost = 2, memcpyDeviceToDevice = 3 /*, memcpyDefault = 4*/ };
    enum enumHostAlloc { hostAllocDefault = 0, hostAllocPortable = 1, hostAllocMapped = 2, hostAllocWriteCombined = 3 };

    typedef enumvudaError error_t;

    typedef vk::Event event_t;
    typedef uint32_t stream_t;
    //struct cudaUUID_t { char bytes[16]; };
    
    struct dim3
    {
        uint32_t x, y, z;

        dim3(void) : x(1), y(1), z(1)
        {}
        dim3(const uint32_t x) : x(x), y(1), z(1)
        {}
        dim3(const uint32_t x, const uint32_t y) : x(x), y(y), z(1)
        {}
        dim3(const uint32_t x, const uint32_t y, const uint32_t z) : x(x), y(y), z(z)
        {}
    };
    
    //
    // https://docs.nvidia.com/cuda/cuda-runtime-api/structcudaDeviceProp.html#structcudaDeviceProp
    struct deviceProp
    {
        char name[256];
        //char luid[8];
        //cudaUUID_t uuid;
        int integrated;

        size_t totalGlobalMem;
        size_t sharedMemPerBlock;

        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        
        int computeMode;
        int concurrentKernels;
        int deviceOverlap;
        int canMapHostMemory;
        int streamPrioritiesSupported;

        int maxSurface1D;
        int maxSurface1DLayered[2];
        int maxSurface2D[2];
        int maxSurface2DLayered[3];
        int maxSurface3D[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        int maxTexture1D;
        int maxTexture1DLayered[2];
        int maxTexture1DLinear;
        int maxTexture1DMipmap;
        int maxTexture2D[2];
        int maxTexture2DGather[2];
        int maxTexture2DLayered[3];
        int maxTexture2DLinear[3];
        int maxTexture2DMipmap[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTextureCubemapLayered[2];

        //unsigned int luidDeviceNodeMask;
        //int regsPerMultiprocessor;
        //int ECCEnabled;
        //int asyncEngineCount;        
        //int canUseHostPointerForRegisteredMem;
        //int clockRate;        
        //int computePreemptionSupported;        
        //int concurrentManagedAccess;
        //int cooperativeLaunch;
        //int cooperativeMultiDeviceLaunch;        
        //int directManagedMemAccessFromHost;
        //int globalL1CacheSupported;
        //int hostNativeAtomicSupported;        
        //int isMultiGpuBoard;
        //int kernelExecTimeoutEnabled;
        //int l2CacheSize;
        //int localL1CacheSupported;
        //int major;
        //int managedMemory;
        //int maxThreadsPerMultiProcessor;
        //size_t memPitch;
        //int memoryBusWidth;
        //int memoryClockRate;
        //int minor;
        //int multiGpuBoardGroupID;
        //int multiProcessorCount;        
        //int pageableMemoryAccess;
        //int pageableMemoryAccessUsesHostPageTables;
        //int pciBusID;
        //int pciDeviceID;
        //int pciDomainID;
        //int regsPerBlock;        
        //size_t sharedMemPerBlockOptin;
        //size_t sharedMemPerMultiprocessor;        
        //int singleToDoublePrecisionPerfRatio;        
        //size_t surfaceAlignment;
        //int tccDriver;
        //size_t textureAlignment;
        //size_t texturePitchAlignment;
        //size_t totalConstMem;        
        //int unifiedAddressing;        
        //int warpSize;
    };

} //namespace vuda