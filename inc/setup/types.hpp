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

        enum memoryPropertiesFlags {
            //eDeviceProperties = vk::MemoryPropertyFlagBits::eDeviceLocal,
            eDeviceProperties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,

            eHostProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
            eHostInternalProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,

            eCachedProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            eCachedInternalProperties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT
        };

    } //namespace detail

    //
    // CUDA equivalents

    enum enumvudaError { vudaSuccess = 0, vudaErrorNotReady = 1 };
    enum memcpyKind { memcpyHostToHost = 0, memcpyHostToDevice = 1, memcpyDeviceToHost = 2, memcpyDeviceToDevice = 3 /*, memcpyDefault = 4*/ };
    enum enumHostAlloc { hostAllocDefault = 0, hostAllocPortable = 1, hostAllocMapped = 2, hostAllocWriteCombined = 3 };

    typedef enumvudaError error_t;

    typedef vk::Event event_t;
    typedef uint32_t stream_t;

    struct dim3
    {
        uint32_t x, y, z;

        dim3(const uint32_t x) : x(x), y(1), z(1)
        {}
        dim3(const uint32_t x, const uint32_t y) : x(x), y(y), z(1)
        {}
        dim3(const uint32_t x, const uint32_t y, const uint32_t z) : x(x), y(y), z(z)
        {}
    };
    
    struct deviceProp
    {
        char name[256];
        size_t totalGlobalMem;
        size_t sharedMemPerBlock;
        int regsPerBlock;
        int warpSize;
        size_t memPitch;
        int maxThreadsPerBlock;
        int maxThreadsDim[3];
        int maxGridSize[3];
        int clockRate;
        size_t totalConstMem;
        int major;
        int minor;
        size_t textureAlignment;
        size_t texturePitchAlignment;
        int deviceOverlap;
        int multiProcessorCount;
        int kernelExecTimeoutEnabled;
        int integrated;
        int canMapHostMemory;
        int computeMode;
        int maxTexture1D;
        int maxTexture1DMipmap;
        int maxTexture1DLinear;
        int maxTexture2D[2];
        int maxTexture2DMipmap[2];
        int maxTexture2DLinear[3];
        int maxTexture2DGather[2];
        int maxTexture3D[3];
        int maxTexture3DAlt[3];
        int maxTextureCubemap;
        int maxTexture1DLayered[2];
        int maxTexture2DLayered[3];
        int maxTextureCubemapLayered[2];
        int maxSurface1D;
        int maxSurface2D[2];
        int maxSurface3D[3];
        int maxSurface1DLayered[2];
        int maxSurface2DLayered[3];
        int maxSurfaceCubemap;
        int maxSurfaceCubemapLayered[2];
        size_t surfaceAlignment;
        int concurrentKernels;
        int ECCEnabled;
        int pciBusID;
        int pciDeviceID;
        int pciDomainID;
        int tccDriver;
        int asyncEngineCount;
        int unifiedAddressing;
        int memoryClockRate;
        int memoryBusWidth;
        int l2CacheSize;
        int maxThreadsPerMultiProcessor;
        int streamPrioritiesSupported;
        int globalL1CacheSupported;
        int localL1CacheSupported;
        size_t sharedMemPerMultiprocessor;
        int regsPerMultiprocessor;
        int managedMemory;
        int isMultiGpuBoard;
        int multiGpuBoardGroupID;
        int singleToDoublePrecisionPerfRatio;
        int pageableMemoryAccess;
        int concurrentManagedAccess;
        int computePreemptionSupported;
        int canUseHostPointerForRegisteredMem;
        int cooperativeLaunch;
        int cooperativeMultiDeviceLaunch;
        int pageableMemoryAccessUsesHostPageTables;
        int directManagedMemAccessFromHost;
    };

} //namespace vuda