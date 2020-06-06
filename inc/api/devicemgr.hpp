#pragma once

namespace vuda
{
    // Wait for compute device to finish.
    /*__host__ ​ __device__*/
    /*inline error_t deviceSynchronize(void)
    {
        std::thread::id tid = std::this_thread::get_id();
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);
        tinfo.GetLogicalDevice()->WaitOn();
        return vudaSuccess;
    }*/

    // Returns which device is currently being used.
    /*__host__ ​ __device__*/
    inline error_t getDevice(int* device)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());
        *device = tinfo->GetDeviceID();
        return vudaSuccess;
    }

    /* __host__ ​ __device__ */
    inline error_t getDeviceCount(int* count)
    {
        *count = (int)detail::Instance::GetPhysicalDeviceCount();
        return vudaSuccess;
    }

    /*__host__*/
    inline error_t getDeviceProperties(deviceProp* prop, int device)
    {
        if(device < 0 || device >= (int)detail::Instance::GetPhysicalDeviceCount())
            return vudaErrorInvalidDevice;

        vk::PhysicalDevice physDevice = detail::Instance::GetPhysicalDevice(device);

        vk::PhysicalDeviceProperties deviceProperties;
        physDevice.getProperties(&deviceProperties);

        /* vendorID
            0x1002 - AMD
            0x1010 - ImgTec
            0x10DE - NVIDIA
            0x13B5 - ARM
            0x5143 - Qualcomm
            0x8086 - INTEL
        */

        for(uint32_t i=0; i<256; ++i)
            prop->name[i] = deviceProperties.deviceName[i];

        // integrated gpu
        if(deviceProperties.deviceType == vk::PhysicalDeviceType::eIntegratedGpu)
            prop->integrated = 1;
        else
            prop->integrated = 0; // eDiscreteGpu

        // find the size of the device-local heap memory
        vuda::detail::VudaMemoryProperties deviceMemProperties(physDevice);
        prop->totalGlobalMem = (size_t)deviceMemProperties.GetDeviceLocalMemorySize();

        // maximum shared memory
        prop->sharedMemPerBlock = deviceProperties.limits.maxComputeSharedMemorySize;
        
        //
        // device threads and blocks
        prop->maxThreadsPerBlock = deviceProperties.limits.maxComputeWorkGroupInvocations;
        for(uint32_t i=0; i<3; ++i)
        { 
            prop->maxThreadsDim[i] = deviceProperties.limits.maxComputeWorkGroupSize[i];
            prop->maxGridSize[i] = deviceProperties.limits.maxComputeWorkGroupCount[i];            
        }

        prop->computeMode = 0; // cudaComputeModeDefault
        prop->canMapHostMemory = 1;
        prop->concurrentKernels = 1;
        prop->deviceOverlap = 1;
        prop->streamPrioritiesSupported = 1;

        // max texture dimensions
        prop->maxTexture1D = deviceProperties.limits.maxImageDimension1D;
        prop->maxTexture2D[0] = deviceProperties.limits.maxImageDimension2D;
        prop->maxTexture2D[1] = deviceProperties.limits.maxImageDimension2D;
        prop->maxTexture3D[0] = deviceProperties.limits.maxImageDimension3D;
        prop->maxTexture3D[1] = deviceProperties.limits.maxImageDimension3D;
        prop->maxTexture3D[2] = deviceProperties.limits.maxImageDimension3D;

        prop->maxTexture1DLayered[0] = deviceProperties.limits.maxImageDimension1D;
        prop->maxTexture1DLayered[1] = deviceProperties.limits.maxImageArrayLayers;

        prop->maxTexture2DLayered[0] = deviceProperties.limits.maxImageDimension2D;
        prop->maxTexture2DLayered[1] = deviceProperties.limits.maxImageDimension2D;
        prop->maxTexture2DLayered[2] = deviceProperties.limits.maxImageArrayLayers;

        return vudaSuccess;
    }

    // __host__
    inline error_t setDevice(int device)
    {
        //
        // lookup physical device
        // (once a logical device has been created it is redundant to lookup the physical device on calls to setDevice.
        //  for now this is done to ensure the correct creation order of the singleton instances)
        const vk::PhysicalDevice& physDevice = detail::Instance::GetPhysicalDevice(device);

        //
        // get the logical device from the vuda instance        
        detail::logical_device* logicalDevice = detail::interface_logical_devices::create(physDevice, device);

        //
        // assign this particular device to the thread
        detail::interface_thread_info::insert(std::this_thread::get_id(), device, logicalDevice);

        return vudaSuccess;
    }

} //namespace vuda