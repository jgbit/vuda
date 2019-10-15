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
        vk::PhysicalDevice physDevice = detail::Instance::GetPhysicalDevice(device);

        vk::PhysicalDeviceProperties deviceProperties;
        physDevice.getProperties(&deviceProperties);
                
        for(uint32_t i=0; i<256; ++i)
            prop->name[i] = deviceProperties.deviceName[i];
         
        //prop->totalGlobalMem = deviceProperties.limits.;
        //prop->sharedMemPerBlock = 0;

        for(uint32_t i=0; i<3; ++i)
        { 
            prop->maxGridSize[i] = deviceProperties.limits.maxComputeWorkGroupCount[i];
            prop->maxThreadsDim[i] = deviceProperties.limits.maxComputeWorkGroupSize[i];
        }
        prop->maxThreadsPerBlock = deviceProperties.limits.maxComputeWorkGroupInvocations;
        
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
        // get the logival device from the vuda instance        
        detail::logical_device* logicalDevice = detail::interface_logical_devices::create(physDevice, device);

        //
        // assign this particular device to the thread
        detail::interface_thread_info::insert(std::this_thread::get_id(), device, logicalDevice);

        return vudaSuccess;
    }

} //namespace vuda