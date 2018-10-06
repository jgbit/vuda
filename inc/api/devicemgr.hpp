#pragma once

namespace vuda
{
    // __host__ ​ __device__ 
    inline vudaError_t GetDeviceCount(int* count)
    {
        vk::UniqueInstance& inst = Instance::get();
        *count = (int)inst->enumeratePhysicalDevices().size();
        return vudaSuccess;
    }

    // __host__
    inline void SetDevice(int device)
    {
        vk::PhysicalDevice physDevice = vudaGetPhysicalDevice(device);

        // get the QueueFamilyProperties of the PhysicalDevice
        std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physDevice.getQueueFamilyProperties();

        // get the first index into queueFamiliyProperties which supports compute
        size_t computeQueueFamilyIndex = std::distance(queueFamilyProperties.begin(), std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](vk::QueueFamilyProperties const& qfp) { return qfp.queueFlags & vk::QueueFlagBits::eCompute; }));
        assert(computeQueueFamilyIndex < queueFamilyProperties.size());

        //
        // create a UniqueDevice

        //
        // HARDCODED MAX NUMBER OF STREAMS
        const uint32_t queueCount = queueFamilyProperties[computeQueueFamilyIndex].queueCount;
        const uint32_t queueComputeCount = 2;
        const float queuePriority[queueComputeCount] = {};
        vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(computeQueueFamilyIndex), queueComputeCount, queuePriority);

        //
        // create or get the logical device associated with the device id
        logical_device* logicalDevice = interface_logical_devices::create(device, deviceQueueCreateInfo, physDevice);

        //
        // assign this particular device to the thread
        interface_thread_info::insert(std::this_thread::get_id(), device, logicalDevice);
    }

    //__host__ ​
    inline vudaError_t GetDeviceProperties(vudaDeviceProp* prop, int device)
    {
        vk::PhysicalDevice physDevice = vudaGetPhysicalDevice(device);

        vk::PhysicalDeviceProperties deviceProperties;
        physDevice.getProperties(&deviceProperties);
                
        for(uint32_t i = 0; i<256; ++i)
            prop->name[i] = deviceProperties.deviceName[i];
         
        //prop->totalGlobalMem = deviceProperties.limits.;
        //prop->sharedMemPerBlock = 0;

        for(uint32_t i = 0; i<3; ++i)
        { 
            prop->maxGridSize[i] = deviceProperties.limits.maxComputeWorkGroupCount[i];
            prop->maxThreadsDim[i] = deviceProperties.limits.maxComputeWorkGroupSize[i];
        }
        prop->maxThreadsPerBlock = deviceProperties.limits.maxComputeWorkGroupInvocations;
        
        return vudaSuccess;
    }

} //namespace vuda