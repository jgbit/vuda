#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // singleton interface for vulkan logical devices
        //
        class interface_logical_devices final : public singleton
        {
        public:

            static logical_device* create(const vk::PhysicalDevice& physDevice, const int device)
            {
                std::unordered_map<int, logical_device>::iterator iter;

                //
                // return the logical device if it exists (shared lock)
                {
                    std::shared_lock<std::shared_mutex> lck(mtx());

                    iter = get().find(device);

                    if(iter != get().end())
                        return &iter->second;
                }

                //
                // create a logical device if it is not already in existence (exclusive lock)
                {                    
                    std::lock_guard<std::shared_mutex> lck(mtx());
                    
                    //
                    // get physical device
                    //const vk::PhysicalDevice& physDevice = Instance::GetPhysicalDevice(device);

                    //
                    // get the QueueFamilyProperties of the PhysicalDevice
                    std::vector<vk::QueueFamilyProperties> queueFamilyProperties = physDevice.getQueueFamilyProperties();

                    //
                    // get the first index into queueFamiliyProperties which supports compute
                    size_t computeQueueFamilyIndex = std::distance(queueFamilyProperties.begin(), std::find_if(queueFamilyProperties.begin(), queueFamilyProperties.end(), [](vk::QueueFamilyProperties const& qfp) { return qfp.queueFlags & vk::QueueFlagBits::eCompute; }));
                    assert(computeQueueFamilyIndex < queueFamilyProperties.size());

                    //
                    // HARDCODED MAX NUMBER OF STREAMS
                    const uint32_t queueCount = queueFamilyProperties[computeQueueFamilyIndex].queueCount;
                    const uint32_t queueComputeCount = queueCount;
                    const std::vector<float> queuePriority(queueComputeCount, 0.0f);
                    vk::DeviceQueueCreateInfo deviceQueueCreateInfo(vk::DeviceQueueCreateFlags(), static_cast<uint32_t>(computeQueueFamilyIndex), queueComputeCount, queuePriority.data());

                #ifdef VUDA_STD_LAYER_ENABLED
                    vk::DeviceCreateInfo info({}, 1, &deviceQueueCreateInfo, 1, Instance::getValidationLayers().data(), 0, nullptr, nullptr);
                #else
                    vk::DeviceCreateInfo info(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo);
                #endif

                    //get().insert({ device, logical_device(info, physDevice) });
                    //auto pair = get().emplace(std::piecewise_construct, std::forward_as_tuple(device), std::forward_as_tuple(info, physDevice));
                    // c++17
                    auto pair = get().try_emplace(device, info, physDevice); 
                    
                    assert(pair.second);
                    return &pair.first->second;
                }                
            }

        private:

            static std::unordered_map<int, logical_device>& get(void)
            {
                static std::unordered_map<int, logical_device> local_logical_devices;
                return local_logical_devices;
            }

            static std::shared_mutex& mtx(void)
            {
                static std::shared_mutex local_mtx;
                return local_mtx;
            }
        };

    } //namespace detail
} //namespace vuda