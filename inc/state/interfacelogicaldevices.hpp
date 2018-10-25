#pragma once

namespace vuda
{

    //
    // singleton interface for vulkan logical devices
    //
    class interface_logical_devices final
    {
    public:

        static logical_device* create(const int device, const vk::DeviceQueueCreateInfo& deviceQueueCreateInfo, const vk::PhysicalDevice& physDevice)
        {
            //
            // create a logical device if it is not already in existence
            std::unique_lock<std::shared_mutex> lck(mtx());

            if(get().find(device) == get().end())
            {                
#ifdef VUDA_STD_LAYER_ENABLED
                vk::DeviceCreateInfo info({}, 1, &deviceQueueCreateInfo, 1, Instance::getValidationLayers().data(), 0, nullptr, nullptr);
#else
                vk::DeviceCreateInfo info(vk::DeviceCreateFlags(), 1, &deviceQueueCreateInfo);
#endif
                //interface_logical_devices::get().insert({ device, logical_device(info, physDevice) });
                
                // c++11
                interface_logical_devices::get().emplace(std::piecewise_construct, std::forward_as_tuple(device), std::forward_as_tuple(info, physDevice));

                // c++17
                //interface_logical_devices::get().try_emplace( device, info, physDevice );
            }

            return &get().at(device);
        }

        static logical_device* GetDevice(const int device)
        {
            std::shared_lock<std::shared_mutex> lck(mtx());
            return &get().at(device);
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

    private:
        interface_logical_devices() = default;
        ~interface_logical_devices() = default;

        // delete copy and move constructors and assign operators
        interface_logical_devices(interface_logical_devices const&) = delete;             // Copy construct
        interface_logical_devices(interface_logical_devices&&) = delete;                  // Move construct
        interface_logical_devices& operator=(interface_logical_devices const&) = delete;  // Copy assign
        interface_logical_devices& operator=(interface_logical_devices &&) = delete;      // Move assign

    };

} //namespace vuda