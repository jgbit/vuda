#pragma once

namespace vuda
{

    /*
        specific retrieve functions that extends the cuda interface starts with vuda
    */
    inline vk::PhysicalDevice vudaGetPhysicalDevice(int device)
    {
        // in C++11 standard library const means thread-safe
        std::vector<vk::PhysicalDevice> physicalDevices = Instance::get()->enumeratePhysicalDevices();

        assert(device >= 0 && device < physicalDevices.size());

        /*if(device < 0 && device >= physicalDevices.size())
        {
            std::string errstr("Failed to find physical device!" + device);
            throw std::runtime_error(errstr);
        }*/

        return physicalDevices[device];
    }

    inline uint32_t vudaFindMemoryType(const vk::PhysicalDevice& device, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
    {
        vk::PhysicalDeviceMemoryProperties deviceMemoryProperties;
        device.getMemoryProperties(&deviceMemoryProperties);

        for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
        {
            if((typeFilter & (1 << i)) && (deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
            {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type!");
    }

} //namespace vuda