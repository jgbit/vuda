#pragma once

namespace vuda
{
    namespace detail
    {

        /*
            specific retrieve functions that extends the cuda interface starts with vuda        
        */

        /*
            https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
        */
        inline int32_t vudaFindMemoryType(const vk::PhysicalDevice& device, uint32_t typeFilter, vk::MemoryPropertyFlags properties)
        {
            vk::PhysicalDeviceMemoryProperties deviceMemoryProperties;
            device.getMemoryProperties(&deviceMemoryProperties);

            /*
                https://developer.nvidia.com/what%E2%80%99s-your-vulkan-memory-type
                on nvidia-hw, the seven memory types with mask=0
                    - (~1) a memory type for buffers 
                    - (~1) A memory type for color images of any format 
                    - (~5) separate memory types for depth/stencil images for each depth/stencil format in system memory

                However, there are stil two device memory types.
                - these are applicable for different typefilters (i.e. different kind of buffers)            
            */

            // (memRequirements.memoryTypeBits >> memoryTypeIndex) & 1)

            for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
            {
                if((typeFilter & (1 << i)) && (deviceMemoryProperties.memoryTypes[i].propertyFlags & properties) == properties)
                {
                    return i;
                }
            }

            return -1;
            //throw std::runtime_error("Failed to find suitable memory type!");
        }

        inline uint32_t vudaGetNumberOfMemoryTypes(const vk::PhysicalDevice& device, std::vector<uint32_t>& memoryIndices)
        {
            vk::PhysicalDeviceMemoryProperties deviceMemoryProperties;
            device.getMemoryProperties(&deviceMemoryProperties);

            uint32_t count = 0;
            for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
            {
                // mask is non-zero
                if(deviceMemoryProperties.memoryTypes[i].propertyFlags)
                {
                    memoryIndices.push_back(i);
                    ++count;
                }
            }

            return count;
        }

        inline vk::DeviceSize findDeviceLocalMemorySize(const vk::PhysicalDevice& physDevice)
        {
            vk::PhysicalDeviceMemoryProperties deviceMemoryProperties;
            physDevice.getMemoryProperties(&deviceMemoryProperties);

            for(uint32_t i = 0; i < deviceMemoryProperties.memoryTypeCount; i++)
            {
                if((deviceMemoryProperties.memoryTypes[i].propertyFlags & vk::MemoryPropertyFlagBits::eDeviceLocal) == vk::MemoryPropertyFlagBits::eDeviceLocal)
                {
                    uint32_t heap = deviceMemoryProperties.memoryTypes[i].heapIndex;
                    return deviceMemoryProperties.memoryHeaps[heap].size;
                }
            }

            throw std::runtime_error("vuda: there must be at least one memory type with the VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT bit set in its propertyFlags!");
        }

        inline uint32_t findMemoryType_Device(const vk::PhysicalDevice& physDevice, const vk::Device& device)
        {
            const std::vector<vk::MemoryPropertyFlags>& candidates = {
                vk::MemoryPropertyFlags(memoryPropertiesFlags::eDeviceProperties),
                vk::MemoryPropertyFlagBits::eDeviceLocal
            };

            //
            // create temporary buffer
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(1)
                .setUsage(vk::BufferUsageFlags(bufferUsageFlags::eDefault))
                .setSharingMode(vk::SharingMode::eExclusive);
            vk::UniqueBuffer buffer = device.createBufferUnique(info);
            const VkMemoryRequirements memreq = device.getBufferMemoryRequirements(buffer.get());

            for(const vk::MemoryPropertyFlags& type : candidates)
            {
                int32_t index = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, type);
                if(index != -1)
                    return index;
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        inline uint32_t findMemoryType_Host(const vk::PhysicalDevice& physDevice, const vk::Device& device)
        {
            const std::vector<vk::MemoryPropertyFlags>& candidates = {
                vk::MemoryPropertyFlags(memoryPropertiesFlags::eHostProperties),
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            };

            //
            // create temporary buffer
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(1)
                .setUsage(vk::BufferUsageFlags(bufferUsageFlags::eDefault))
                .setSharingMode(vk::SharingMode::eExclusive);
            vk::UniqueBuffer buffer = device.createBufferUnique(info);
            const VkMemoryRequirements memreq = device.getBufferMemoryRequirements(buffer.get());

            for(const vk::MemoryPropertyFlags& type : candidates)
            {
                int32_t index = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, type);
                if(index != -1)
                    return index;
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

        inline uint32_t findMemoryType_Cached(const vk::PhysicalDevice& physDevice, const vk::Device& device)
        {
            /*
                https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkPhysicalDeviceMemoryProperties.html
                There must be at least one memory type with both the VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT and VK_MEMORY_PROPERTY_HOST_COHERENT_BIT bits set in its propertyFlags.
                There must be at least one memory type with the VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT bit set in its propertyFlags.
            */

            const std::vector<vk::MemoryPropertyFlags>& candidates = { 
                vk::MemoryPropertyFlags(memoryPropertiesFlags::eCachedProperties),
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent
            };

            //
            // create temporary buffer
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(1)
                .setUsage(vk::BufferUsageFlags(bufferUsageFlags::eDefault))
                .setSharingMode(vk::SharingMode::eExclusive);
            vk::UniqueBuffer buffer = device.createBufferUnique(info);
            const VkMemoryRequirements memreq = device.getBufferMemoryRequirements(buffer.get());

            for(const vk::MemoryPropertyFlags& type : candidates)
            {
                int32_t index = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, type);
                if(index != -1)
                    return index;
            }

            throw std::runtime_error("Failed to find suitable memory type!");
        }

    } //namespace detail
} //namespace vuda