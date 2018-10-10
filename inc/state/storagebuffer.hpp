#pragma once

#include <thread>
#include <shared_mutex>

namespace vuda
{

    /*
        binary tree for storage buffers
        address tree for memory lookup

        The Curiously Recurring Template Pattern (CRTP)
    */
    
    class storage_buffer_node : public bst_node<storage_buffer_node, void*>
    {
    public:
        storage_buffer_node(const vk::PhysicalDevice &physDevice, const vk::UniqueDevice& device, size_t size) :            
            m_size(size)
        {
            init(physDevice, device);            
        }

        void print(int depth = 0) const
        {
            std::ostringstream ostr;
            ostr << std::this_thread::get_id() << ": ";
            for(int i = 0; i < depth; ++i)
                ostr << "-";
            ostr << m_key << " " << (uintptr_t)m_key << std::endl;
            std::cout << ostr.str();
        }

        void set_data(storage_buffer_node* node)
        {
            // copy node's satellite data
            m_bufferHost = node->m_bufferHost;
            m_bufferDevice = node->m_bufferDevice;
            
            m_size = node->m_size;
            m_memoryHost = node->m_memoryHost;
            m_memoryDevice = node->m_memoryDevice;
        }

        void destroy(const vk::UniqueDevice& device)
        {
            //
            // host objects
            device->unmapMemory(m_memoryHost);
            device->freeMemory(m_memoryHost, nullptr);
            device->destroyBuffer(m_bufferHost);

            //
            // device objects            
            device->freeMemory(m_memoryDevice, nullptr);
            device->destroyBuffer(m_bufferDevice);
        }

        vk::Buffer GetBufferHost(void) const
        {
            return m_bufferHost;
        }

        vk::Buffer GetBufferDevice(void) const
        {
            return m_bufferDevice;
        }

        vk::DeviceSize GetSize(void) const
        {
            return m_size;
        }

        /*void copyBufferToDevice(const vk::UniqueCommandBuffer& commandBuffer, const vk::DeviceSize size) const
        {
            
        }*/

    private:

        void init(const vk::PhysicalDevice &physDevice, const vk::UniqueDevice& device)
        {        
            //
            // staging buffer
            create(physDevice, device, m_size,
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst, // | vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                m_bufferHost, m_memoryHost);

            //
            // local device buffer
            create(physDevice, device, m_size,
                vk::BufferUsageFlagBits::eTransferSrc | vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eStorageBuffer,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                m_bufferDevice, m_memoryDevice);

            //
            // the memory remains mapped until it is freed by the user calling free/destroy      
            m_key = device->mapMemory(m_memoryHost, 0, m_size);
        }

        void create(const vk::PhysicalDevice &physDevice, const vk::UniqueDevice& device, const vk::DeviceSize& size, const vk::BufferUsageFlags& usage, const vk::MemoryPropertyFlags& properties, vk::Buffer& buffer, vk::DeviceMemory& memory)
        {

            vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo()
                .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
                .setPInheritanceInfo(nullptr);

            //
            // create info
            // [ only exclusive right now, i.e. access from one queue family (that can still be multiple queues/streams) ]
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(size)
                .setUsage(usage)
                .setSharingMode(vk::SharingMode::eExclusive);

            //
            // create stoarge buffer
            buffer = device->createBuffer(info);

            //
            // find suitable memory on the physcial device
            const VkMemoryRequirements memreq = device->getBufferMemoryRequirements(buffer);
            uint32_t memoryTypeIndex = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, properties);

            //
            // allocate the memory        
            memory = device->allocateMemory(vk::MemoryAllocateInfo(memreq.size, memoryTypeIndex));

            //
            // bind memory
            device->bindBufferMemory(buffer, memory, 0);
        }

    private:

        //
        // buffers
        vk::Buffer m_bufferHost;
        vk::Buffer m_bufferDevice;

        //
        // allocated memory for buffers
        size_t m_size;        
        vk::DeviceMemory m_memoryHost;
        vk::DeviceMemory m_memoryDevice;
    };

} //namespace vuda
