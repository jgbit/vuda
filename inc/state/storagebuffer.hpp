#pragma once

namespace vuda
{

    /*
        binary tree for storage buffers
        address tree for memory lookup

        The Curiously Recurring Template Pattern (CRTP)
    */

    class default_storage_node : public bst_node<default_storage_node, void*>
    {
    public:

        default_storage_node(bool hostVisible) : m_hostVisible(hostVisible)
        {            
        }

        void set_data(default_storage_node* node)
        {
            // copy node's satellite data
            m_hostVisible = node->m_hostVisible;
        }

        bool isHostVisible(void) const
        {
            return m_hostVisible;
        }
                
        virtual void destroy(const vk::Device& device) = 0;                
        virtual vk::DeviceSize GetSize(void) const = 0;
        //virtual vk::Buffer GetBuffer(void) const = 0;
        virtual vk::Buffer GetBufferHost(void) const = 0;
        virtual vk::Buffer GetBufferDevice(void) const = 0;
        
    protected:

        vk::MemoryRequirements create(const vk::PhysicalDevice &physDevice, const vk::Device& device, const vk::DeviceSize& size, const vk::BufferUsageFlags& usage, const vk::MemoryPropertyFlags& properties, memory_allocator& allocator, vk::Buffer& buffer, memory_block*& ptrMemBlock)
        {
            //
            // create info
            // [ only exclusive right now, i.e. access from one queue family (that can still be multiple queues/streams) ]
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(size)
                .setUsage(usage)
                .setSharingMode(vk::SharingMode::eExclusive);

            //
            // create stoarge buffer
            buffer = device.createBuffer(info);

            //
            // check that memory requirements overlap with allocator
            const vk::MemoryRequirements memreq = device.getBufferMemoryRequirements(buffer);
            uint32_t memoryTypeIndex = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, properties);

            if(memoryTypeIndex != allocator.getMemoryTypeIndex())
                throw std::runtime_error("vuda: can not allocate host buffer, allocator does not have the required type bits!");

            //
            // allocate
            ptrMemBlock = allocator.allocate(memreq.size, memreq.alignment);

            //
            // bind buffer to memory
            device.bindBufferMemory(buffer, ptrMemBlock->get_memory(), ptrMemBlock->get_offset());

            //
            return memreq;
        }        
            
    private:

        bool m_hostVisible;
    };

} //namespace vuda