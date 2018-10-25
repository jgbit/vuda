#pragma once

namespace vuda
{
    /*
    
    */
    class host_cached_node : public default_storage_node
    {
    public:
        host_cached_node(const vk::PhysicalDevice &physDevice, const vk::Device& device, const size_t size, memory_allocator& allocator) :
            default_storage_node(true),
            m_size(size)
        {
            const vk::BufferUsageFlags usage(vuda::bufferUsageFlags::eCachedUsage);
            const vk::MemoryPropertyFlags properties(vuda::bufferPropertiesFlags::eCachedProperties);

            //
            // create buffer
            create(physDevice, device, size, usage, properties, allocator, m_buffer, m_ptrMemBlock);

            //
            // the memory remains mapped until it is freed by the user calling free/destroy            
            set_key(m_ptrMemBlock->get_ptr(), m_ptrMemBlock->get_size());
        }

        void set_data(default_storage_node* node)
        {
            // invoke base
            default_storage_node::set_data(node);

            host_cached_node* deriv = static_cast<host_cached_node*>(node);
            m_buffer = deriv->m_buffer;
            m_ptrMemBlock = deriv->m_ptrMemBlock;
        }

        void destroy(const vk::Device& device)
        {
            device.destroyBuffer(m_buffer);

            // return memory to pool
            m_ptrMemBlock->deallocate();
        }

        void print(int depth = 0) const
        {
            std::ostringstream ostr;
            ostr << std::this_thread::get_id() << ": ";
            for(int i = 0; i < depth; ++i)
                ostr << "-";
            ostr << key() << " " << (uintptr_t)key() << " " << range() << " " << (uintptr_t)key() + range() << std::endl;
            std::cout << ostr.str();
        }

        vk::Buffer GetBufferHost(void) const
        {
            return m_buffer;
        }

        vk::Buffer GetBufferDevice(void) const
        {
            assert(0);
            return m_buffer;
        }

        vk::DeviceSize GetSize(void) const
        {
            return m_size;
        }

    private:

        vk::DeviceSize m_size;

        //
        // buffer and memory block
        vk::Buffer m_buffer;
        memory_block* m_ptrMemBlock;
    };

    //
    // internal host cached node

    class host_cached_node_internal// : public default_storage_node
    {
    public:

        host_cached_node_internal(const vk::PhysicalDevice& physDevice, const vk::Device& device, const size_t size, memory_allocator& allocator) :
            //default_storage_node(true),
            m_free(false), // lock node initially
            m_size(size)
        {
            const vk::BufferUsageFlags usage(vuda::bufferUsageFlags::eCachedInternalUsage);
            const vk::MemoryPropertyFlags properties(vuda::bufferPropertiesFlags::eCachedInternalProperties);

            //
            // create stoarge buffer
            vk::BufferCreateInfo info = vk::BufferCreateInfo()
                .setSize(size)
                .setUsage(usage)
                .setSharingMode(vk::SharingMode::eExclusive);

            m_buffer = device.createBufferUnique(info);

            //
            // check that memory requirements overlap with allocator
            vk::MemoryRequirements memreq = device.getBufferMemoryRequirements(m_buffer.get());
            uint32_t memoryTypeIndex = vudaFindMemoryType(physDevice, memreq.memoryTypeBits, properties);

            //
            // typebit filter check
            if(memoryTypeIndex != allocator.getMemoryTypeIndex())
                throw std::runtime_error("vuda: can not allocate host buffer, allocator does not have the required type bits!");

            //
            // allocate
            m_ptrMemBlock = allocator.allocate(memreq.size, memreq.alignment);

            //
            // bind buffer to memory
            device.bindBufferMemory(m_buffer.get(), m_ptrMemBlock->get_memory(), m_ptrMemBlock->get_offset());
        }
        
        void set_free(bool value)
        {
            m_free = value;
        }
        bool is_free(void) const
        {
            return m_free;
        }

        vk::DeviceSize get_size(void) const
        {
            return m_size;
        }

        vk::Buffer get_buffer(void) const
        {
            return m_buffer.get();
        }

        void* get_memptr(void) const
        {
            return m_ptrMemBlock->get_ptr();
        }

    private:

        std::atomic<bool> m_free;
        vk::DeviceSize m_size;
            
        vk::UniqueBuffer m_buffer;        
        memory_block* m_ptrMemBlock;
    };

} //namespace vuda