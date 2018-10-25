#pragma once

namespace vuda
{

    /*
        
    */

    class host_buffer_node : public default_storage_node
    {
    public:
        host_buffer_node(const vk::PhysicalDevice &physDevice, const vk::Device& device, const size_t size, memory_allocator& allocator) :
            default_storage_node(true),
            m_size(size)
        {
            const vk::BufferUsageFlags usage(vuda::bufferUsageFlags::eHostUsage);
            const vk::MemoryPropertyFlags properties(vuda::bufferPropertiesFlags::eHostProperties);

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

            host_buffer_node* deriv = static_cast<host_buffer_node*>(node);
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

} //namespace vuda