#pragma once

namespace vuda
{

    /*
        device buffer holds a buffer that is backed by device local memory

        NOTE: since vuda must return a pointer to memory which can be passed to e.g. a kernel
        and vulkan does not support a natural way to get unique mem addresses to device local mem
        all device buffers are backed by a equalently sized host visible (coherent) buffer.
        
        !!! This is an abysmal approach (hopefully temporary) to ensure that the memory space is not violated.
    */

    class device_buffer_node : public default_storage_node
    {
    public:
        device_buffer_node(const vk::PhysicalDevice &physDevice, const vk::Device& device, const size_t size, memory_allocator& device_alloctor, memory_allocator& host_alloctor) :
            default_storage_node(false),
            m_size(size)
        {
            const vk::BufferUsageFlags usageDevice(vuda::bufferUsageFlags::eDeviceUsage);
            const vk::MemoryPropertyFlags propertiesDevice(vuda::bufferPropertiesFlags::eDeviceProperties);            
            const vk::BufferUsageFlags usageHost(vuda::bufferUsageFlags::eHostUsage);
            const vk::MemoryPropertyFlags propertiesHost(vuda::bufferPropertiesFlags::eHostProperties);
            
            //
            // local device buffer
            //vk::MemoryRequirements memreq = 
            create(physDevice, device, size, usageDevice, propertiesDevice, device_alloctor, m_buffer, m_ptrMemBlock);

            //
            // abysmal approach (hopefully temporary) to ensure that the memory space is not violated.
            create(physDevice, device, size, usageHost, propertiesHost, host_alloctor, m_bufferHost, m_ptrMemBlockHost);
            //m_ptrMemBlockHost = host_alloctor.allocate(memreq.size, memreq.alignment);

            //
            // the memory remains mapped until it is freed by the user calling free/destroy
            set_key(m_ptrMemBlockHost->get_ptr(), m_ptrMemBlockHost->get_size());
        }

        void set_data(default_storage_node* node)
        {
            // invoke base
            default_storage_node::set_data(node);
            // copy data
            device_buffer_node* deriv = static_cast<device_buffer_node*>(node);
            
            m_buffer = deriv->m_buffer;
            m_bufferHost = deriv->m_bufferHost;
            m_ptrMemBlock = deriv->m_ptrMemBlock;
            m_ptrMemBlockHost = deriv->m_ptrMemBlockHost;
        }

        void destroy(const vk::Device& device)
        {
            device.destroyBuffer(m_buffer);
            device.destroyBuffer(m_bufferHost);

            // return memory to pool
            m_ptrMemBlock->deallocate();
            m_ptrMemBlockHost->deallocate();
        }

        //
        // get functions        

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
            return m_bufferHost;
        }

        vk::Buffer GetBufferDevice(void) const
        {
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

        //
        // memory pointer        
        vk::Buffer m_bufferHost;
        memory_block* m_ptrMemBlockHost;
    };

} //namespace vuda