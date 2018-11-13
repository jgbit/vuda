#pragma once

namespace vuda
{

    /*
        device_buffer_node holds a buffer that is backed by device local memory

        NOTE: since vuda must return a pointer to memory which can be passed to e.g. a kernel
        and vulkan does not support a natural way to get unique mem addresses to device local mem
        all device buffers are backed by a reservation of a region of pages in the virtual address space.

        This is 'unfortunately' requires an OS depedendent implementation, i.e. VirtualAlloc / mmap.
    */

    class device_buffer_node : public default_storage_node
    {
    public:
        device_buffer_node(const size_t size, memory_allocator& allocator) :
            default_storage_node(vk::MemoryPropertyFlags(memoryPropertiesFlags::eDeviceProperties), size, allocator)
        {
            //
            // reserve address range in virtual memory (platform dependent)
            m_ptrVirtual = detail::VirtAlloc(size, m_allocatedSize);

#ifdef VUDA_DEBUG_ENABLED
            if(m_ptrVirtual == nullptr)
                throw std::runtime_error("vuda: virtual memory reservation failed!");
#endif

            //
            // the memory remains mapped until it is freed by the user calling free/destroy
            // use the requested size (m_size) to specify the range in the bst instead of the full size m_allocatedSize (m_allocatedSize >= m_size)
            set_key(m_ptrVirtual, m_size);
        }

        void set_data(default_storage_node* node)
        {
            // invoke base
            default_storage_node::set_data(node);
            
            //
            // copy data
            device_buffer_node* deriv = static_cast<device_buffer_node*>(node);
            m_allocatedSize = deriv->m_allocatedSize;
            m_ptrVirtual = deriv->m_ptrVirtual;
        }

        void destroy(const vk::Device& device)
        {
            // invoke base
            default_storage_node::destroy(device);

            //
            // clean up virtual mem reservation
            int retflag = detail::VirtFree(m_ptrVirtual, m_allocatedSize);

#ifdef VUDA_DEBUG_ENABLED
            if(retflag == 0)
                throw std::runtime_error("vuda: failed to free virtual memory reservation!");
#endif
        }

        //
        // get functions        

        void* mem_ptr(void) const
        {
            return m_ptrVirtual;
        }
            
    private:        

        size_t m_allocatedSize;
        void* m_ptrVirtual;
    };

} //namespace vuda