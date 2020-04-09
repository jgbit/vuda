#pragma once

namespace vuda
{
    namespace detail
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
                m_ptrVirtual = VirtAlloc(size, m_allocatedSize);

                //
                // the memory remains mapped until it is freed by the user calling free/destroy
                // use the requested size (m_size) to specify the range in the bst instead of the full size m_allocatedSize (m_allocatedSize >= m_size)
                set_key(m_ptrVirtual, m_size);
            }

            ~device_buffer_node()
            {
            }

            /*void set_data(default_storage_node* node)
            {
                // invoke base
                default_storage_node::set_data(node);
            
                //
                // copy data
                std::cout << "UNSAFE STATIC CAST" << std::endl;
                device_buffer_node* deriv = static_cast<device_buffer_node*>(node);
                m_allocatedSize = deriv->m_allocatedSize;
                m_ptrVirtual = deriv->m_ptrVirtual;

                //
                // debug:
                std::ostringstream ostr;
                ostr << std::this_thread::get_id() << " copy data: " << m_ptrVirtual << " ";
                ostr << key() << " " << (uintptr_t)key() << " " << range() << " " << (uintptr_t)key() + range() << " (device buffer node)" << std::endl;
                std::cout << ostr.str();
            }*/

            void destroy(void) override
            {
                // invoke base
                default_storage_node::destroy();

                //
                // clean up virtual mem reservation
                VirtFree(m_ptrVirtual, m_allocatedSize);
                m_ptrVirtual = nullptr;
            }

            //
            // get functions            

            std::ostringstream print(int depth = 0) const
            {   
                std::ostringstream ostr;
                ostr << std::this_thread::get_id() << ": ";
                for(int i = 0; i < depth; ++i)
                    ostr << "-";
                ostr << key() << " " << (uintptr_t)key() << " " << range() << " " << (uintptr_t)key() + range() << " (device buffer node)" << std::endl;
                return ostr;
            }
            
        private:

            size_t m_allocatedSize;
            void* m_ptrVirtual;
        };

    } //namespace detail
} //namespace vuda