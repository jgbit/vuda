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

        default_storage_node(const vk::MemoryPropertyFlags& memory_properties, const size_t size, memory_allocator& allocator) :
            m_size(size),
            m_hostVisible(memory_properties & vk::MemoryPropertyFlagBits::eHostVisible)            
        {
            //
            // get pointer into memory chunk
            m_ptrMemBlock = allocator.allocate(memory_properties, m_size);

            //
            // the memory remains mapped until it is freed by the user calling free/destroy
            set_key(m_ptrMemBlock->get_ptr(), m_ptrMemBlock->get_size());
        }

        virtual ~default_storage_node()
        {
        }

        void set_data(default_storage_node* node)
        {
            // copy node's satellite data            
            m_size = node->m_size;
            m_ptrMemBlock = node->m_ptrMemBlock;
            m_hostVisible = node->m_hostVisible;
        }

        virtual void destroy(void)
        {
            /*if(m_ptrMemBlock == nullptr)
            {
                std::ostringstream ostr;
                ostr << std::this_thread::get_id() << ": value of m_ptrMemBlock is: " << m_ptrMemBlock << std::endl;
                std::cout << ostr.str();
            }*/

            m_ptrMemBlock->deallocate();
            m_ptrMemBlock = nullptr;
        }

        //
        // get

        void print(int depth = 0) const
        {
            std::ostringstream ostr;
            ostr << std::this_thread::get_id() << ": ";
            for(int i = 0; i < depth; ++i)
                ostr << "-";
            ostr << key() << " " << (uintptr_t)key() << " " << range() << " " << (uintptr_t)key() + range() << std::endl;
            std::cout << ostr.str();
        }

        bool isHostVisible(void) const
        {
            return m_hostVisible;
        }
        
        vk::Buffer GetBuffer(void) const
        {
            return m_ptrMemBlock->get_buffer();
        }

        vk::DeviceSize GetOffset(void) const
        {
            return m_ptrMemBlock->get_offset();
        }

        vk::DeviceSize GetSize(void) const
        {
            return m_size;
        }
    
    protected:

        //
        // memory block        
        vk::DeviceSize m_size;
        memory_block* m_ptrMemBlock;

    private:

        bool m_hostVisible;
    };

} //namespace vuda