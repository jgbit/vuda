#pragma once

namespace vuda
{
    namespace detail
    {

        /*
            binary tree for storage buffers
            address tree for memory lookup

            The Curiously Recurring Template Pattern (CRTP)
        */

        class default_storage_node : public internal_node, public bst_node<default_storage_node, void*>
        {
        public:

            default_storage_node(const vudaMemoryTypes& memory_type, const size_t size, memory_allocator& allocator) :
                internal_node(size, allocator.IsHostVisible(memory_type), allocator.IsHostCoherent(memory_type))
            {
                //
                // get pointer into memory chunk
                m_ptrMemBlock = allocator.allocate(memory_type, m_size);

                //
                // the memory remains mapped until it is freed by the user calling free/destroy
                set_key(m_ptrMemBlock->get_ptr(), m_ptrMemBlock->get_size());
            }

            virtual ~default_storage_node()
            {
            }

            virtual void destroy(void)
            {
                /*if(m_ptrMemBlock == nullptr)
                {
                    std::ostringstream ostr;
                    ostr << std::this_thread::get_id() << ": value of m_ptrMemBlock is: " << m_ptrMemBlock << std::endl;
                    std::cout << ostr.str();
                }*/

                //m_ptrMemBlock->deallocate();
                //m_ptrMemBlock = nullptr;
                set_free();
            }

            //
            // get
            std::ostringstream print(int depth = 0) const override
            {
                std::ostringstream ostr;
                ostr << std::this_thread::get_id() << ": ";
                for(int i = 0; i < depth; ++i)
                    ostr << "-";
                ostr << key() << " " << (uintptr_t)key() << " " << range() << " " << (uintptr_t)key() + range() << std::endl;
                return ostr;
            }
        };
    
    } //namespace detail
} //namespace vuda