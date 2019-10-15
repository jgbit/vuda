#pragma once

namespace vuda
{
    namespace detail
    {

        class internal_node// : public default_storage_node
        {
        public:

            internal_node(const size_t size) : m_size(size), m_ptrMemBlock(nullptr)
            {
                // lock node initially            
                //m_locked.test_and_set(std::memory_order_acquire);
            }

            void set_free(void) const
            {
                //m_locked.clear(std::memory_order_release);
                m_ptrMemBlock->deallocate();
            }            
            
            bool test_and_set(void) const
            {
                //return m_locked.test_and_set(std::memory_order_acquire);
                return m_ptrMemBlock->test_and_set();
            }

            //
            //

            vk::DeviceSize get_size(void) const
            {
                return m_size;
            }

            vk::Buffer GetBuffer(void) const
            {
                return m_ptrMemBlock->get_buffer();
            }

            vk::DeviceSize GetOffset(void) const
            {
                return m_ptrMemBlock->get_offset();
            }

            void* get_memptr(void) const
            {
                return m_ptrMemBlock->get_ptr();
            }            

        protected:
            vk::DeviceSize m_size;
            memory_block* m_ptrMemBlock;

        private:

            //std::atomic_flag m_locked = ATOMIC_FLAG_INIT;
        };

        /*
        internal_buffers is a guarded list of buffers

         - host_cached_node_internal
         - host_pinned_node_internal
        */

        template <typename BufferType>
        class internal_buffers
        {
        public:

            internal_buffers(void)
            {
                m_mtx = std::make_unique<std::mutex>();
            }

            BufferType* get_buffer(const size_t size, memory_allocator& allocator)
            {
                //
                // lock
                std::lock_guard<std::mutex> lck(*m_mtx);

                //
                // find free buffer
                BufferType *hcb = nullptr;

                for(int i = 0; i < (int)m_buffers.size(); ++i)
                {
                    if(m_buffers[i]->test_and_set() == false)
                    {
                        // when the buffer is locked, the size is checked
                        if(m_buffers[i]->get_size() >= size)
                        {
                            hcb = m_buffers[i].get();
                            break;
                        }
                        else
                        {
                            m_buffers[i]->set_free();
                        }
                    }
                }

                //
                // if none are free, create a new one (potentially slow)
                if(hcb == nullptr)
                {
                    //m_buffers.push_back(std::make_unique<BufferType>(physDevice, device, size, allocator));
                    create_buffer(size, allocator);
                    //m_buffers.back()->test_and_set();
                    hcb = m_buffers.back().get();
                }

                return hcb;
            }

        private:

            void create_buffer(const size_t size, memory_allocator& allocator)
            {
                //
                // assumes that the lock m_mtx is taken                

                m_buffers.push_back(std::make_unique<BufferType>(size, allocator));
                //m_buffers.back().get()->set_free();
            }

        private:

            std::unique_ptr<std::mutex> m_mtx;
            std::vector<std::unique_ptr<BufferType>> m_buffers;
        };

    } //namespace detail
} //namespace vuda