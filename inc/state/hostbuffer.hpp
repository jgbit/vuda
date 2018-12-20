#pragma once

namespace vuda
{
    namespace detail
    {
     
        /*class host_buffer_node : public default_storage_node
        {
        public:
            host_buffer_node(const size_t size, memory_allocator& allocator) :            
                default_storage_node(vk::MemoryPropertyFlags(memoryPropertiesFlags::eHostProperties), size, allocator)
            {
            
            }
        };*/

        //
        // internal pinned host node
        //  - unique buffer associated (if memchunks get one buffer to rule them all, this node can be promoted to a default storage node)
        //

        class host_pinned_node_internal : public internal_node
        {
        public:

            host_pinned_node_internal(const size_t size, memory_allocator& allocator) :            
                m_size(size)
            {
                //
                // allocate            
                m_ptrMemBlock = allocator.allocate(vk::MemoryPropertyFlags(memoryPropertiesFlags::eHostInternalProperties), size);
            }

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

        private:

            vk::DeviceSize m_size;
            memory_block* m_ptrMemBlock;
        };

    } //namespace detail
} //namespace vuda