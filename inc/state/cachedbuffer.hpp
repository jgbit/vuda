#pragma once

namespace vuda
{
    namespace detail
    {

        /*class host_cached_node : public default_storage_node
        {
        public:
            host_cached_node(const size_t size, memory_allocator& allocator) :            
                default_storage_node(vk::MemoryPropertyFlags(memoryPropertiesFlags::eCachedProperties), size, allocator)
            {
                //
                // the memory remains mapped until it is freed by the user calling free/destroy            
                set_key(m_ptrMemBlock->get_ptr(), m_ptrMemBlock->get_size());
            }
        };*/

        //
        // internal host cached node
        //  - unique buffer associated (if memchunks get one buffer to rule them all, this node can be promoted to a default storage node)
        //

        class host_cached_node_internal : public internal_node
        {
        public:

            host_cached_node_internal(const size_t size, memory_allocator& allocator) : internal_node(size)
            {
                //
                // allocate
                m_ptrMemBlock = allocator.allocate(vk::MemoryPropertyFlags(memoryPropertiesFlags::eCachedInternalProperties), size);
            }
        };

    } //namespace detail
} //namespace vuda