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

            host_pinned_node_internal(const size_t size, memory_allocator& allocator) : internal_node(size)
            {
                //
                // allocate            
                m_ptrMemBlock = allocator.allocate(vk::MemoryPropertyFlags(memoryPropertiesFlags::eHostInternalProperties), size);
            }
        };

    } //namespace detail
} //namespace vuda