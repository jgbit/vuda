#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // internal host cached node
        //  - unique buffer associated (if memchunks get one buffer to rule them all, this node can be promoted to a default storage node)
        //

        class host_cached_node_internal : public internal_node
        {
        public:

            host_cached_node_internal(const size_t size, memory_allocator& allocator) : 
                internal_node(size, allocator.IsHostVisible(vudaMemoryTypes::eCached_Internal), allocator.IsHostCoherent(vudaMemoryTypes::eCached_Internal))
            {
                //
                // allocate
                m_ptrMemBlock = allocator.allocate(vudaMemoryTypes::eCached_Internal, size);
            }
        };

    } //namespace detail
} //namespace vuda