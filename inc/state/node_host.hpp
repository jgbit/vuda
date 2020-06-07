#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // internal pinned host node
        //  - unique buffer associated (if memchunks get one buffer to rule them all, this node can be promoted to a default storage node)
        //

        class host_pinned_node_internal : public internal_node
        {
        public:

            host_pinned_node_internal(const size_t size, memory_allocator& allocator) :
                internal_node(size, allocator.IsHostVisible(vudaMemoryTypes::ePinned_Internal), allocator.IsHostCoherent(vudaMemoryTypes::ePinned_Internal))
            {
                //
                // allocate
                m_ptrMemBlock = allocator.allocate(vudaMemoryTypes::ePinned_Internal, size);
            }
        };

    } //namespace detail
} //namespace vuda