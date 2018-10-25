#pragma once


namespace vuda
{

    //
    // memory_block is a suballocation of a memory_chunk
    class memory_block
    {
    public:

        memory_block(const vk::DeviceSize offset, const vk::DeviceSize size, const vk::DeviceMemory& mem) : m_free(true), m_offset(offset), m_size(size), m_ptr(nullptr), m_memory(mem)
        {
        }

        //
        //
        void allocate(const vk::DeviceSize offset, const vk::DeviceSize size, void* ptr)
        {
            m_free = false;
            m_offset = offset;
            m_size = size;
            m_ptr = ptr;
        }

        void deallocate(void)
        {
            //
            // m_free is atomic so no need to guard            
            m_free = true;
        }

        //
        //

        bool is_free(void) const
        {
            return m_free;
        }

        vk::DeviceSize get_offset(void) const
        {
            return m_offset;
        }

        vk::DeviceSize get_size(void) const
        {
            return m_size;
        }

        void* get_ptr(void) const
        {
            return m_ptr;
        }

        vk::DeviceMemory get_memory(void) const
        {
            return m_memory;
        }

    private:
        std::atomic<bool> m_free;
        vk::DeviceSize m_offset;
        vk::DeviceSize m_size;
        
        void* m_ptr;
        vk::DeviceMemory m_memory;
    };

    //
    //    
    class memory_chunk
    {
    public:

        /*
            public synchronized interface
        */

        memory_chunk(const vk::Device& device, const bool hostVisible, const uint32_t memoryTypeIndex, const vk::DeviceSize size) :            
            m_size(size),

            //
            // allocate chunck of memory            
            m_memory(device.allocateMemoryUnique(vk::MemoryAllocateInfo(m_size, memoryTypeIndex))),

            //
            // map memory            
            m_ptr(hostVisible ? device.mapMemory(m_memory.get(), 0, VK_WHOLE_SIZE) : nullptr)
        {
            //
            //
            m_mtxBlocks = std::make_unique<std::mutex>();

            //
            // create initial block
            m_blocks.emplace_back(std::make_unique<memory_block>(0, m_size, m_memory.get()));
        }

        //
        // tries to find a free block within the chunk of memory with the required size
        // returns true if this succeeds
        // returns false if this fails
        memory_block* allocate(const vk::DeviceSize size, const vk::DeviceSize alignment)
        {
            if(size > m_size)
                return nullptr;

            //
            // find a free block that has sufficiently range

            for(size_t i = 0; i < m_blocks.size(); ++i)
            {   
                //
                // first find, first serve
                if(m_blocks[i]->is_free())
                {
                    std::unique_lock<std::mutex> lck(*m_mtxBlocks);

                    //
                    // available size after alignment
                    vk::DeviceSize virtualSize = m_blocks[i]->get_size();
                    vk::DeviceSize rest = (vk::DeviceSize)m_blocks[i]->get_offset() % alignment;
                    vk::DeviceSize incr_offset = 0;

                    if(rest != 0)
                    {
                        incr_offset = alignment - rest;
                        virtualSize -= incr_offset;
                    }   

                    // if the block can fit, (push mem pointer)
                    if(virtualSize >= size)
                    {   
                        vk::DeviceSize offset = (vk::DeviceSize)m_blocks[i]->get_offset() + incr_offset;

                        //
                        // create a new block for the remaining memory
                        if(virtualSize != size)
                        {
                            //
                            // create new block
                            m_blocks.emplace_back(std::make_unique<memory_block>(offset + size, virtualSize - size, m_memory.get()));                            
                        }

                        // if host visible push the memory pointer
                        void* ptr = nullptr;
                        if(m_ptr != nullptr)
                             ptr = (char*)m_ptr + offset;
                        
                        m_blocks[i]->allocate(offset, size, ptr);
                        return m_blocks[i].get();
                    }
                }            
            }
            return nullptr;
        }

    private:

        //
        // memory
        vk::DeviceSize m_size;        
        vk::UniqueDeviceMemory m_memory;        
        void *m_ptr;

        //
        // blocks in the chunk
        std::unique_ptr<std::mutex> m_mtxBlocks;
        std::vector<std::unique_ptr<memory_block>> m_blocks;
    };

    class memory_allocator
    {
    public:

        /*
            public synchronized interface
        */

        memory_allocator(const vk::Device& device, const bool hostVisible, const uint32_t memoryTypeIndex, const vk::DeviceSize default_alloc_size = (vk::DeviceSize)1 << 20) :
            m_device(device), 
            m_hostVisible(hostVisible),
            m_memoryTypeIndex(memoryTypeIndex),
            m_defaultChunkSize(default_alloc_size)
        {
            //
            // set default chunk size
            //set_default_alloc_size(default_alloc_size); // 1 mebibyte

            //
            // create initial chunck
            //m_chunks.emplace_back(m_device, m_memoryTypeIndex, m_defaultChunkSize);
        }        

        memory_block* allocate(const vk::DeviceSize size, const vk::DeviceSize alignment)
        {
            memory_block* block;

            //
            // find a suitable chunk that has a free chunk with suitable size             
            {
                std::shared_lock<std::shared_mutex> lck(m_mtxChunks);

                for(auto& chunk : m_chunks)
                {
                    // all chunks have the same memory type

                    //
                    // attempt to retrieve block from chunk
                    block = chunk.allocate(size, alignment);
                    if(block != nullptr)
                        return block;
                }
            }            

            //
            // create new memory chunk
            {
                std::unique_lock<std::shared_mutex> lck(m_mtxChunks);

                vk::DeviceSize used_size = m_defaultChunkSize;
                if(size > m_defaultChunkSize)
                {
                    // increase default allocation size
                    used_size = size;
                }

                m_chunks.emplace_back(m_device, m_hostVisible, m_memoryTypeIndex, used_size);

                //
                // retrieve block from chunk ( if this fails it is an implementation fault )
                block = m_chunks.back().allocate(size, alignment);
                assert(block);
            }

            return block;
        }

        uint32_t getMemoryTypeIndex(void) const
        {
            return m_memoryTypeIndex;
        }

    private:

        /*
            private non-synchronized implementation        
        */

        /*void set_default_alloc_size(const vk::DeviceSize default_alloc_size)
        {
            m_defaultChunkSize = default_alloc_size;
        }*/

    private:

        //
        // keep memory allocator associated to one device
        vk::Device m_device;

        //
        // [ associating each memory allocator with one specific memory type we can shorten the chunck search 
        //   and make it more explicit what type of memory a buffer uses
        //   however an additional allocator has to be created for each type of memory]
        //
        // memory information
        const bool m_hostVisible;
        uint32_t m_memoryTypeIndex;
        

        //
        // memory allocation settings
        vk::DeviceSize m_defaultChunkSize;

        //
        // chunks of memory        
        std::shared_mutex m_mtxChunks;
        std::vector<memory_chunk> m_chunks;
    };

} // namespace vuda