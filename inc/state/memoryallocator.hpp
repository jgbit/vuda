#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // memory_block is a suballocation of a memory_chunk
        class memory_block
        {
        public:

            // allocate
            memory_block(const vk::DeviceMemory memory, const vk::DeviceSize offset, const vk::DeviceSize size, const vk::Buffer& buffer) : m_memory(memory), m_offset(offset), m_size(size), m_ptr(nullptr), m_buffer(buffer)
            {
            }

            //
            //

            bool test_and_set(void)
            {
                return m_locked.test_and_set(std::memory_order_acquire);
            }

            void reallocate(const vk::DeviceSize offset, const vk::DeviceSize size, void* ptr)
            {
                // before calling allocate test_and_set must have been called
                m_offset = offset;
                m_size = size;
                m_ptr = ptr;
            }

            void deallocate(void)
            {
                /*std::ostringstream ostr;
                ostr << std::this_thread::get_id() << ": value of lock is: " << m_locked.test_and_set() << std::endl;
                std::cout << ostr.str();*/

                //
                // m_free is atomic so no need to guard
                m_locked.clear(std::memory_order_release);
            }

            //
            // get

            vk::Buffer get_buffer(void) const
            {
                return m_buffer;
            }

            vk::DeviceMemory get_memory(void) const
            {
                return m_memory;
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

        private:            
            std::atomic_flag m_locked = ATOMIC_FLAG_INIT;

            vk::DeviceMemory m_memory;
            vk::DeviceSize m_offset;
            vk::DeviceSize m_size;
        
            void* m_ptr;
            const vk::Buffer m_buffer;
        };

        //
        //    
        class memory_chunk
        {
        public:

            /*
                public synchronized interface
            */

            //const uint32_t memoryTypeIndex, const bool hostVisible,
            //memory_chunk(const vk::PhysicalDevice& physDevice, const vk::Device& device, const vk::MemoryPropertyFlags& memory_properties, const vk::DeviceSize size) :
            memory_chunk(const vk::PhysicalDevice& physDevice, const vk::Device& device, const bool hostVisible, const uint32_t memoryTypeIndex, const vk::DeviceSize size) :
                //
                // create buffer
                m_buffer(device.createBufferUnique(vk::BufferCreateInfo(vk::BufferCreateFlags(), size, vk::BufferUsageFlags(bufferUsageFlags::eDefault), vk::SharingMode::eExclusive))),
                m_memreq(device.getBufferMemoryRequirements(m_buffer.get())),

                //
                // memory information
                m_hostVisible(hostVisible),
                m_memoryTypeIndex(memoryTypeIndex),

                //
                // allocate chunck of memory with the correct memory alignment 
                m_size(m_memreq.size),
                m_memory(device.allocateMemoryUnique(vk::MemoryAllocateInfo(m_size, m_memoryTypeIndex))),

                //
                // map memory (if host visible)
                m_memptr(m_hostVisible ? device.mapMemory(m_memory.get(), 0, VK_WHOLE_SIZE) : nullptr)
            {
                //
                // bind buffer to memory
                device.bindBufferMemory(m_buffer.get(), m_memory.get(), 0);

                //
                //
                m_mtxBlocks = std::make_unique<std::mutex>();

                //
                // create initial block
                m_blocks.emplace_back(std::make_unique<memory_block>(m_memory.get(), 0, m_size, m_buffer.get()));

                //
                // hello there
                //std::cout << "new block bound" << std::endl;
            }

            //
            // tries to find a free block within the chunk of memory with the required size
            // returns true if this succeeds
            // returns false if this fails
            memory_block* allocate(const vk::DeviceSize size)
            {
                if(size > m_size)
                    return nullptr;

                vk::DeviceSize alignment = m_memreq.alignment;

                //
                // find a free block that has sufficiently range

                std::lock_guard<std::mutex> lck(*m_mtxBlocks);

                for(size_t i = 0; i < m_blocks.size(); ++i)
                {   
                    //
                    // first find, first serve
                    if(m_blocks[i]->test_and_set() == false)
                    {
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
                                m_blocks.emplace_back(std::make_unique<memory_block>(m_memory.get(), offset + size, virtualSize - size, m_buffer.get()));
                            }

                            // if host visible push the memory pointer
                            void* ptr = nullptr;
                            if(m_memptr != nullptr)
                                 ptr = (char*)m_memptr + offset;
                        
                            // can call realloc since test_and_set have been called beforehand
                            m_blocks[i]->reallocate(offset, size, ptr);
                            return m_blocks[i].get();
                        }
                    }
                }
                return nullptr;
            }

            /*
                only for debugging purposes
        
            size_t get_size(void) const
            {
                std::lock_guard<std::mutex> lck(*m_mtxBlocks);
                return m_blocks.size();
            }*/

        private:

            //
            // buffer bounded to the chunk
            vk::UniqueBuffer m_buffer;
            const vk::MemoryRequirements m_memreq;
            //vk::DeviceSize m_alignment;

            //
            // memory information
            const bool m_hostVisible;
            const uint32_t m_memoryTypeIndex;

            //
            // memory
            const vk::DeviceSize m_size;
            vk::UniqueDeviceMemory m_memory;
            const void *m_memptr;

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
            memory_allocator(const vk::PhysicalDevice& physDevice, const vk::Device& device, const vk::DeviceSize default_alloc_size = (vk::DeviceSize)(1 << 28)) :
                m_physDevice(physDevice),
                m_device(device),
                m_defaultChunkSize(default_alloc_size)
            {
                //
                // fetch the memory types and properties from the device
                VudaMemoryProperties deviceMemProperties(physDevice);

                //
                // resize
                const uint32_t vudaNumMemTypes = vudaMemoryTypes::eLast;
                m_memoryAllocatorIndices.resize(vudaNumMemTypes);
                m_memoryProperties.resize(vudaNumMemTypes);

                vk::MemoryPropertyFlags candidate;
                uint32_t memoryTypeIndex;

                //
                // find and define suitable memory types
                /*
                    some general rule of thumbs:
                    | OS        | Vendor                            | Comment
                    =======================================================================================================
                    | Windows   | all three major PC GPU vendors    | all memory that is HOST_VISIBLE is also HOST_COHERENT
                    | Windows   | Intel                             | all memory is HOST_VISIBLE
                    | Mac OS    | AMD, Intel                        | memory that is HOST_CACHED is not HOST_COHERENT
                */
                //
                // memory properties -> memory types
                memoryTypeIndex = deviceMemProperties.FindMemoryTypeFromCandidates(device, std::vector<vk::MemoryPropertyFlags>{
                    vk::MemoryPropertyFlagBits::eDeviceLocal
                }, candidate);
                m_memoryAllocatorIndices[vudaMemoryTypes::eDeviceLocal] = memoryTypeIndex;
                m_memoryProperties[vudaMemoryTypes::eDeviceLocal] = candidate;
                
                memoryTypeIndex = deviceMemProperties.FindMemoryTypeFromCandidates(device, std::vector<vk::MemoryPropertyFlags>{ 
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent                
                }, candidate);
                m_memoryAllocatorIndices[vudaMemoryTypes::ePinned] = memoryTypeIndex;
                m_memoryProperties[vudaMemoryTypes::ePinned] = candidate;
                m_memoryAllocatorIndices[vudaMemoryTypes::ePinned_Internal] = memoryTypeIndex;
                m_memoryProperties[vudaMemoryTypes::ePinned_Internal] = candidate;

                memoryTypeIndex = deviceMemProperties.FindMemoryTypeFromCandidates(device, std::vector<vk::MemoryPropertyFlags>{
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached | vk::MemoryPropertyFlagBits::eHostCoherent, // first try if we can get coherent memory
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCached,                                             // then we settle non-coherent but cached
                    vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent                                            // guaranteed fallback
                }, candidate);
                m_memoryAllocatorIndices[vudaMemoryTypes::eCached] = memoryTypeIndex;
                m_memoryProperties[vudaMemoryTypes::eCached] = candidate;
                m_memoryAllocatorIndices[vudaMemoryTypes::eCached_Internal] = memoryTypeIndex;
                m_memoryProperties[vudaMemoryTypes::eCached_Internal] = candidate;

                //
                //
                vk::DeviceSize device_memory_size = deviceMemProperties.GetDeviceLocalMemorySize();
                m_defaultChunkSize = std::min<vk::DeviceSize>(device_memory_size / 16, m_defaultChunkSize);

                //
                // types of memory
                std::vector<uint32_t> memoryIndices;
                uint32_t numMemTypes = deviceMemProperties.GetNumberOfUniqueMemoryTypes(memoryIndices);

                // pure index -> memory index
                for(uint32_t i = 0; i < numMemTypes; ++i)
                    m_memoryIndexToPureIndex[memoryIndices[i]] = i;

                //
                // allocate a vector for each type of memory
                m_type_chunks.resize(numMemTypes);
                m_creation_locks.resize(numMemTypes);
                m_mtxTypeChunks.resize(numMemTypes);

                //
                // allocate creation locks and mutex for each type of memory
                for(size_t i = 0; i < numMemTypes; ++i)
                {
                    m_creation_locks[i] = std::make_unique<std::atomic<bool>>();
                    m_creation_locks[i]->store(false);
                    m_mtxTypeChunks[i] = std::make_unique<std::shared_mutex>();
                }
            }

            //memory_block* allocate(const vk::MemoryPropertyFlags& memory_properties, const vk::DeviceSize size)
            memory_block* allocate(const vudaMemoryTypes& memory_type, const vk::DeviceSize size)
            {
                memory_block* block;

                //
                // will throw exception if m_memoryAllocatorTypes does not know the set of memory properties
                //uint32_t pureIndex = m_memoryIndexToPureIndex.at(VkFlags(memory_properties));
                uint32_t pureIndex = m_memoryIndexToPureIndex[m_memoryAllocatorIndices[memory_type]];
            
                while(true)
                {
                    //
                    // find a suitable chunk that has a free chunk with suitable size
                    {
                        std::shared_lock<std::shared_mutex> lck(*m_mtxTypeChunks[pureIndex]);

                        for(auto& chunk : m_type_chunks[pureIndex])
                        {
                            //
                            // attempt to retrieve block from chunk
                            block = chunk.allocate(size);
                            if(block != nullptr)
                                return block;
                        }

                        /*std::stringstream ostr;
                        ostr << "requires a new chunk" << std::endl;
                        std::cout << ostr.str();*/
                    }
            
                    //
                    // create new memory chunk
                    if(m_creation_locks[pureIndex]->exchange(true) == false)
                    {
                        std::lock_guard<std::shared_mutex> lck(*m_mtxTypeChunks[pureIndex]);

                        vk::DeviceSize use_size = m_defaultChunkSize;
                        if(size > m_defaultChunkSize)
                        {
                            // increase default allocation size
                            use_size = size;
                        }

                        m_type_chunks[pureIndex].emplace_back(m_physDevice, m_device, IsHostVisible(memory_type), m_memoryAllocatorIndices[memory_type], use_size);

                        /*std::stringstream ostr;
                        ostr << "creating chunk" << std::endl;
                        std::cout << ostr.str();*/

                        m_creation_locks[pureIndex]->store(false);
                    }
                    else
                    {
                        //
                        // wait for creation proccess to complete and try again
                        while(m_creation_locks[pureIndex]->load() == true)
                            ;
                    }
                }
            }

            inline bool IsHostVisible(const vudaMemoryTypes& memory_type)
            {
                if(m_memoryProperties[memory_type] & vk::MemoryPropertyFlagBits::eHostVisible)
                    return true;
                else
                    return false;
            }

            inline bool IsHostCoherent(const vudaMemoryTypes& memory_type)
            {
                if(m_memoryProperties[memory_type] & vk::MemoryPropertyFlagBits::eHostCoherent)
                    return true;
                else
                    return false;
            }

            /*memory_block* allocate(const vk::DeviceSize size, const vk::DeviceSize alignment)
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
                    std::lock_guard<std::shared_mutex> lck(m_mtxChunks);

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
            }*/

            /*
                only for debugging purposes
        
            size_t get_size(const unsigned int index) const
            {
                std::shared_lock<std::shared_mutex> lck(*m_mtxTypeChunks[index]);

                if(m_type_chunks[index].size() > 0)
                    return m_type_chunks[index][0].get_size();
                else
                    return 0;
            }
            */

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
            const vk::PhysicalDevice m_physDevice;
            const vk::Device m_device;

            //
            // memory allocation settings
            vk::DeviceSize m_defaultChunkSize;

            //
            // chunks of memory
            /*std::shared_mutex m_mtxChunks;
            std::vector<memory_chunk> m_chunks;*/

            //
            //
            std::vector<std::unique_ptr<std::atomic<bool>>> m_creation_locks;
            std::vector<std::unique_ptr<std::shared_mutex>> m_mtxTypeChunks;
            std::vector<std::vector<memory_chunk>> m_type_chunks;

            std::vector<uint32_t> m_memoryAllocatorIndices;
            std::vector<vk::MemoryPropertyFlags> m_memoryProperties;
            std::unordered_map<uint32_t, uint32_t> m_memoryIndexToPureIndex;
        };

    } //namespace detail
} // namespace vuda