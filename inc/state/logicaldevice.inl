
namespace vuda
{
    namespace detail
    {

        inline logical_device::logical_device(const vk::DeviceCreateInfo& deviceCreateInfo, const vk::PhysicalDevice& physDevice) :
            /*
                create (unique) logical device from the physical device specified
                create a command pool on the queue family index (assuming that there is only one family!)
            */
            //m_physDevice(physDevice),
            m_device(physDevice.createDeviceUnique(deviceCreateInfo)),
            m_queueFamilyIndex(deviceCreateInfo.pQueueCreateInfos->queueFamilyIndex),
            m_queueComputeCount(deviceCreateInfo.pQueueCreateInfos->queueCount),/*,
            m_commandPool(m_device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), m_queueFamilyIndex))),
            m_commandBuffers(m_device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(m_commandPool.get(), vk::CommandBufferLevel::ePrimary, m_queueComputeCount))),
            m_commandBufferState(m_queueComputeCount, cbReset)*/
            m_storageBST_root(nullptr),
            m_allocator(physDevice, m_device.get()) // findDeviceLocalMemorySize(physDevice) / 16
        {
            // device allocator     : device local mem type
            // host allocator       : host local mem type
            // cached allocator     : suitable for stage buffers for device to host transfers
            // ...                  : suitable for stage buffers for host to device transfers        
            
            /*m_allocatorDevice(physDevice, m_device.get(), false, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eDeviceProperties), findDeviceLocalMemorySize(physDevice) / 16),
            m_allocatorHost(physDevice, m_device.get(), true, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eHostProperties), findDeviceLocalMemorySize(physDevice) / 16),
            m_allocatorCached(physDevice, m_device.get(), true, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eCachedProperties), findDeviceLocalMemorySize(physDevice) / 16)*/

            //
            // retrieve timestampPeriod
            vk::PhysicalDeviceProperties deviceProperties;
            physDevice.getProperties(&deviceProperties);
            m_timestampPeriod = deviceProperties.limits.timestampPeriod;
            m_nonCoherentAtomSize = deviceProperties.limits.nonCoherentAtomSize;

            //
            // create unique mutexes
            m_mtxResources = std::make_unique<std::shared_mutex>();
            //m_mtxModules = std::make_unique<std::shared_mutex>();
            m_mtxKernels = std::make_unique<std::shared_mutex>();
            m_mtxCmdPools = std::make_unique<std::shared_mutex>();
            m_mtxEvents = std::make_unique<std::shared_mutex>();

            m_kernel_creation_lock = std::make_unique<std::atomic<bool>>();
            m_kernel_creation_lock->store(false);
        
            m_mtxQueues.resize(m_queueComputeCount);
            for(unsigned int i = 0; i < m_queueComputeCount; ++i)
                m_mtxQueues[i] = std::make_unique<std::mutex>();
        
            //
            // generate queue indices (this is write protected)
            /*const uint32_t queueFamilyCount = 1;        
            for(uint32_t family = 0; family < queueFamilyCount; ++family)
            {
                std::vector<vk::Queue> queueIndexList;
                queueIndexList.resize(m_queueComputeCount);

                for(uint32_t queue = 0; queue < m_queueComputeCount; ++queue)
                    queueIndexList[queue] = m_device->getQueue(family, queue);

                m_queues[family] = queueIndexList;
            }*/

            m_queues.resize(m_queueComputeCount);
            for(uint32_t queue = 0; queue < m_queueComputeCount; ++queue)
                m_queues[queue] = m_device->getQueue(m_queueFamilyIndex, queue);

            //
            // [ assume that there are only one queue family for now ]
            m_internal_pinned_buffers_in_use.resize(m_queueComputeCount);
        }

        #ifdef VUDA_DEBUG_ENABLED
        inline vk::Device logical_device::GetDeviceHandle(void)
        {
            return m_device.get();
        }
        #endif

        //
        // event management
        //

        inline void logical_device::CreateEvent(event_t* event)
        {
            std::lock_guard<std::shared_mutex> lck(*m_mtxEvents);

            *event = m_device->createEvent(vk::EventCreateInfo());
            m_events.emplace(*event, event_tick());
        }

        inline void logical_device::DestroyEvent(const event_t event)
        {
            std::lock_guard<std::shared_mutex> lck(*m_mtxEvents);

            //
            // force a flush of the the command buffer before destroying the event
            FlushQueue(std::this_thread::get_id(), m_events.at(event).get_stream());

            //
            // destroy event
            m_device->destroyEvent(event);
            m_events.erase(event);
        }

        inline void logical_device::RecordEvent(const std::thread::id tid, const event_t& event, const stream_t stream)
        {
            //
            // record timestamp host side
            {
                std::lock_guard<std::shared_mutex> lck(*m_mtxEvents);

                event_tick& ref = m_events.at(event);
                ref.set_stream(stream);
                ref.tick();
            }

            //
            // every thread can look up its command pool in the list
            {
                std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);

                const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);
                pool->SetEvent(m_device, event, stream);
            }
        }

        inline float logical_device::GetElapsedTimeBetweenEvents(const event_t& start, const event_t& end)
        {
            //
            // every thread can look in the events
            std::shared_lock<std::shared_mutex> lckEvents(*m_mtxEvents);

            //
            // events must belong to the same stream
            assert(m_events[start].get_stream() == m_events[end].get_stream());
        
            // return in milli seconds
            return m_events[end].toc_diff(m_events[start].get_tick()) * 1e3f;
        }

        //
        // event management (through queries)
        //

        inline void logical_device::GetQueryID(const std::thread::id tid, uint32_t* event) const
        {
            //
            // every thread can look up its command pool in the list
            {
                std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);                
                *event = pool->GetQueryID();
            }
        }

        inline void logical_device::WriteTimeStamp(const std::thread::id tid, const uint32_t event, const stream_t stream) const
        {        
            //
            // every thread can look up its command pool in the list
            {
                std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);                
                        
                pool->WriteTimeStamp(m_device, event, stream);
            }
        }

        inline float logical_device::GetQueryPoolResults(const std::thread::id tid, const uint32_t startQuery, const uint32_t stopQuery) const
        {
            uint64_t ticks;

            //
            // every thread can look up its command pool in the list
            {
                std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);
                
                uint64_t ticks0 = pool->GetQueryPoolResults(m_device, startQuery);
                uint64_t ticks1 = pool->GetQueryPoolResults(m_device, stopQuery);

                ticks = ticks1 - ticks0;
            }

            // time in nano seconds
            float elapsedTime = m_timestampPeriod * (float)ticks;
            // return time in milli seconds
            return elapsedTime * 1e-6f;
        }

        //
        // memory management
        //

        inline void logical_device::malloc(void** devPtr, size_t size)
        {
            //
            // create new node in storage_bst
            device_buffer_node* node = new device_buffer_node(size, m_allocator);

            /*std::stringstream ostr;
            ostr << "tid: " << std::this_thread::get_id() << ", buffer: " << node->GetBuffer() << ", offset: " << node->GetOffset() << std::endl;
            std::cout << ostr.str();*/

            //
            // return the memory pointer
            (*devPtr) = node->key();

            //
            // insert the node in the storage tree
            push_mem_node(node);

            //
            // make sure we have a staging buffer of equal size
            // [ for now each device buffer is backed by equal sized buffers, this is not very economical to say the least ]
            // [ for concurrent transfers we want to have sufficient pre-allocation, but we dont want to overcommit as the current implementation ]            
            host_pinned_node_internal* stage_ptr = m_pinnedBuffers.get_buffer(size, m_allocator);
            stage_ptr->set_free();
            host_cached_node_internal* dstptr = m_cachedBuffers.get_buffer(size, m_allocator);
            dstptr->set_free();
        }

        inline void logical_device::mallocHost(void** ptr, size_t size)
        {
            //
            // create new node in pinned host visible in storage_bst 
            default_storage_node* node = new default_storage_node(vudaMemoryTypes::eCached, size, m_allocator);

            //
            // return the memory pointer
            (*ptr) = node->key();

            //
            // insert the node in the storage tree
            push_mem_node(node);
        }

        inline void logical_device::hostAlloc(void** ptr, size_t size)
        {
            //
            // create new node in cached, pinned, host visible mem
            default_storage_node* node = new default_storage_node(vudaMemoryTypes::ePinned, size, m_allocator);

            //
            // return the memory pointer
            (*ptr) = node->key();

            //
            // insert the node in the storage tree
            push_mem_node(node);
        }

        inline void logical_device::free(void* devPtr)
        {
            std::lock_guard<std::shared_mutex> lck(*m_mtxResources);

            /*std::ostringstream ostr;
            //ostr << std::this_thread::get_id() << ": took lock" << std::endl;
            ostr << std::this_thread::get_id() << ", allocator: " << m_allocator.get_size(0) << std::endl;
            std::cout << ostr.str();
            ostr.str("");*/

            //
            // check whether the memory exists on the set device
            default_storage_node* node = m_storage.search(m_storageBST_root, devPtr);
            if(node == nullptr)
                throw std::runtime_error("vuda: Failed to find memory on the specified device!");
                
            /*ostr << std::this_thread::get_id() << ": destroying memory with ptr: " << devPtr << ", node: " << node << std::endl;
            std::cout << ostr.str();
            //m_storage.walk(m_storageBST_root);
            m_storage.walk_depth(m_storageBST_root);
            std::cout << std::endl;*/

            //
            // destroy the satellite data on the node
            node->destroy();
        
            //
            // remove node from the bst tree        
            default_storage_node* doomed = m_storage.delete_node(m_storageBST_root, node);

            //
            // remove the node from the bst storage tree data        
            //m_storageBST.erase(std::remove(m_storageBST.begin(), m_storageBST.end(), doomed), m_storageBST.end());

            //
            // remove the spliced node from the heap
            // (should perhaps recycle nodes? we know there is an upper limit on allocations)
            delete doomed;
            //doomed = nullptr;

            //
            //m_storage.walk_depth(m_storageBST_root);
            //std::cout << std::endl;
        }

        inline vk::DescriptorBufferInfo logical_device::GetBufferDescriptor(void* devPtr) const
        {
            //
            // all threads can read buffer indices
            std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

            const default_storage_node* node = m_storage.search_range(m_storageBST_root, devPtr);
            assert(node != nullptr);

            /*
            https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/VkDescriptorBufferInfo.html
            - (1) offset must be less than the size of buffer
            - (2) If range is not equal to VK_WHOLE_SIZE, range must be greater than 0
            - (3) If range is not equal to VK_WHOLE_SIZE, range must be less than or equal to the size of buffer minus offset
            */

            //
            // memory offset
            vk::DeviceSize gOffset = node->GetOffset();
        
            // the offset must always be unsigned!
            assert(static_cast<const char*>(devPtr) >= static_cast<const char*>(node->key()));
            vk::DeviceSize offset = static_cast<const char*>(devPtr) - static_cast<const char*>(node->key());

            // (1)
            vk::DeviceSize size = node->GetSize();
            assert(offset < size);

            // (2) and (3)
            vk::DeviceSize range = size - offset;        
        
            vk::DescriptorBufferInfo desc = vk::DescriptorBufferInfo()
                .setBuffer(node->GetBuffer())
                .setOffset(gOffset + offset)
                .setRange(range);

            //
            // hello there
            /*std::ostringstream ostr;
            ostr << std::this_thread::get_id() << ", descriptor data: " << desc.buffer << ", offset: " << desc.offset << ", range: " << desc.range << std::endl;
            std::cout << ostr.str();*/

            return desc;
        }

        //
        // kernels associated with the logical device
        //

        template <typename... specialTypes, size_t bindingSize>
        inline void logical_device::SubmitKernel(   const std::thread::id tid, std::string const& identifier, char const* entry,
                                                    const std::array<vk::DescriptorSetLayoutBinding, bindingSize>& bindings,
                                                    const specialization<specialTypes...>& specials,
                                                    const std::array<vk::DescriptorBufferInfo, bindingSize>& bufferDescriptors,                                                    
                                                    const dim3 blocks,
                                                    const stream_t stream)
        {
            assert(stream >= 0 && stream < m_queueComputeCount);

            //
            // NOTE: for now, we use a fairly naive phrase as a "unique" identifier in order to accommodate embedded kernels
            const std::string identifier_entry(identifier + "::" + entry);

            while(true)
            {
                kernelprogram<specialization<specialTypes...>::m_bytesize>* kernel = nullptr;
                std::unordered_map<std::string, std::shared_ptr<kernel_interface>>::const_iterator it;

                //
                // check if the kernel is already created (shared lock)
                {
                    std::shared_lock<std::shared_mutex> lck(*m_mtxKernels);
                    it = m_kernels.find(identifier_entry);
                    if(it != m_kernels.end())
                    {
                        kernel = static_cast<kernelprogram<specialization<specialTypes...>::m_bytesize>*>((*it).second.get());
                        
                        //
                        // every thread can look up its command pool in the list
                        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                        const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

                        //
                        // 1. record command buffer: update descriptor and command buffer            
                        pool->UpdateDescriptorAndCommandBuffer<specialization<specialTypes...>::m_bytesize, specialTypes...>(m_device, *kernel, specials, bufferDescriptors, blocks, stream);

                        //
                        // lock queue
                        std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
                        vk::Queue q = m_queues.at(stream);

                        //
                        // 2. submit kernel, kernel launches are asynchronous with respect to the host.
                        pool->Execute(m_device, q, stream);

                        return;
                    }

                    /*std::stringstream ostr;
                    ostr << "Will attempt kernel creation" << std::endl;
                    std::cout << ostr.str();*/
                }
            
                if(m_kernel_creation_lock->exchange(true) == false)
                {
                    std::lock_guard<std::shared_mutex> lck(*m_mtxKernels);

                    //
                    // retrieve shader module
                    vk::ShaderModule shaderModule = CreateShaderModule(identifier, entry);                    
                    m_kernels.try_emplace(identifier_entry, std::make_unique<kernelprogram<specialization<specialTypes...>::m_bytesize>>(m_device, shaderModule, entry, bindings, specials));
                    //assert(pair.second);

                    /*std::stringstream ostr;
                    ostr << "creating kernel" << std::endl;
                    std::cout << ostr.str();*/

                    m_kernel_creation_lock->store(false);
                }
                else
                {
                    //
                    // wait for creation proccess to complete and try again
                    while(m_kernel_creation_lock->load() == true)
                        ;
                }
            }
        }

        inline void logical_device::WaitOn(void) const
        {
            //
            // To wait on the host for the completion of outstanding queue operations for all queues on a given logical device

            m_device->waitIdle();
        }

        //
        // command buffer functions
        //

        inline void logical_device::CreateCommandPool(const std::thread::id tid)
        {
            //
            // only one thread at a time can create its command pool
            std::lock_guard<std::shared_mutex> lck(*m_mtxCmdPools);
            //m_thrdCommandPools.insert({ tid, thrdcmdpool(m_device, m_queueFamilyIndex, m_queueComputeCount) });
            //m_thrdCommandPools.emplace(std::piecewise_construct, std::forward_as_tuple(tid), std::forward_as_tuple(m_device, m_queueFamilyIndex, m_queueComputeCount));

            // c++17
            m_thrdCommandPools.try_emplace(tid, m_device, m_queueFamilyIndex, m_queueComputeCount);
            //m_thrdCommandPools.try_emplace(tid, std::make_unique<thrdcmdpool>(m_device, m_queueFamilyIndex, m_queueComputeCount));
        }

        inline void logical_device::memcpyHtH(const std::thread::id tid, void* dst, const void* src, const size_t count, const stream_t stream) const
        {
        
        }

        inline void logical_device::memcpyToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const stream_t stream)
        {
            /*
            conformity to cudaMemcpy synchronous
            https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior        
            */

            assert(stream >= 0 && stream < m_queueComputeCount);
            bool use_staged = false;

            //
            // all threads can read from the memory resources on the logical device
            {
                std::shared_lock<std::shared_mutex> lckResources(*m_mtxResources);

                const default_storage_node* dst_node = m_storage.search_range(m_storageBST_root, dst);
                assert(dst_node != nullptr);

                //
                // NOTE: src can be any ptr, we must check if the pointer is a known resource node or a pageable host allocation
                internal_node* src_ptr = m_storage.search_range(m_storageBST_root, const_cast<void*>(src));
                
                if(src_ptr == nullptr)
                {
                    // the src is not known, assume that we are copying from pageable host mem and copy to an internal staging buffer

                    //
                    // perform stream sync before initiating the copy
                    FlushQueue(tid, stream);

                    //
                    // request a pinned (internal) staging buffer
                    // copy the memory to a pinned staging buffer which is allocated with host visible memory (this is the infamous double copy)
                    // copy from stage buffer to device
                    use_staged = true;
                    src_ptr = m_pinnedBuffers.get_buffer(count, m_allocator);
                    std::memcpy(src_ptr->get_memptr(), src, count);

                    /*std::ostringstream ostr;
                    ostr << "tid: " << std::this_thread::get_id() << ", using staged node: " << stage_ptr << std::endl;
                    std::cout << ostr.str();*/
                }
                assert(src_ptr != nullptr);

                if(src_ptr->IsHostVisible() == false)
                    throw std::runtime_error("vuda: the source must be a pointer to host visible memory!");

                //
                // if the source is non-coherent the CPU writes to the mapped buffer must be flushed before the GPU can read it
                if(src_ptr->IsHostCoherent() == false)
                {
                    // VkMappedMemoryRange
                    // offset and size must specify a range contained within the currently mapped range of memory
                    // offset must be a multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize
                    // size must either be a multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize, or offset plus size must equal the size of memory
                    // NOTE: the size of the buffer should always be a multiple of the atom size - otherwise the memory allocator is doing something shady

                    vk::DeviceSize offset_noncoherentatomsize = (src_ptr->GetOffset() / m_nonCoherentAtomSize) * m_nonCoherentAtomSize;
                    vk::DeviceSize count_noncoherentatomsize = ((count + m_nonCoherentAtomSize - 1) / m_nonCoherentAtomSize) * m_nonCoherentAtomSize;
                    assert(offset_noncoherentatomsize % m_nonCoherentAtomSize == 0u);
                    assert(count_noncoherentatomsize % m_nonCoherentAtomSize == 0u);
                    m_device->flushMappedMemoryRanges(vk::MappedMemoryRange(src_ptr->GetMemory(), offset_noncoherentatomsize, count_noncoherentatomsize));
                }

                //
                // issue the copy

                //
                // every thread can look up its command pool in the list
                {
                    std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                    const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);
                    std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
                    vk::Queue q = m_queues.at(stream);

                    if(use_staged == true)
                    {
                        //
                        // 2. The function will return once the pageable buffer has been copied to the staging memory for DMA transfer to device memory, but the DMA to final destination may not have completed.

                        //
                        // remember stage_ptr (protected by m_mtxQueues)
                        std::vector<std::thread::id>& ref = m_internal_pinned_buffers_in_use.at(stream)[src_ptr];
                        ref.push_back(tid);
                    }

                    pool->memcpyDevice(m_device, dst_node->GetBuffer(), dst_node->GetOffset(), src_ptr->GetBuffer(), src_ptr->GetOffset(), count, q, stream);

                    if(use_staged == true)
                    {
                        //
                        // 3. For transfers from pinned host memory to device memory, the function is synchronous with respect to the host.
                        //
                        // internal flush queue
                        flush_queue(tid, stream, pool, q);
                    }
                }
            }
        }

        inline void logical_device::memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const stream_t stream) const
        {
            /*
            conformity to cudaMemcpy synchronous
            https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior

            5. For transfers from device memory to device memory, no host-side synchronization is performed.
            */

            assert(stream >= 0 && stream < m_queueComputeCount);

            //
            // all threads can read from the memory resources on the logical device
            {
                std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

                //
                // copy from node to node
                const default_storage_node* dst_node = m_storage.search_range(m_storageBST_root, dst);
                const default_storage_node* src_node = m_storage.search_range(m_storageBST_root, const_cast<void*>(src));

                vk::Buffer dstbuf = dst_node->GetBuffer();
                vk::DeviceSize dstOffset = dst_node->GetOffset();

                vk::Buffer srcbuf = src_node->GetBuffer();
                vk::DeviceSize srcOffset = src_node->GetOffset();

                //
                // every thread can look up its command pool in the list
                std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

                std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
                vk::Queue q = m_queues.at(stream);

                pool->memcpyDevice(m_device, dstbuf, dstOffset, srcbuf, srcOffset, count, q, stream);
            }
        }

        inline void logical_device::memcpyToHost(const std::thread::id tid, void* dst, const void* src, const size_t count, const stream_t stream)
        {
            /*
            conformity to cudaMemcpy synchronous
            https://docs.nvidia.com/cuda/cuda-runtime-api/api-sync-behavior.html#api-sync-behavior
            */

            //
            // execute kernels that have access to modify src
            // the address of src will only be associated with one logical device.
            // the logical device associated with the calling thread must be the same for all calling threads accessing src

            assert(stream >= 0 && stream < m_queueComputeCount);
            bool use_staged = false;

            //
            // all threads can read from the memory resources on the logical device
            {
                std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

                //
                // internal copy in the node
                const default_storage_node* src_node = m_storage.search_range(m_storageBST_root, const_cast<void*>(src));
                assert(src_node != nullptr);

                //
                // check if destination is known to vuda
                internal_node* dst_ptr = m_storage.search_range(m_storageBST_root, dst);

                if(dst_ptr == nullptr)
                {
                    // the dst address is not known to vuda, assume that we are copying to pageable host memory
                    // use staged buffer (pinned host cached memory) to perform internal copy before we copy to pageable host memory
                    use_staged = true;
                    dst_ptr = m_cachedBuffers.get_buffer(src_node->GetSize(), m_allocator);

                    /*std::ostringstream ostr;
                    ostr << "tid: " << std::this_thread::get_id() << ", using staged mem: " << dst_ptr->get_memptr() << std::endl;
                    std::cout << ostr.str();*/
                }
                assert(dst_ptr != nullptr);

                //
                // make sure the memory is a host visible memory target
                if(dst_ptr->IsHostVisible() == false)
                    throw std::runtime_error("vuda: the destination must be a pointer to host visible memory!");

                assert(count <= dst_ptr->GetSize());

                //
                // every thread can look up its command pool in the list
                {
                    std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
                    const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);
                    std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
                    vk::Queue q = m_queues.at(stream);

                    pool->memcpyDevice(m_device, dst_ptr->GetBuffer(), dst_ptr->GetOffset(), src_node->GetBuffer(), src_node->GetOffset(), count, q, stream);

                    //
                    // 4. For transfers from device to either pageable or pinned host memory, the function returns only once the copy has completed.
                    //
                    // internal flush queue
                    flush_queue(tid, stream, pool, q);
                }

                //
                // if the destination is non-coherent the the mapped buffer must be invalidated after GPU writes before CPU reads
                if(dst_ptr->IsHostCoherent() == false)
                {
                    // VkMappedMemoryRange
                    // offset and size must specify a range contained within the currently mapped range of memory
                    // offset must be a multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize
                    // size must either be a multiple of VkPhysicalDeviceLimits::nonCoherentAtomSize, or offset plus size must equal the size of memory
                    // NOTE: the size of the buffer should always be a multiple of the atom size - otherwise the memory allocator is doing something shady

                    vk::DeviceSize offset_noncoherentatomsize = (dst_ptr->GetOffset() / m_nonCoherentAtomSize) * m_nonCoherentAtomSize;
                    vk::DeviceSize count_noncoherentatomsize = ((count + m_nonCoherentAtomSize - 1) / m_nonCoherentAtomSize) * m_nonCoherentAtomSize;
                    assert(offset_noncoherentatomsize % m_nonCoherentAtomSize == 0u);
                    assert(count_noncoherentatomsize % m_nonCoherentAtomSize == 0u);
                    m_device->invalidateMappedMemoryRanges(vk::MappedMemoryRange(dst_ptr->GetMemory(), offset_noncoherentatomsize, count_noncoherentatomsize));
                }

                //
                // if we are using staging buffers
                if(use_staged == true)
                {
                    //
                    // copy the memory back to the staging buffer (host visible memory)
                    std::memcpy(dst, dst_ptr->get_memptr(), count);

                    //
                    // release the internal cached node
                    dst_ptr->set_free();
                }
            }
        }

        inline void logical_device::FlushQueue(const std::thread::id tid, const stream_t stream)
        {
            //
            // every thread can look up its command pool in the list
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);
        
            //
            // control queue submissions on this level
            std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
            vk::Queue q = m_queues.at(stream);

            //
            // internal flush queue
            flush_queue(tid, stream, pool, q);
        }

        inline void logical_device::FlushEvent(const std::thread::id tid, const event_t event)
        {
            //
            // retrieve stream id associated with event
            std::lock_guard<std::shared_mutex> lckEvents(*m_mtxEvents);
            const stream_t stream = m_events.at(event).get_stream();

            //
            // every thread can look up its command pool in the list
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

            //
            // control queue submissions on this level
            std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
            vk::Queue q = m_queues.at(stream);

            //
            // internal flush event
            flush_event(stream, event, pool, q);
        }

        //
        // private
        //

        inline void logical_device::flush_queue(const std::thread::id tid, const stream_t stream, const thrdcmdpool *pool, const vk::Queue q)
        {
            //
            // assumes that pool and queue are locked (see FlushQueue).

            //
            // 
            /*std::ostringstream ostr;
            ostr << "thrd: " << std::this_thread::get_id() << ", locked queue: " << stream << std::endl;
            std::cout << ostr.str();
            ostr.str("");*/

            //
            // execute and wait for stream
            pool->ExecuteAndWait(m_device, q, stream);

            //
            // hello there
            /*std::queue<internal_node*> temp_copy = m_internal_pinned_buffers_in_use[stream];
            std::ostringstream ostr;
            while(!temp_copy.empty())
            {
                ostr << "   " << temp_copy.front() << std::endl;
                temp_copy.pop();
            }
            std::cout << ostr.str();*/

            //
            // free internal pinned buffers (if any)
            std::unordered_map<internal_node*, std::vector<std::thread::id>>& ref = m_internal_pinned_buffers_in_use.at(stream);

            // the order of the elements that are not erased is preserved (this makes it possible to erase individual elements while iterating through the container) (since C++14)
            auto it = ref.begin();
            while(it != ref.end())
            {
                std::vector<std::thread::id>& using_threads = it->second;

                // [tid](std::thread::id& cur) { return cur == tid; }
                using_threads.erase(std::remove(using_threads.begin(), using_threads.end(), tid), using_threads.end());

                if(using_threads.size() == 0)
                {
                    //
                    // all threads are done with the memory, return it
                    it->first->set_free();

                    //
                    // hello
                    /*std::ostringstream ostr;
                    ostr << "tid: " << std::this_thread::get_id() << ", releasing staged node: " << it->first << std::endl;
                    std::cout << ostr.str();*/

                    //
                    // remove node, otherwise every thread will set the memory free
                    it = ref.erase(it);
                }
                else
                    ++it;
            }

            //
            //
            /*ostr << "thrd: " << std::this_thread::get_id() << ", unlocked queue: " << stream << std::endl;
            std::cout << ostr.str();*/
        }

        inline void logical_device::flush_event(const stream_t stream, const event_t event, const thrdcmdpool *pool, const vk::Queue q)
        {
            //
            // assumes that event, pool and queue are locked (see FlushEvent).

            //
            // start executing stream if it has not been submitted already
            pool->Execute(m_device, q, stream);

            //
            // wait for event on host by using 'spin-lock'/'busy-wait'
            // (this is not particularly efficient)
            //int count = 0;
            while(m_device->getEventStatus(event) == vk::Result::eEventReset)
            {
                //count++;
            }

            //
            // record tick host side
            m_events.at(event).tick();

            /*std::stringstream ostr;
            ostr << "vuda: event status attempt lock count: " << count << std::endl;
            std::cout << ostr.str();*/

            //
            // reset event
            m_device->resetEvent(event);
        }

        inline void logical_device::push_mem_node(default_storage_node* node)
        {
            //
            // protect storage tree
            std::lock_guard<std::shared_mutex> lck(*m_mtxResources);

            /*std::ostringstream ostr;
            ostr << std::this_thread::get_id() << ": took lock" << std::endl;
            std::cout << ostr.str();
            ostr.str("");*/

            //
            // push the node onto the bst storage tree data
            //m_storageBST.emplace_back(node);

            //
            // insert the node in the bst tree
            //m_storage.insert_node(m_storageBST_root, m_storageBST.back());
            m_storage.insert_node(m_storageBST_root, node);

            //
            // show storage tree
            /*m_storage.walk_depth(m_storageBST_root);
            ostr << std::this_thread::get_id() << ": releasing lock" << std::endl;
            std::cout << ostr.str();*/
        }

        //
        // create shader module
        //

        inline vk::ShaderModule logical_device::CreateShaderModule(std::string const& identifier, char const* entry)
        {
            std::unordered_map<std::string, vk::UniqueShaderModule>::const_iterator iter;

            //
            // take exclusive lock
            // only one thread can read (and potentially create a shader module at a time)
            // [ contention ]
            //std::lock_guard<std::shared_mutex> lck(*m_mtxModules);
            // NOTE: CreateShaderModule is protected by SubmitKernel

            const std::string identifier_entry(identifier + "::" + entry);

            //
            // check existence of the shader module
            iter = m_modules.find(identifier_entry);
            if(iter != m_modules.end())
            {
            /*#ifdef VUDA_DEBUG_ENABLED
                //
                // in debug mode check whether there exists a name clash between shader module entry names
                auto debug_iter = m_debug_module_entries.find(identifier);
                if(debug_iter == m_debug_module_entries.end())
                {
                    std::stringstream ostr; ostr << "shader module entry name '" << entry << "' clashes with a different binary!" << std::endl;
                    throw std::runtime_error(ostr.str());
                }
            #endif*/

                return iter->second.get();
            }

            //
            // determine whether the identifier is a binary else treat it as a filename.
            size_t bytesize = identifier.size();
                
            // NOTE: keep bin in scope when setting the pointer to binary
            std::vector<char> bin;
            const uint32_t* binary;

            //
            // create shader module
            if(bytesize >= 4 && identifier[0] == 0x03 && identifier[1] == 0x02 && identifier[2] == 0x23 && identifier[3] == 0x07)
            {
                //assert(program[0] == 0x03 && program[1] == 0x02 && program[2] == 0x23 && program[3] == 0x07); // "vuda: spir-v data is wrong, rebuild or check binaries"
                //assert(kernelByteSize % 4 == 0);// "vuda: spir-v data is wrong, codesize must be a multiple of 4"
                binary = reinterpret_cast<const uint32_t*>(identifier.c_str());
            /*#ifdef VUDA_DEBUG_ENABLED
                m_debug_module_entries.try_emplace(identifier, entry);
            #endif*/
            }
            else
            {
                bin = ReadShaderModuleFromFile(identifier);
                binary = reinterpret_cast<const uint32_t*>(bin.data());
                bytesize = bin.size();
            /*#ifdef VUDA_DEBUG_ENABLED
                m_debug_module_entries.try_emplace(std::string(bin.begin(), bin.end()), entry);
            #endif*/
            }

            auto pair = m_modules.try_emplace(identifier_entry, m_device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, bytesize, binary)));
            iter = pair.first;
            return iter->second.get();
        }

        inline std::vector<char> logical_device::ReadShaderModuleFromFile(const std::string& filename) const
        {
            std::ifstream file(filename, std::ios::ate | std::ios::binary);

            if(!file.is_open())
                throw std::runtime_error("Failed to open shader file!");

            size_t fileSize = (size_t)file.tellg();
            std::vector<char> buffer(fileSize);

            file.seekg(0);
            file.read(buffer.data(), fileSize);

            file.close();
            return buffer;
        }

    } //namespace detail
} //namespace vuda