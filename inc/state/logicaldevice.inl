
namespace vuda
{
    inline logical_device::logical_device(const vk::DeviceCreateInfo& deviceCreateInfo, const vk::PhysicalDevice& physDevice) :
        /*
            create (unique) logical device from the physical device specified
            create a command pool on the queue family index (assuming that there is only one family!)
        */
        m_physDevice(physDevice),
        m_device(physDevice.createDeviceUnique(deviceCreateInfo)),
        m_queueFamilyIndex(deviceCreateInfo.pQueueCreateInfos->queueFamilyIndex),
        m_queueComputeCount(deviceCreateInfo.pQueueCreateInfos->queueCount),/*,
        m_commandPool(m_device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), m_queueFamilyIndex))),
        m_commandBuffers(m_device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(m_commandPool.get(), vk::CommandBufferLevel::ePrimary, m_queueComputeCount))),
        m_commandBufferState(m_queueComputeCount, cbReset)*/
        m_storageBST_root(nullptr),
        m_memoryAllocatorTypes{
            {bufferPropertiesFlags::eDeviceProperties, findMemoryType_Device(m_physDevice, m_device.get())},
            {bufferPropertiesFlags::eHostProperties, findMemoryType_Host(m_physDevice, m_device.get())},
            {bufferPropertiesFlags::eCachedProperties, findMemoryType_Cached(m_physDevice, m_device.get())} },
            m_allocatorDevice(m_device.get(), false, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eDeviceProperties), findDeviceLocalMemorySize(physDevice) / 16),
            m_allocatorHost(m_device.get(), true, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eHostProperties), findDeviceLocalMemorySize(physDevice) / 16),
            m_allocatorCached(m_device.get(), true, m_memoryAllocatorTypes.at(vuda::bufferPropertiesFlags::eCachedProperties), findDeviceLocalMemorySize(physDevice) / 16)
    {
        // device allocator     : device local mem type
        // host allocator       : host local mem type                
        // cached allocator     : suitable for stage buffers for device to host transfers
        // ...                  : suitable for stage buffers for host to device transfers        

        //
        // retrieve timestampPeriod
        vk::PhysicalDeviceProperties deviceProperties;
        physDevice.getProperties(&deviceProperties);
        m_timestampPeriod = deviceProperties.limits.timestampPeriod;

        //
        // create unique mutexes
        m_mtxResources = std::make_unique<std::shared_mutex>();
        m_mtxKernels = std::make_unique<std::shared_mutex>();
        m_mtxCmdPools = std::make_unique<std::shared_mutex>();
        m_mtxEvents = std::make_unique<std::shared_mutex>();
        
        // memory allocators
        m_mtxAllocatorDevice = std::make_unique<std::mutex>();
        m_mtxAllocatorHost = std::make_unique<std::mutex>();
        m_mtxAllocatorCached = std::make_unique<std::mutex>();
        m_mtxCached_internal = std::make_unique<std::mutex>();
        
        m_mtxQueues.resize(m_queueComputeCount);
        for(unsigned int i = 0; i < m_queueComputeCount; ++i)
            m_mtxQueues[i] = std::make_unique<std::mutex>();
        
        //
        // generate queue indices (this is write protected)
        const uint32_t queueFamilyCount = 1;
        
        for(uint32_t family = 0; family < queueFamilyCount; ++family)
        {
            std::vector<vk::Queue> queueIndexList;
            queueIndexList.resize(m_queueComputeCount);

            for(uint32_t queue = 0; queue < m_queueComputeCount; ++queue)
                queueIndexList[queue] = m_device->getQueue(family, queue);

            m_queues[family] = queueIndexList;
        }
    }

    //
    // event management
    //

    inline void logical_device::CreateEvent(event_t* event)
    {
        std::unique_lock<std::shared_mutex> lck(*m_mtxEvents);

        *event = m_device->createEvent(vk::EventCreateInfo());
        m_events[*event] = event_tick();
    }

    inline void logical_device::RecordEvent(const std::thread::id tid, const event_t& event, const stream_t stream)
    {
        //
        // every thread can look up its command pool in the list
        {
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            thrdcmdpool* pool = &m_thrdCommandPools.at(tid);

            pool->SetEvent(m_device, event, stream);
        }

        //
        // record timestamp at the end
        {
            std::unique_lock<std::shared_mutex> lck(*m_mtxEvents);
            m_events[event].setStream(stream);
            m_events[event].tick();
        }
    }

    inline float logical_device::GetElapsedTimeBetweenEvents(const event_t& start, const event_t& end)
    {
        //
        // every thread can look in the events
        std::shared_lock<std::shared_mutex> lckEvents(*m_mtxEvents);

        //
        // events must belong to the same stream
        assert(m_events[start].getStream() == m_events[end].getStream());
        
        // return in milli seconds
        return m_events[end].toc_diff(m_events[start].getTick()) * 1e3f;
    }

    /*inline void logical_device::GetQueryID(const std::thread::id tid, event_t* event) const
    {
        //
        // every thread can look up its command pool in the list
        {
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);

            *event = pool->GetQueryID();
        }
    }

    inline void logical_device::WriteTimeStamp(const std::thread::id tid, const event_t event, const stream_t stream) const
    {        
        //
        // every thread can look up its command pool in the list
        {
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);

            pool->WriteTimeStamp(m_device, event, stream);
        }
    }

    inline float logical_device::GetQueryPoolResults(const std::thread::id tid, const event_t event) const
    {
        uint64_t ticks;

        //
        // every thread can look up its command pool in the list
        {
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            const thrdcmdpool* pool = &m_thrdCommandPools.at(tid);

            ticks = pool->GetQueryPoolResults(m_device, event);
        }

        // time in nano seconds
        float elapsedTime = m_timestampPeriod * (float)ticks;
        // return time in milli seconds
        return elapsedTime * 1e-6f;
    }*/

    //
    // memory management
    //

    inline void logical_device::malloc(void** devPtr, size_t size)
    {
        //std::ostringstream ostr;        
        device_buffer_node* node;
        {
            std::unique_lock<std::mutex> lock1(*m_mtxAllocatorDevice, std::defer_lock);
            std::unique_lock<std::mutex> lock2(*m_mtxAllocatorHost, std::defer_lock);

            // lock both unique_locks without deadlock
            std::lock(lock1, lock2);

            /*ostr << std::this_thread::get_id() << ": took lock" << std::endl;
            std::cout << ostr.str();
            ostr.str("");*/
                        
            //
            // create new node in storage_bst
            node = new device_buffer_node(m_physDevice, m_device.get(), size, m_allocatorDevice, m_allocatorHost);            
        }
        /*ostr << std::this_thread::get_id() << ": released lock" << std::endl;
        std::cout << ostr.str();
        ostr.str("");*/

        //
        // return the memory pointer
        (*devPtr) = node->key();

        //
        // insert the node in the storage tree
        push_mem_node(node);
    }

    inline void logical_device::mallocHost(void** ptr, size_t size)
    {
        //
        // create new node in storage_bst 
        host_buffer_node* node;
        {
            std::unique_lock<std::mutex> lckalloc(*m_mtxAllocatorHost);
            node = new host_buffer_node(m_physDevice, m_device.get(), size, m_allocatorHost);
        }        

        //
        // return the memory pointer
        (*ptr) = node->key();

        //
        // insert the node in the storage tree
        push_mem_node(node);
    }

    inline void logical_device::hostAlloc(void** ptr, size_t size)
    {
        // cached memory

        //
        // create new node in storage_bst 
        host_cached_node* node;
        {
            std::unique_lock<std::mutex> lckalloc(*m_mtxAllocatorCached);
            node = new host_cached_node(m_physDevice, m_device.get(), size, m_allocatorCached);
        }        

        //
        // return the memory pointer
        (*ptr) = node->key();

        //
        // insert the node in the storage tree
        push_mem_node(node);
    }

    inline void logical_device::free(void* devPtr)
    {
        std::unique_lock<std::shared_mutex> lck(*m_mtxResources);

        //std::ostringstream ostr;
        //ostr << std::this_thread::get_id() << ": took lock" << std::endl;

        //
        // check whether the memory exists on the set device
        default_storage_node* node = m_storage.search(m_storageBST_root, devPtr);
        if(node == nullptr)
            throw std::runtime_error("Failed to find memory on the specified device!");
                
        //ostr << std::this_thread::get_id() << ": destroying memory with ptr: " << devPtr << std::endl;
        //std::cout << ostr.str();
        //m_storage.walk(m_storageBST_root);

        //
        // destroy the nodes satellite data
        node->destroy(m_device.get());
        
        //
        // remove node from the bst tree        
        default_storage_node* doomed = m_storage.delete_node(m_storageBST_root, node);

        //
        // remove the spliced node from the heap
        // (should perhaps recycle nodes? we know there is an upper limit on allocations)
        delete doomed;
        doomed = nullptr;
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
        
        // the offset must always be unsigned!
        assert(static_cast<char*>(devPtr) >= static_cast<char*>(node->key()));
        vk::DeviceSize offset = static_cast<char*>(devPtr) - static_cast<char*>(node->key());

        // (1)
        vk::DeviceSize size = node->GetSize();
        assert(offset < size);

        // (2) and (3)
        vk::DeviceSize range = size - offset;        
        
        vk::DescriptorBufferInfo desc = vk::DescriptorBufferInfo()
            .setBuffer(node->GetBufferDevice())
            .setOffset(offset)
            .setRange(range);

        return desc;
    }

    //
    // kernels associated with the logical device
    //

    template <typename... specialTypes>
    inline void logical_device::SubmitKernel(   const std::thread::id tid, char const* filename, char const* entry,
                                                const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                                specialization<specialTypes...>& specials,
                                                const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors,
                                                const uint32_t blocks,
                                                const uint32_t stream)
    {
        bool create = false;
        kernelprogram<specials.bytesize>* kernel = nullptr;
        std::vector<std::shared_ptr<kernel_interface>>::iterator it;

        //
        // check if the kernel is already created and registered
        // [ this should be optimized to be ~O(1) or compile-const ]
        {
            std::shared_lock<std::shared_mutex> lck(*m_mtxKernels);

            it = std::find_if(m_kernels.begin(), m_kernels.end(), [&filename, &entry](std::shared_ptr<kernel_interface>& kernel)
            {
                return (kernel->GetFileName() == filename && kernel->GetEntryName() == entry);
            });

            //
            // get specialized kernel pointer before releasing lock
            if(it == m_kernels.end())
                create = true;
            else
                kernel = static_cast<kernelprogram<specials.bytesize>*>((*it).get());
        }

        //
        // if the kernel does not exist, it has to be created
        // lock all access to kernels - all pointers to m_kernels become invalid
        if(create)
        {
            std::unique_lock<std::shared_mutex> lck(*m_mtxKernels);

            m_kernels.push_back(std::make_shared<kernelprogram<specials.bytesize>>(m_device, filename, entry, bindings, specials));
            it = std::prev(m_kernels.end());

            kernel = static_cast<kernelprogram<specials.bytesize>*>((*it).get());
        }
        
        //
        // update descriptor and command buffer

        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        {
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

            //
            // every thread can read the kernels array
            {
                std::shared_lock<std::shared_mutex> lckKernels(*m_mtxKernels);

                pool->UpdateDescriptorAndCommandBuffer<specials.bytesize, specialTypes...>(m_device, *kernel, specials, bufferDescriptors, blocks, stream);
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
        std::unique_lock<std::shared_mutex> lck(*m_mtxCmdPools);
        //m_thrdCommandPools.insert({ tid, thrdcmdpool(m_device, m_queueFamilyIndex, m_queueComputeCount) });
        m_thrdCommandPools.emplace(std::piecewise_construct, std::forward_as_tuple(tid), std::forward_as_tuple(m_device, m_queueFamilyIndex, m_queueComputeCount));
    }

    inline void logical_device::memcpyHtH(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const
    {
        
    }

    inline void logical_device::memcpyToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // all threads can read from the memory resources on the logical device
        {
            std::shared_lock<std::shared_mutex> lckResources(*m_mtxResources);

            const default_storage_node* dst_node = m_storage.search_range(m_storageBST_root, dst);
            vk::Buffer dstbuf = dst_node->GetBufferDevice();
            vk::Buffer srcbuf;

            //
            // NOTE: src can be any ptr, we must check if the pointer is in known vk pinnned memory range or pure host allocated
            const default_storage_node* src_node = m_storage.search_range(m_storageBST_root, const_cast<void*>(src));            

            if(src_node == nullptr || src_node->isHostVisible() == false)
            {
                //
                // if src_node is null - assume that this is a pure host allocation and perform stage copy
                /*if(src_node == nullptr)
                    std::cout << "src node: nullptr" << std::endl;
                else
                    std::cout << "src node: " << src_node->key() << std::endl;*/

                //
                // copy the memory to a pinned staging buffer which is allocated with host visible memory (this is the infamous double copy)                
                std::memcpy(dst, src, count);
                
                //
                // internal copy in the node
                srcbuf = dst_node->GetBufferHost();
            }
            else //if(src_node->isHostVisible() == true)
            {
                //std::cout << "src node: " << src_node->key() << std::endl;

                srcbuf = src_node->GetBufferHost();
            }            

            std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
            vk::Queue q = m_queues.at(m_queueFamilyIndex).at(stream);

            pool->memcpyDevice(m_device, dstbuf, srcbuf, count, q, stream);
        }
    }

    inline void logical_device::memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream)
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // all threads can read from the memory resources on the logical device
        {
            std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

            //
            // copy from node to node
            vk::Buffer dstbuf = m_storage.search_range(m_storageBST_root, dst)->GetBufferDevice();
            vk::Buffer srcbuf = m_storage.search_range(m_storageBST_root, const_cast<void*>(src))->GetBufferDevice();

            std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
            vk::Queue q = m_queues.at(m_queueFamilyIndex).at(stream);

            pool->memcpyDevice(m_device, dstbuf, srcbuf, count, q, stream);
        }
    }

    inline void logical_device::memcpyToHost(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream)
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        host_cached_node_internal* dstptr = nullptr;        
        void *dst_memptr = nullptr;

        //
        // all threads can read from the memory resources on the logical device
        {
            std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

            //
            // internal copy in the node
            const default_storage_node* src_node = m_storage.search_range(m_storageBST_root, const_cast<void*>(src));            
            vk::Buffer srcbuf = src_node->GetBufferDevice();
            vk::Buffer dstbuf;
                                    
            const default_storage_node* dst_node = m_storage.search_range(m_storageBST_root, dst);

            if(dst_node == nullptr || dst_node->isHostVisible() == false)
            {
                // the dst adress is not known to vuda, assume that we are copying to pageable host mem
                // use staged buffer (pinned host cached memory) to perform internal copy before we copy to pageable host mem
                dstptr = get_cached_buffer(src_node->GetSize());
                dstbuf = dstptr->get_buffer();
                dst_memptr = dstptr->get_memptr();                
            }
            else // if(dst_node->isHostVisible() == true)
            {
                //
                // pinned memory target
                dstbuf = dst_node->GetBufferHost();
            }

            //
            // every thread can look up its command pool in the list
            std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
            thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

            std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
            vk::Queue q = m_queues.at(m_queueFamilyIndex).at(stream);

            pool->memcpyDevice(m_device, dstbuf, srcbuf, count, q, stream);
        }

        if(dstptr != nullptr)
        {
            //            
            // execute kernels that have access to modify src
            // the address of src will only be associated with one logical device.
            // the logical device associated with the calling thread must be the same for all calling threads accessing src
            //
            FlushQueue(tid, stream);

            //
            // copy the memory back to the staging buffer (host visible memory)            
            std::memcpy(dst, dst_memptr, count);

            //
            // release the internal cached node
            dstptr->set_free(true);
        }
    }

    /*inline std::vector<uint32_t> logical_device::GetStreamIdentifiers(const void* src)
    {
        //std::lock_guard<std::mutex> lck(*m_mtx);
        //return m_datacommabdbuffers[devPtr];

        std::vector<uint32_t> temp(1, 0);

        return temp;
    }*/

    inline void logical_device::FlushQueue(const std::thread::id tid, const uint32_t stream) const
    {
        //
        //
        //std::thread::id tid, uint32_t stream;
        //std::vector<uint32_t> stream_id = GetStreamIdentifiers(src);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);
        
        //
        // control queue submissions on this level        
        std::lock_guard<std::mutex> lck(*m_mtxQueues[stream]);        
        vk::Queue q = m_queues.at(m_queueFamilyIndex).at(stream);

        //
        // 
        /*std::ostringstream ostr;
        ostr << "thrd: " << std::this_thread::get_id() << ", locked queue: " << stream << std::endl;
        std::cout << ostr.str();
        ostr.str("");*/

        //
        // execute and wait for stream
        pool->Wait(m_device, q, stream);
        
        //
        //
        //ostr << "thrd: " << std::this_thread::get_id() << ", unlocked queue: " << stream << std::endl;
        //std::cout << ostr.str();
    }

    inline void logical_device::FlushEvent(const std::thread::id tid, const event_t event)
    {
        //
        // retrieve stream id associated with event
        //std::shared_lock<std::shared_mutex> lckEvents(*m_mtxEvents);
        std::unique_lock<std::shared_mutex> lck(*m_mtxEvents);
        const stream_t stream = m_events.at(event).getStream();

        //
        // control queue submissions on this level
        std::lock_guard<std::mutex> lckQueues(*m_mtxQueues[stream]);
        //vk::Queue q = m_queues[m_queueFamilyIndex][stream];
        vk::Queue q = m_queues.at(m_queueFamilyIndex).at(stream);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        const thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // execute stream
        pool->Execute(m_device, q, stream);
        //pool->ExecuteAndWait(m_device, q, stream);

        //
        // wait for event on host by using 'spin-lock'
        // [ pretty dirty ]
        int count = 0;
        while(m_device->getEventStatus(event) == vk::Result::eEventReset)        
            count++;        

        //
        // record tick host side
        m_events[event].tick();
        
        std::stringstream ostr;
        ostr << "vuda: event status attempt lock count: " << count << std::endl;
        std::cout << ostr.str();

        //
        // reset event
        m_device->resetEvent(event);
    }

    //
    // private
    //

    inline void logical_device::push_mem_node(default_storage_node* node)
    {
        //
        // protect storage tree
        std::unique_lock<std::shared_mutex> lck(*m_mtxResources);

        /*std::ostringstream ostr;
        ostr << std::this_thread::get_id() << ": took lock" << std::endl;
        std::cout << ostr.str();
        ostr.str("");*/

        //
        // push the node onto the bst storage tree data
        m_storageBST.emplace_back(node);

        //
        // insert the node in the bst tree
        m_storage.insert_node(m_storageBST_root, m_storageBST.back());

        //
        // show storage tree
        /*m_storage.walk_depth(m_storageBST_root);
        ostr << std::this_thread::get_id() << ": releasing lock" << std::endl;
        std::cout << ostr.str();*/
    }


    inline host_cached_node_internal* logical_device::get_cached_buffer(const vk::DeviceSize size)
    {
        //
        // lock
        std::unique_lock<std::mutex> lck(*m_mtxCached_internal);

        //
        // find free buffer        
        host_cached_node_internal *hcb = nullptr;

        for(int i = 0; i < (int)m_cachedBuffers.size(); ++i)
        {
            if(m_cachedBuffers[i]->is_free() == true && m_cachedBuffers[i]->get_size() >= size)
            {
                // atomic
                m_cachedBuffers[i]->set_free(false);
                hcb = m_cachedBuffers[i].get();
                break;
            }
        }
                
        //
        // if none are free, create a new one (potentially slow)
        if(hcb == nullptr)
        {
            std::unique_lock<std::mutex> lckalloc(*m_mtxAllocatorCached);

            m_cachedBuffers.push_back(std::make_unique<host_cached_node_internal>(m_physDevice, m_device.get(), size, m_allocatorCached));
            hcb = m_cachedBuffers.back().get();
        }

        return hcb;
    }

} //namespace vuda