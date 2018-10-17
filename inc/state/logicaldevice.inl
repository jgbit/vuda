
namespace vuda
{
    inline logical_device::logical_device(const vk::DeviceCreateInfo& deviceCreateInfo, const vk::PhysicalDevice& physDevice) :
        /*
            create (unique) logical device from the physical device specified
            create a command pool on the queue family index (assuming that there is only one family!)
        */
        m_storageBST_root(nullptr),
        m_physDevice(physDevice),
        m_device(physDevice.createDeviceUnique(deviceCreateInfo)),
        m_queueFamilyIndex(deviceCreateInfo.pQueueCreateInfos->queueFamilyIndex),
        m_queueComputeCount(deviceCreateInfo.pQueueCreateInfos->queueCount)/*,
        m_commandPool(m_device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), m_queueFamilyIndex))),
        m_commandBuffers(m_device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(m_commandPool.get(), vk::CommandBufferLevel::ePrimary, m_queueComputeCount))),
        m_commandBufferState(m_queueComputeCount, cbReset)*/
    {
        //
        // create unique mutexes

        m_mtxResources = std::make_unique<std::shared_mutex>();
        m_mtxKernels = std::make_unique<std::shared_mutex>();
        m_mtxCmdPools = std::make_unique<std::shared_mutex>();
        
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

    inline void logical_device::malloc(void** devPtr, size_t size)
    {
        std::unique_lock<std::shared_mutex> lck(*m_mtxResources);

        /*std::ostringstream ostr;
        ostr << std::this_thread::get_id() << ": took lock" << std::endl;        
        std::cout << ostr.str();
        ostr.str("");*/

        //
        // create new node in storage_bst 
        storage_buffer_node* node = new storage_buffer_node(m_physDevice, m_device, size);

        //
        // push the node onto the bst storage tree data
        m_storageBST.push_back(node);
        
        //
        // insert the node in the bst tree
        m_storage.insert_node(m_storageBST_root, m_storageBST.back());

        //
        // return the memory pointer
        (*devPtr) = node->key();

        //ostr << std::this_thread::get_id() << ": releasing lock, memptr: " << *devPtr << std::endl;
        //std::cout << ostr.str();
    }

    inline void logical_device::free(void* devPtr)
    {
        std::unique_lock<std::shared_mutex> lck(*m_mtxResources);

        //std::ostringstream ostr;
        //ostr << std::this_thread::get_id() << ": took lock" << std::endl;

        //
        // check whether the memory exists on the set device
        storage_buffer_node* node = static_cast<storage_buffer_node*>(m_storage.search(m_storageBST_root, devPtr));
        if(node == nullptr)
            throw std::runtime_error("Failed to find memory on the specified device!");

        
        //ostr << std::this_thread::get_id() << ": destroying memory with ptr: " << devPtr << std::endl;
        //std::cout << ostr.str();
        //m_storage.walk(m_storageBST_root);

        //
        // destroy the nodes satellite data
        node->destroy(m_device);
        
        //
        // remove node from the bst tree        
        storage_buffer_node* doomed = m_storage.delete_node(m_storageBST_root, node);

        //
        // remove the spliced node from the heap
        // (should perhaps recycle nodes? we know there is an upper limit on allocations)
        delete doomed;
        doomed = nullptr;
    }

    inline vk::DescriptorBufferInfo logical_device::GetBufferDescriptor(void* devPtr)
    {
        //
        // all threads can read buffer indices
        std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

        const storage_buffer_node* node = m_storage.search_range(m_storageBST_root, devPtr);
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
        //
        // check if the kernel is already created and registered
        // [ this should be optimized to be ~O(1) or compile-const ]

        std::vector<std::shared_ptr<kernel_interface>>::iterator it;
        {
            std::shared_lock<std::shared_mutex> lck(*m_mtxKernels);

            it = std::find_if(m_kernels.begin(), m_kernels.end(), [&filename, &entry](std::shared_ptr<kernel_interface>& kernel)
            {
                return (kernel->GetFileName() == filename && kernel->GetEntryName() == entry);
            });
        }

        //
        // if the kernel does not exist, it has to be created
        // lock all access to kernels - all pointers to m_kernels become invalid
        if(it == m_kernels.end())
        {
            std::unique_lock<std::shared_mutex> lck(*m_mtxKernels);

            m_kernels.push_back(std::make_shared<kernelprogram<specials.bytesize>>(m_device, filename, entry, bindings, specials));
            it = std::prev(m_kernels.end());
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
        
                //
                // get specialized kernel
                kernelprogram<specials.bytesize>* kernel = static_cast<kernelprogram<specials.bytesize>*>((*it).get());

                pool->UpdateDescriptorAndCommandBuffer<specials.bytesize, specialTypes...>(m_device, *kernel, specials, bufferDescriptors, blocks, stream);
            }
        }
    }

    /*template <typename... specialTypes>
    inline uint64_t logical_device::CreateKernel(char const* filename, char const* entry, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, specialization<specialTypes...>& specials, int blocks)
    {
        //
        // check if the kernel is already created and registered
        // [ this should be optimized to be ~O(1) or compile-const ]
                
        const size_t specializationByteSize = specials.bytesize;

        std::vector<std::shared_ptr<kernel_interface>>::iterator it;
        {
            std::shared_lock<std::shared_mutex> lck(*m_mtxKernels);

            it = std::find_if(m_kernels.begin(), m_kernels.end(), [&filename, &entry](std::shared_ptr<kernel_interface>& kernel) 
            {
                return (kernel->GetFileName() == filename && kernel->GetEntryName() == entry);
            });

            //
            // retrieve specialization of the kernel            
            kernel->is_special_known(specials.data());

            if(it != m_kernels.end())
                return std::distance(m_kernels.begin(), it);
        }
        
        //
        // if the kernel does not exist, it is created
        // lock all access to kernels - all pointers to m_kernels become invalid
        {
            std::unique_lock<std::shared_mutex> lck(*m_mtxKernels);

            m_kernels.push_back(std::make_shared<kernelprogram<specializationByteSize>>(m_device, filename, entry, bindings, specials, blocks));
            it = std::prev(m_kernels.end());
        }

        return std::distance(m_kernels.begin(), it);
    }*/

    inline void logical_device::WaitOn(void)
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
        m_thrdCommandPools.insert({ tid, thrdcmdpool(m_device, m_queueFamilyIndex, m_queueComputeCount) });
    }

    inline void logical_device::memcpyToDevice(const std::thread::id tid, void* dst, const size_t count, const uint32_t stream)
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // all threads can read from the memory resources on the logical device
        std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

        //
        // internal copy in the node
        vk::Buffer dstbuf = m_storage.search_range(m_storageBST_root, dst)->GetBufferDevice();
        vk::Buffer srcbuf = m_storage.search_range(m_storageBST_root, dst)->GetBufferHost();
        pool->memcpyDevice(m_device, dstbuf, srcbuf, count, stream);
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
        std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

        //
        // copy from node to node
        vk::Buffer dstbuf = m_storage.search_range(m_storageBST_root, dst)->GetBufferDevice();
        vk::Buffer srcbuf = m_storage.search_range(m_storageBST_root, const_cast<void*>(src))->GetBufferDevice();
        pool->memcpyDevice(m_device, dstbuf, srcbuf, count, stream);
    }

    inline void logical_device::memcpyToHost(const std::thread::id tid, const void* src, const size_t count, const uint32_t stream)
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // all threads can read from the memory resources on the logical device
        std::shared_lock<std::shared_mutex> lck(*m_mtxResources);

        //
        // internal copy in the node
        vk::Buffer dstbuf = m_storage.search_range(m_storageBST_root, const_cast<void*>(src))->GetBufferHost();
        vk::Buffer srcbuf = m_storage.search_range(m_storageBST_root, const_cast<void*>(src))->GetBufferDevice();
        pool->memcpyDevice(m_device, dstbuf, srcbuf, count, stream);
    }

    /*inline void logical_device::UpdateDescriptorAndCommandBuffer(const std::thread::id tid, const uint64_t kernelIndex, const std::vector<void*>& memaddr, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t stream)
    {
        assert(stream >= 0 && stream < m_queueComputeCount);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        thrdcmdpool *pool = &m_thrdCommandPools.at(tid);

        //
        // every thread can read the kernels array        
        std::shared_lock<std::shared_mutex> lckKernels(*m_mtxKernels);
        pool->UpdateDescriptorAndCommandBuffer(m_device, *m_kernels[kernelIndex], bufferDescriptors, stream);

        //
        // create (tid, stream) identifiers
        //std::unique_lock<std::shared_mutex> lck(*m_mtxResourceKernelAccess);        
    }*/

    inline std::vector<uint32_t> logical_device::GetStreamIdentifiers(const void* src)
    {
        //std::lock_guard<std::mutex> lck(*m_mtx);
        //return m_datacommabdbuffers[devPtr];

        std::vector<uint32_t> temp(1, 0);

        return temp;
    }

    inline void logical_device::FlushQueue(const std::thread::id tid, const uint32_t stream)
    {
        //
        //
        //std::thread::id tid, uint32_t stream;
        //std::vector<uint32_t> stream_id = GetStreamIdentifiers(src);

        //
        // every thread can look up its command pool in the list
        std::shared_lock<std::shared_mutex> lckCmdPools(*m_mtxCmdPools);
        thrdcmdpool *pool = &m_thrdCommandPools.at(tid);
        
        //
        // control queue submissions on this level        
        std::lock_guard<std::mutex> lck(*m_mtxQueues[stream]);
        vk::Queue q = m_queues[m_queueFamilyIndex][stream];

        //
        // hello
        /*std::ostringstream ostr;
        ostr << "thrd: " << std::this_thread::get_id() << ", locked queue: " << stream << std::endl;
        std::cout << ostr.str();
        ostr.str("");*/

        //
        // execute and wait for stream
        pool->ExecuteAndWait(m_device, q, stream);
        
        //
        //
        //ostr << "thrd: " << std::this_thread::get_id() << ", unlocked queue: " << stream << std::endl;
        //std::cout << ostr.str();
    }
    

} //namespace vuda