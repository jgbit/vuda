#pragma once

namespace vuda
{

    class thrdcmdpool
    {
        //friend class logical_device;
        
    public:

        
        /*

        For each thread that sets/uses the device a single command pool is created
         - this pool have m_queueComputeCount command buffers allocated.

        This way,
         - VkCommandBuffers are allocated from a "parent" VkCommandPool
         - VkCommandBuffers written to in different threads must come from different pools

        =======================================================
        command buffers \ threads : 0 1 2 3 4 ... #n
                                    0
                                    1
                                    .
                                    .
                                    .
                                    m_queueComputeCount
        =======================================================

        */

        thrdcmdpool(const vk::UniqueDevice& device, uint32_t queueFamilyIndex, uint32_t queueComputeCount) :
            m_commandPool(device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex))),
            m_commandBuffers(device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(m_commandPool.get(), vk::CommandBufferLevel::ePrimary, queueComputeCount))),
            m_commandBufferState(queueComputeCount, cbReset)
        {
            //
            // create unique mutexes

            m_mtxCommandBuffers.resize(queueComputeCount);
            for(unsigned int i = 0; i < queueComputeCount; ++i)
                m_mtxCommandBuffers[i] = std::make_unique<std::mutex>();

            //m_mtxResourceDependency = std::make_unique<std::mutex>();

            //
            // create fences

            m_ufences.reserve(queueComputeCount);
            for(unsigned int i = 0; i < queueComputeCount; ++i)
                m_ufences.push_back(device->createFenceUnique(vk::FenceCreateFlags()));
        }

        /*
            public synchronized interface
        */

        void memcpyDevice(const vk::UniqueDevice& device, const vk::Buffer& bufferDst, const vk::Buffer& bufferSrc, const vk::DeviceSize size, const uint32_t stream)
        {
            //
            // lock access to the streams commandbuffer
            std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);
            
            /*//
            // hello
            std::ostringstream ostr;
            ostr << "thrd: " << std::this_thread::get_id() << ", pooladrr: " << this << ", locked and modifying command buffer: " << stream << std::endl;
            std::cout << ostr.str();
            ostr.str("");*/

            //
            // check state of command buffer and see if we should call begin
            CheckStateAndBeginCommandBuffer(device, stream);
            
            //
            // submit copy buffer call to command buffer
            
            vk::BufferCopy copyRegion = vk::BufferCopy()
                .setSrcOffset(0)
                .setDstOffset(0)
                .setSize(size);

            //
            // the order of src and dst is interchanged compared to memcpy
            m_commandBuffers[stream]->copyBuffer(bufferSrc, bufferDst, copyRegion);

            //
            // insert pipeline barrier?            
            //assert(0);

            //ostr << "thrd: " << std::this_thread::get_id() << ", pooladrr: " << this << ", unlocking command buffer: " << stream << std::endl;
            //std::cout << ostr.str();            
        }

        template <size_t specializationByteSize, typename... specialTypes>
        void UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const kernelprogram<specializationByteSize>& kernel, specialization<specialTypes...>& specials, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t blocks, const uint32_t stream)
        {
            //
            // lock access to the streams commandbuffer
            std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

            //
            // check state of command buffer and see if we should call begin
            CheckStateAndBeginCommandBuffer(device, stream);
            
            //
            // record command buffer
            bool ret = kernel.UpdateDescriptorAndCommandBuffer(device, m_commandBuffers[stream], bufferDescriptors, specials, blocks);

            /*//
            // if the recording failed, it is solely due to the limited amount of descriptor sets
            if(ret == false)
            {
                //
                // if we are doing a new recording no need to end and execute
                if(newRecording == false)
                {
                    std::ostringstream ostr;
                    std::thread::id tid = std::this_thread::get_id();
                    ostr << tid << ": ran out of descriptors! Got to execute now and retry!" << std::endl;
                    std::cout << ostr.str();

                    // end recording/execute, wait, begin recording
                    ExecuteQueue(device, queueFamilyIndex, stream);
                    WaitAndReset(device, stream);
                    BeginRecordingCommandBuffer(stream);
                }

                //
                // record
                ret = kernel.UpdateDescriptorAndCommandBuffer(device, m_commandBuffers[stream], bufferDescriptors);

                assert(ret == true);
            }*/
        }

        void ExecuteAndWait(const vk::UniqueDevice& device, const vk::Queue& queue, const uint32_t stream)
        {
            //
            // lock access to the streams commandbuffer
            std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

            ExecuteQueue(device, queue, stream);
            WaitAndReset(device, stream);
        }

    private:

        /*
            private non-synchronized implementation
            -  assumes m_mtxCommandBuffers[stream] is locked
        */

        void CheckStateAndBeginCommandBuffer(const vk::UniqueDevice& device, const uint32_t stream)
        {
            //
            // assumes m_mtxCommandBuffers[stream] is locked
            
            commandBufferState state = m_commandBufferState[stream];

            if(state == cbReset)
            {
                // we can begin a new recording
                BeginRecordingCommandBuffer(stream);
            }
            else if(state == cbSubmitted)
            {
                // wait on completion and start new recording
                WaitAndReset(device, stream);
                BeginRecordingCommandBuffer(stream);
            }
            else
            {
                // this is a continued recording call
                //newRecording = false;
            }
        }

        void BeginRecordingCommandBuffer(const uint32_t stream)
        {
            //
            // assumes m_mtxCommandBuffers[stream] is locked

            vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo()
                .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
                .setPInheritanceInfo(nullptr);

            m_commandBuffers[stream]->begin(commandBufferBeginInfo);
            m_commandBufferState[stream] = cbRecording;
        }

        void ExecuteQueue(const vk::UniqueDevice& device, const vk::Queue& queue, const uint32_t stream)
        {
            //
            // assumes m_mtxCommandBuffers[stream] is locked

            if(m_commandBufferState[stream] == cbRecording)
            {
                //
                // end recording and submit
                m_commandBuffers[stream]->end();
                //m_commandBufferState[stream] = cbReadyForExecution;

                //
                // submit program to compute queue

                // Each element of the pCommandBuffers member of each element of pSubmits must have been allocated from a VkCommandPool that was created for the same queue family queue belongs to.

                //vk::Queue queue = device->getQueue(queueFamilyIndex, stream);
                queue.submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &m_commandBuffers[stream].get(), 0, nullptr), m_ufences[stream].get());
                m_commandBufferState[stream] = cbSubmitted;
            }
        }

        void WaitAndReset(const vk::UniqueDevice& device, const uint32_t stream)
        {
            //
            // assumes m_mtxCommandBuffers[stream] is locked

            //
            // for now just wait for the queue to become idle        
            device->waitForFences(1, &m_ufences[stream].get(), VK_FALSE, std::numeric_limits<uint64_t>::max());
            device->resetFences(1, &m_ufences[stream].get());

            //
            // reset command buffer
            m_commandBuffers[stream]->reset(vk::CommandBufferResetFlags());
            m_commandBufferState[stream] = cbReset;
        }

        /*std::vector<uint32_t> GetStreamList(const void* src)
        {
            std::lock_guard<std::mutex> lck(*m_mtxResourceDependency);
            std::vector<uint32_t> list = m_src2stream[src];
            m_src2stream.erase(src);
            return list;
        }*/

    private:

        vk::UniqueCommandPool m_commandPool;
        std::vector<std::unique_ptr<std::mutex>> m_mtxCommandBuffers;
        std::vector<commandBufferState> m_commandBufferState;
        std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
        std::vector<vk::UniqueFence> m_ufences;

        //
        // resource management
        //std::unique_ptr<std::mutex> m_mtxResourceDependency;
        //std::unordered_map<const void*, std::vector<uint32_t>> m_src2stream;
    };

} //namespace vuda