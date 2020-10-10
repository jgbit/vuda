#pragma once

namespace vuda
{
    namespace detail
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

            thrdcmdpool(const vk::UniqueDevice& device, const uint32_t queueFamilyIndex, const uint32_t queueComputeCount) :            
                m_commandPool(device->createCommandPoolUnique(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(vk::CommandPoolCreateFlagBits::eResetCommandBuffer), queueFamilyIndex))),
                m_commandBuffers(device->allocateCommandBuffersUnique(vk::CommandBufferAllocateInfo(m_commandPool.get(), vk::CommandBufferLevel::ePrimary, queueComputeCount))),
                m_commandBufferState(queueComputeCount, cbReset),

                m_queryPool(device->createQueryPoolUnique(vk::QueryPoolCreateInfo(vk::QueryPoolCreateFlags(), vk::QueryType::eTimestamp, VUDA_MAX_QUERY_COUNT, vk::QueryPipelineStatisticFlags()))),
                m_queryIndex(0)
            {
                //
                // create unique mutexes

                /*m_mtxCommandBuffers.resize(queueComputeCount);
                for(unsigned int i = 0; i < queueComputeCount; ++i)
                    m_mtxCommandBuffers[i] = std::make_unique<std::mutex>();*/

                //
                // create fences

                m_ufences.reserve(queueComputeCount);
                for(unsigned int i = 0; i < queueComputeCount; ++i)
                    m_ufences.push_back(device->createFenceUnique(vk::FenceCreateFlags()));
            }

            /*
                public synchronized interface
            */

            void SetEvent(const vk::UniqueDevice& device, const event_t event, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                //
                //
                CheckStateAndBeginCommandBuffer(device, stream);

                //
                //
                m_commandBuffers[stream]->setEvent(event, vk::PipelineStageFlagBits::eBottomOfPipe);
            }

            uint32_t GetQueryID(void) const
            {
                assert(m_queryIndex != VUDA_MAX_QUERY_COUNT);
                return m_queryIndex++;
            }

            void WriteTimeStamp(const vk::UniqueDevice& device, const uint32_t queryID, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                //
                // submit write time stamp command
                CheckStateAndBeginCommandBuffer(device, stream);
            
                // reset
                m_commandBuffers[stream]->resetQueryPool(m_queryPool.get(), queryID, 1);

                // write
                m_commandBuffers[stream]->writeTimestamp(vk::PipelineStageFlagBits::eBottomOfPipe, m_queryPool.get(), queryID);
            }

            uint64_t GetQueryPoolResults(const vk::UniqueDevice& device, const uint32_t queryID) const
            {
                // vkGetQueryPoolResults(sync)
                // vkCmdCopyQueryPoolResults(async)
            
                const uint32_t numQueries = 1; // VUDA_MAX_QUERY_COUNT;
                uint64_t result[numQueries];
                size_t stride = sizeof(uint64_t);
                size_t size = numQueries * stride;
            
                //
                // vkGetQueryPoolResults will wait for the results to be available when VK_QUERY_RESULT_WAIT_BIT is specified
                vk::Result res =
                device->getQueryPoolResults(m_queryPool.get(), queryID, numQueries, size, &result[0], stride, vk::QueryResultFlagBits::e64 | vk::QueryResultFlagBits::eWait);
                assert(res == vk::Result::eSuccess);

                return result[0];
            }

            /*void ResetQueryPool(const vk::UniqueDevice& device, const uint32_t queryID, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                //
                // reset query
                CheckStateAndBeginCommandBuffer(device, stream);
                m_commandBuffers[stream]->resetQueryPool(m_queryPool.get(), queryID, 1);
            }*/

            void memcpyDevice(const vk::UniqueDevice& device, const vk::Buffer& bufferDst, const vk::DeviceSize dstOffset, const vk::Buffer& bufferSrc, const vk::DeviceSize srcOffset, const vk::DeviceSize size, const vk::Queue& queue, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);
            
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
                    .setSrcOffset(srcOffset)
                    .setDstOffset(dstOffset)
                    .setSize(size);

                //
                // the order of src and dst is interchanged compared to memcpy
                m_commandBuffers[stream]->copyBuffer(bufferSrc, bufferDst, copyRegion);

                //
                // hello there
                /*std::ostringstream ostr;
                ostr << std::this_thread::get_id() << ", commandbuffer: " << &m_commandBuffers[stream] << ", src: " << bufferSrc << ", dst: " << bufferDst << ", copy region: srcOffset: " << copyRegion.srcOffset << ", dstOffset: " << copyRegion.dstOffset << ", size: " << copyRegion.size << std::endl;
                std::cout << ostr.str();*/

                //
                // insert pipeline barrier?

                /*vk::BufferMemoryBarrier bmb(vk::AccessFlagBits::eTransferRead, vk::AccessFlagBits::eTransferWrite, VK_QUEUE_FAMILY_IGNORED, VK_QUEUE_FAMILY_IGNORED, bufferDst, dstOffset, size);

                m_commandBuffers[stream]->pipelineBarrier(
                    vk::PipelineStageFlagBits::eTransfer,
                    vk::PipelineStageFlagBits::eBottomOfPipe,
                    vk::DependencyFlagBits::eByRegion,
                    0, nullptr,
                    1, &bmb,
                    0, nullptr);*/

                //
                // execute the command buffer
                ExecuteQueue(device, queue, stream);

                //ostr << "thrd: " << std::this_thread::get_id() << ", pooladrr: " << this << ", unlocking command buffer: " << stream << std::endl;
                //std::cout << ostr.str();
            }

            template <size_t specializationByteSize, typename... specialTypes, size_t bindingSize>
            void UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const kernelprogram<specializationByteSize>& kernel, const specialization<specialTypes...>& specials, const std::array<vk::DescriptorBufferInfo, bindingSize>& bufferDescriptors, const dim3 blocks, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                //
                // check state of command buffer and see if we should call begin
                CheckStateAndBeginCommandBuffer(device, stream);
            
                //
                // record command buffer
                kernel.UpdateDescriptorAndCommandBuffer(device, m_commandBuffers[stream], bufferDescriptors, specials, blocks);

                //
                // insert (buffer) memory barrier
                // [ memory barriers are created for each resource, it would be better to apply the barriers based on readonly, writeonly information ]

                const uint32_t numbuf = (uint32_t)bufferDescriptors.size();
                std::vector<vk::BufferMemoryBarrier> bmb(numbuf);
                for(uint32_t i=0; i<numbuf; ++i)
                {
                    bmb[i] = vk::BufferMemoryBarrier(
                        vk::AccessFlagBits::eShaderWrite,
                        vk::AccessFlagBits::eShaderRead,
                        VK_QUEUE_FAMILY_IGNORED,
                        VK_QUEUE_FAMILY_IGNORED,
                        bufferDescriptors[i].buffer,
                        bufferDescriptors[i].offset,
                        bufferDescriptors[i].range);
                }

                m_commandBuffers[stream]->pipelineBarrier(
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::PipelineStageFlagBits::eComputeShader,
                    vk::DependencyFlagBits::eByRegion,
                    0, nullptr,
                    numbuf, bmb.data(),
                    0, nullptr);
                        
                //
                // statistics
                /*std::ostringstream ostr;
                ostr << "tid: " << std::this_thread::get_id() << ", stream_id: " << stream << ", command buffer addr: " << &m_commandBuffers[stream] << std::endl;
                std::cout << ostr.str();*/

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

            void Execute(const vk::UniqueDevice& device, const vk::Queue& queue, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                ExecuteQueue(device, queue, stream);
            }

            /*void Wait(const vk::UniqueDevice& device, const vk::Queue& queue, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);
                        
                WaitAndReset(device, stream);
            }*/

            void ExecuteAndWait(const vk::UniqueDevice& device, const vk::Queue& queue, const stream_t stream) const
            {
                //
                // lock access to the streams commandbuffer
                //std::lock_guard<std::mutex> lck(*m_mtxCommandBuffers[stream]);

                ExecuteQueue(device, queue, stream);
                WaitAndReset(device, stream);
            }

        private:

            /*
                private non-synchronized implementation
                -  assumes m_mtxCommandBuffers[stream] is locked
            */

            void CheckStateAndBeginCommandBuffer(const vk::UniqueDevice& device, const stream_t stream) const
            {
                //
                // assumes m_mtxCommandBuffers[stream] is locked
            
                commandBufferStateFlags state = m_commandBufferState[stream];

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

            void BeginRecordingCommandBuffer(const stream_t stream) const
            {
                //
                // assumes m_mtxCommandBuffers[stream] is locked

                vk::CommandBufferBeginInfo commandBufferBeginInfo = vk::CommandBufferBeginInfo()
                    .setFlags(vk::CommandBufferUsageFlagBits::eOneTimeSubmit)
                    .setPInheritanceInfo(nullptr);

                m_commandBuffers[stream]->begin(commandBufferBeginInfo);
                m_commandBufferState[stream] = cbRecording;
            }

            void ExecuteQueue(const vk::UniqueDevice& device, const vk::Queue& queue, const stream_t stream) const
            {
                //
                // assumes m_mtxCommandBuffers[stream] is locked

                if(m_commandBufferState[stream] == cbRecording)
                {
                    //
                    // end recording and submit
                    m_commandBuffers[stream]->end();

                    //
                    // submit command buffer to compute queue
                    /*
                    https://www.khronos.org/registry/vulkan/specs/1.1-extensions/man/html/vkQueueSubmit.html
                    Each element of the pCommandBuffers member of each element of pSubmits must have been allocated from a VkCommandPool that was created for the same queue family queue belongs to.
                    */
                    queue.submit(vk::SubmitInfo(0, nullptr, nullptr, 1, &m_commandBuffers[stream].get(), 0, nullptr), m_ufences[stream].get());
                    m_commandBufferState[stream] = cbSubmitted;
                }
            }

            void WaitAndReset(const vk::UniqueDevice& device, const stream_t stream) const
            {
                //
                // assumes m_mtxCommandBuffers[stream] is locked

                if(m_commandBufferState[stream] == cbSubmitted)
                {
                    //
                    // for now just wait for the queue to become idle
                    vk::Result res = device->waitForFences(1, &m_ufences[stream].get(), VK_FALSE, (std::numeric_limits<uint64_t>::max)());
                    assert(res == vk::Result::eSuccess);
                    device->resetFences(1, &m_ufences[stream].get());

                    //
                    // reset command buffer
                    m_commandBuffers[stream]->reset(vk::CommandBufferResetFlags());
                    m_commandBufferState[stream] = cbReset;
                }
            }

            /*std::vector<uint32_t> GetStreamList(const void* src)
            {
                std::lock_guard<std::mutex> lck(*m_mtxResourceDependency);
                std::vector<uint32_t> list = m_src2stream[src];
                m_src2stream.erase(src);
                return list;
            }*/

        private:
        
            //std::vector<std::unique_ptr<std::mutex>> m_mtxCommandBuffers;
            std::vector<vk::UniqueFence> m_ufences;

            vk::UniqueCommandPool m_commandPool;        
            std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
            mutable std::vector<commandBufferStateFlags> m_commandBufferState;

            //
            // time stamp queries
            vk::UniqueQueryPool m_queryPool;
            mutable std::atomic<uint32_t> m_queryIndex;
            //mutable std::array<std::atomic<uint32_t>, VUDA_MAX_QUERY_COUNT> m_querytostream;
        

            //
            // resource management
            //std::unique_ptr<std::mutex> m_mtxResourceDependency;
            //std::unordered_map<const void*, std::vector<uint32_t>> m_src2stream;
        };

    } //namespace detail
} //namespace vuda