#pragma once

namespace vuda
{
    /*
        the kernel program combines one shader module with a compute pipeline
    */

    class kernelprogram
    {
    public:

        //
        // create compute pipeline
        // - pipeline layout
        //   - create shader stage info
        //     - shader module
        //     - specialization info
        //   - descriptor set layout
        //     - descriptor set layout bindings
        //     - push constants
        //
        
        //
        // (1) Descriptor pool
        // (2) allocate descriptor sets
        // (3) descriptor buffer info
        //

        kernelprogram(const vk::UniqueDevice& device, const std::string& filename, const std::string& entryName, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, const std::vector<specialization>& specials, const int blocks, const int threads) :
            m_fileName(filename),
            m_entryName(entryName),
            m_blocks(blocks),
            m_threads(threads),
            m_code(ReadFile(filename)),
            m_shaderModule(device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, m_code.size(), reinterpret_cast<const uint32_t*>(m_code.data())))),
            m_descriptorSetLayout(device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, (uint32_t)bindings.size(), bindings.data()))),
            m_pipelineLayout(device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, 1, &m_descriptorSetLayout.get()))),
            m_computePipeline(device->createComputePipelineUnique(nullptr, vk::ComputePipelineCreateInfo({}, CreateShaderStageInfo(specials), m_pipelineLayout.get()))),
            m_descriptorPoolSize(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, (uint32_t)bindings.size())),

            //
            // allocate a descriptor sets used for multiple submissions of the kernel
            m_descriptorSetIndex(0),
            m_maxDescriptorSet( VUDA_MAX_KERNEL_DESCRIPTOR_SETS ),            

            m_descriptorSetLayouts(m_maxDescriptorSet, m_descriptorSetLayout.get()),
            m_descriptorPool(device->createDescriptorPoolUnique(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(vk::DescriptorPoolCreateFlagBits::eFreeDescriptorSet /*| vk::DescriptorPoolCreateFlagBits::eUpdateAfterBindEXT*/), m_maxDescriptorSet, 1, &m_descriptorPoolSize))),
            m_descriptorSet(device->allocateDescriptorSetsUnique(vk::DescriptorSetAllocateInfo(m_descriptorPool.get(), m_maxDescriptorSet, m_descriptorSetLayouts.data())))
        {
            /*
                - VkDescriptorSets are allocated from a "parent" VkDescriptorPool
                - descriptors allocated in different threads must come from different pools
                - But VkDescriptorSets from the same pool can be written to by different threads
            */

            // clear code module
            m_code.clear();

            // create mutex
            m_mtxDescriptorSets = std::make_unique<std::mutex>();

            /*std::ostringstream ostr;
            std::thread::id tid = std::this_thread::get_id();            
            ostr << tid << ": shader module created!" << std::endl;
            std::cout << ostr.str();*/
        }
        
        std::string GetFileName(void) const
        {
            return m_fileName;
        }

        std::string GetEntryName(void) const
        {
            return m_entryName;
        }

        bool UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const vk::UniqueCommandBuffer& commandBuffer, const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos) const
        {

            uint32_t descIndex;
            {
                //
                // take lock
                std::lock_guard<std::mutex> lck(*m_mtxDescriptorSets);

                //
                // check if we have sufficient descriptor sets available for recording
                if(m_descriptorSetIndex == m_maxDescriptorSet)
                {
                    m_descriptorSetIndex = 0;

                    //
                    // we should have a pool implementation of descriptor sets
                    assert(0);
                    return false;
                }

                //
                // allocate an index
                descIndex = m_descriptorSetIndex;
                
                m_descriptorSetIndex++;
            }

            //
            // descriptor update
            //std::vector<vk::DescriptorBufferInfo> descriptorBufferInfos(buffers.size());
            std::vector<vk::WriteDescriptorSet> writeDescriptorSets(descriptorBufferInfos.size());
            
            for(uint32_t index = 0; index < descriptorBufferInfos.size(); ++index)
            {
                //descriptorBufferInfos[index] = vk::DescriptorBufferInfo(buffers[index], 0, VK_WHOLE_SIZE);
                writeDescriptorSets[index] = vk::WriteDescriptorSet(m_descriptorSet[descIndex].get(), index, 0, 1, vk::DescriptorType::eStorageBuffer, 0, &descriptorBufferInfos[index], 0);
            }           
            device->updateDescriptorSets(writeDescriptorSets, nullptr);

            //
            // record the command buffer

            // [ redudancy: multiple computepipeline binds if function is called multiple times ]

            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, m_computePipeline.get());
            commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout.get(), 0, 1, &m_descriptorSet[descIndex].get(), 0, 0);
            commandBuffer->dispatch(m_blocks, 1, 1);

            //
            // insert pipeline barrier?
            //assert(0);
                        
            return true;
        }

    private:

        const vk::PipelineShaderStageCreateInfo& CreateShaderStageInfo(const std::vector<specialization>& specials)
        {
            //
            // dimensions
            const uint32_t numBlockDim = 3;
            const uint32_t numSpecializedConstants = numBlockDim + (uint32_t)specials.size();
            m_entries.resize(numSpecializedConstants);
            m_specialData.resize(numSpecializedConstants);
            //
            // hardcoded workgroup size
            m_workGroupSize[0] = (uint32_t)m_threads;
            m_workGroupSize[1] = 1;
            m_workGroupSize[2] = 1;

            size_t totalSize = 0;
            for(uint32_t i = 0; i < numBlockDim; ++i)
            {
                m_entries[i].constantID = i;
                m_entries[i].offset = (uint32_t)totalSize;
                m_entries[i].size = sizeof(uint32_t);
                totalSize += sizeof(uint32_t);

                m_specialData[i] = m_workGroupSize[i];
            }

            //
            // additional specialization constants
            uint32_t iter = 0;
            for(uint32_t i = numBlockDim; i < numSpecializedConstants; ++i)
            {
                m_entries[i].constantID = numBlockDim + specials[iter].m_mapEntry.constantID;
                m_entries[i].offset = (uint32_t)totalSize + specials[iter].m_mapEntry.offset;
                m_entries[i].size = specials[iter].m_mapEntry.size;
                m_specialData[i] = specials[iter].m_data;

                totalSize += specials[iter].m_mapEntry.size;
                ++iter;
            }

            m_specializationInfo.mapEntryCount = numSpecializedConstants;
            m_specializationInfo.pMapEntries = m_entries.data();
            m_specializationInfo.dataSize = totalSize;
            m_specializationInfo.pData = m_specialData.data();

            m_pipelineShaderStageCreateInfo.stage = vk::ShaderStageFlagBits::eCompute;
            m_pipelineShaderStageCreateInfo.module = m_shaderModule.get();
            m_pipelineShaderStageCreateInfo.pName = m_entryName.c_str();
            m_pipelineShaderStageCreateInfo.pSpecializationInfo = &m_specializationInfo;

            return m_pipelineShaderStageCreateInfo;
        }

        std::vector<char> ReadFile(const std::string& filename)
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

    private:

        //
        // identifiers
        std::string m_fileName;
        std::string m_entryName;

        //
        // blocks and threads
        int m_blocks;
        int m_threads;

        //
        //
        std::unique_ptr<std::mutex> m_mtxDescriptorSets;
        uint32_t m_maxDescriptorSet;
        mutable uint32_t m_descriptorSetIndex;

        //
        //
        uint32_t m_workGroupSize[3];
        std::vector<uint32_t> m_specialData;
        std::vector<vk::SpecializationMapEntry> m_entries;
        vk::SpecializationInfo m_specializationInfo;
        vk::PipelineShaderStageCreateInfo m_pipelineShaderStageCreateInfo;

        //
        // shader module
        std::vector<char> m_code;        
        vk::UniqueShaderModule m_shaderModule;        

        //
        // descriptor set layout        
        vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
        std::vector<vk::DescriptorSetLayout> m_descriptorSetLayouts;

        //    
        // pipeline layout
        vk::UniquePipelineLayout m_pipelineLayout;
        //
        // compute pipelines
        vk::UniquePipeline m_computePipeline;

        //
        // descriptor pool
        vk::DescriptorPoolSize m_descriptorPoolSize;
        vk::UniqueDescriptorPool m_descriptorPool;

        //
        // descriptor set        
        std::vector<vk::UniqueDescriptorSet> m_descriptorSet;

        /*//
        // command buffers
        std::vector<vk::UniqueCommandBuffer> m_commandBuffers;
        std::vector<vk::Fence> m_fences;
        std::vector<vk::UniqueFence> m_ufences;*/
    };

} //namespace vuda