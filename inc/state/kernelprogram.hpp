#pragma once

namespace vuda
{
    namespace detail
    {

        /*
            the kernel program combines one shader module with a compute pipeline
        */

        class kernel_interface
        {
        public:

            kernel_interface(const std::string& filename, const std::string& entryName) :
                m_fileName(filename),
                m_entryName(entryName)
            {

            }

            //virtual bool UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const vk::UniqueCommandBuffer& commandBuffer, const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos, const uint8_t* specialSignature) const = 0;                

            std::string GetFileName(void) const
            {
                return m_fileName;
            }

            std::string GetEntryName(void) const
            {
                return m_entryName;
            }

        protected:
            //
            // identifiers
            std::string m_fileName;
            std::string m_entryName;
        };

        template <size_t specializationByteSize>
        class kernelprogram : public kernel_interface
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

            template <typename... specialTypes>
            kernelprogram(const vk::UniqueDevice& device, const std::string& filename, const std::string& entryName, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, specialization<specialTypes...>& specials) :
                kernel_interface(filename, entryName),            
                m_code(ReadFile(filename)),
                m_shaderModule(device->createShaderModuleUnique(vk::ShaderModuleCreateInfo({}, m_code.size(), reinterpret_cast<const uint32_t*>(m_code.data())))),
                m_descriptorSetLayout(device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, (uint32_t)bindings.size(), bindings.data()))),
                m_pipelineLayout(device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, 1, &m_descriptorSetLayout.get()))),                                
                m_descriptor_pool(device.get(), (uint32_t)bindings.size(), m_descriptorSetLayout.get())
            {
                //
                // clear code module
                m_code.clear();

                //
                // create mutex
                m_mtxComputePipelines = std::make_unique<std::shared_mutex>();
            }

            template <typename... specialTypes>
            bool UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const vk::UniqueCommandBuffer& commandBuffer, const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos, specialization<specialTypes...>& specials, const dim3 blocks) const
            {
                //
                // get available descriptor set
                vk::DescriptorSet descriptor_set;
                vuda::detail::descriptor_pool_allocator<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet>* pool;
                m_descriptor_pool.get_element(&descriptor_set, &pool);

                //
                // descriptor update                
                std::vector<vk::WriteDescriptorSet> writeDescriptorSets(descriptorBufferInfos.size());
                for(uint32_t index = 0; index < descriptorBufferInfos.size(); ++index)
                {
                    //writeDescriptorSets[index] = vk::WriteDescriptorSet(m_descriptorSet[descIndex].get(), index, 0, 1, vk::DescriptorType::eStorageBuffer, 0, &descriptorBufferInfos[index], 0);
                    writeDescriptorSets[index] = vk::WriteDescriptorSet(descriptor_set, index, 0, 1, vk::DescriptorType::eStorageBuffer, 0, &descriptorBufferInfos[index], 0);
                }           
                device->updateDescriptorSets(writeDescriptorSets, nullptr);

                //
                // record the command buffer
                // [ redudancy: multiple computepipeline binds if function is called multiple times ]
                commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, GetSpecializedPipeline(device, specials));
                commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout.get(), 0, 1, &descriptor_set, 0, 0);
                commandBuffer->dispatch(blocks.x, blocks.y, blocks.z);

                return true;
            }

        private:

            template <typename... specialTypes>
            vk::Pipeline GetSpecializedPipeline(const vk::UniqueDevice& device, specialization<specialTypes...>& specials) const
            {
                {
                    std::shared_lock<std::shared_mutex> lck(*m_mtxComputePipelines);
                
                    auto it = m_specializedComputePipelines.find(specials.data());
                        
                    if(it != m_specializedComputePipelines.end())                    
                        return (*it).second.get();
                }

                return CreatePipeline(device, specials);
            }

            template <typename... specialTypes>
            vk::Pipeline CreatePipeline(const vk::UniqueDevice& device, specialization<specialTypes...>& specials) const
            {
                std::unique_lock<std::shared_mutex> lck(*m_mtxComputePipelines);

                vk::PipelineShaderStageCreateInfo info = vk::PipelineShaderStageCreateInfo()
                    .setStage(vk::ShaderStageFlagBits::eCompute)
                    .setModule(m_shaderModule.get())
                    .setPName(m_entryName.c_str())
                    .setPSpecializationInfo(&specials.info());

                //std::pair<std::map<std::array<uint8_t, specializationByteSize>, vk::UniquePipeline>::iterator, bool> ret;
                auto ret = m_specializedComputePipelines.insert(std::make_pair(specials.data(), device->createComputePipelineUnique(nullptr, vk::ComputePipelineCreateInfo({}, info, m_pipelineLayout.get()))));

                return (*ret.first).second.get();
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
            // shader module
            std::vector<char> m_code;
            vk::UniqueShaderModule m_shaderModule;

            //
            // descriptor set layout
            vk::UniqueDescriptorSetLayout m_descriptorSetLayout;
        
            //    
            // pipeline layout
            vk::UniquePipelineLayout m_pipelineLayout;

            //
            // compute pipelines
            //vk::UniquePipeline m_computePipeline;
            // [ for now use map implementation of specialization constants -> pipeline ]
            std::unique_ptr<std::shared_mutex> m_mtxComputePipelines;
            mutable std::map<std::array<uint8_t, specializationByteSize>, vk::UniquePipeline> m_specializedComputePipelines;
            //std::unordered_map<std::array<uint8_t, specializationByteSize>, vk::UniquePipeline, hash_function> m_specializedComputePipelines;
        
            //
            // pool of finite descriptor pools
            mutable pool_finite_sets<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet, descriptor_pool_allocator<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet>, vk::Device, uint32_t, vk::DescriptorSetLayout> m_descriptor_pool;
        };

    } //namespace detail
} //namespace vuda