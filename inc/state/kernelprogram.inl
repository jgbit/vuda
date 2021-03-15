
namespace vuda
{
    namespace detail
    {

        //
        // public:
        //

        template <size_t specializationByteSize>
        template <typename... specialTypes, size_t bindingSize>
        kernelprogram<specializationByteSize>::kernelprogram(const vk::UniqueDevice& device, const vk::ShaderModule& shaderModule, const std::string& entryName, const std::array<vk::DescriptorSetLayoutBinding, bindingSize>& bindings, const specialization<specialTypes...>& specials) :
            kernel_interface(entryName, shaderModule),
            m_descriptorSetLayout(device->createDescriptorSetLayoutUnique(vk::DescriptorSetLayoutCreateInfo({}, (uint32_t)bindingSize, bindings.data()))),
            m_pipelineLayout(device->createPipelineLayoutUnique(vk::PipelineLayoutCreateInfo({}, 1, &m_descriptorSetLayout.get()))),
            m_descriptor_pool(device.get(), (uint32_t)bindingSize, m_descriptorSetLayout.get())
        {
            //
            // create mutex
            m_mtxComputePipelines = std::make_unique<std::shared_mutex>();
        }

        template <size_t specializationByteSize>
        template <typename... specialTypes, size_t bindingSize>
        bool kernelprogram<specializationByteSize>::UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const vk::UniqueCommandBuffer& commandBuffer, const std::array<vk::DescriptorBufferInfo, bindingSize>& descriptorBufferInfos, const specialization<specialTypes...>& specials, const dim3 blocks) const
        {
            //
            // get available descriptor set
            vk::DescriptorSet descriptor_set;
            vuda::detail::descriptor_pool_allocator<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet>* pool;
            m_descriptor_pool.get_element(&descriptor_set, &pool);

            //
            // descriptor update
            std::vector<vk::WriteDescriptorSet> writeDescriptorSets(bindingSize);
            for(uint32_t index = 0; index < bindingSize; ++index)
            {
                //writeDescriptorSets[index] = vk::WriteDescriptorSet(m_descriptorSet[descIndex].get(), index, 0, 1, vk::DescriptorType::eStorageBuffer, 0, &descriptorBufferInfos[index], 0);
                writeDescriptorSets[index] = vk::WriteDescriptorSet(descriptor_set, index, 0, 1, vk::DescriptorType::eStorageBuffer, 0, &descriptorBufferInfos[index], 0);
            }
            device->updateDescriptorSets(writeDescriptorSets, nullptr);

            //
            // record the command buffer
            // [ redundancy: multiple computepipeline binds if function is called multiple times ]
            commandBuffer->bindPipeline(vk::PipelineBindPoint::eCompute, GetSpecializedPipeline(device, specials));
            commandBuffer->bindDescriptorSets(vk::PipelineBindPoint::eCompute, m_pipelineLayout.get(), 0, 1, &descriptor_set, 0, 0);
            commandBuffer->dispatch(blocks.x, blocks.y, blocks.z);

            return true;
        }

        //
        // private:
        //

        template <size_t specializationByteSize>
        template <typename... specialTypes>
        vk::Pipeline kernelprogram<specializationByteSize>::GetSpecializedPipeline(const vk::UniqueDevice& device, const specialization<specialTypes...>& specials) const
        {
            {
                std::shared_lock<std::shared_mutex> lck(*m_mtxComputePipelines);

                auto it = m_specializedComputePipelines.find(specials.data());

                if(it != m_specializedComputePipelines.end())
                    return (*it).second.get();
            }

            return CreatePipeline(device, specials);
        }

        template <size_t specializationByteSize>
        template <typename... specialTypes>
        vk::Pipeline kernelprogram<specializationByteSize>::CreatePipeline(const vk::UniqueDevice& device, const specialization<specialTypes...>& specials) const
        {
            std::lock_guard<std::shared_mutex> lck(*m_mtxComputePipelines);

            vk::PipelineShaderStageCreateInfo info = vk::PipelineShaderStageCreateInfo()
                .setStage(vk::ShaderStageFlagBits::eCompute)
                .setModule(m_shaderModule)
                .setPName(m_entryName.c_str())
                .setPSpecializationInfo(&specials.info());

            //std::pair<std::map<std::array<uint8_t, specializationByteSize>, vk::UniquePipeline>::iterator, bool> ret;
            auto ret = m_specializedComputePipelines.insert(std::make_pair(specials.data(), device->createComputePipelineUnique(nullptr, vk::ComputePipelineCreateInfo({}, info, m_pipelineLayout.get())).value));

            return (*ret.first).second.get();
        }

        /*template <size_t specializationByteSize>
        std::vector<char> kernelprogram<specializationByteSize>::ReadFile(const std::string& filename) const
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
        }*/

    } //namespace detail
} //namespace vuda