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
            kernelprogram(const vk::UniqueDevice& device, const std::string& filename, const std::string& entryName, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, specialization<specialTypes...>& specials);

            template <typename... specialTypes>
            bool UpdateDescriptorAndCommandBuffer(const vk::UniqueDevice& device, const vk::UniqueCommandBuffer& commandBuffer, const std::vector<vk::DescriptorBufferInfo>& descriptorBufferInfos, specialization<specialTypes...>& specials, const dim3 blocks) const;            

        private:

            template <typename... specialTypes>
            vk::Pipeline GetSpecializedPipeline(const vk::UniqueDevice& device, specialization<specialTypes...>& specials) const;

            template <typename... specialTypes>
            vk::Pipeline CreatePipeline(const vk::UniqueDevice& device, specialization<specialTypes...>& specials) const;

            std::vector<char> ReadFile(const std::string& filename) const;

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

//
// inline function definitions
#include "kernelprogram.inl"