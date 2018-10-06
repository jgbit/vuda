#pragma once

namespace vuda
{

    //
    // template kernel launch function
    //
    
    /*
        the user applied parameters must match with the spv kernel.

        NOTE: For now the function parameters are explicitly given        
        until the vuda pipeline is better developed taking into account
        the shader module layout
        - descriptor set layout
        - specialization constants
        - push constants

        It would be ideal if the parameters could be adapted to variadic arguments
    */    
    //template <typename... Args>
    inline void kernelLaunch(char const* filename, char const* entry, int blocks, int threads, int stream, // required parameters
                               int* param0, int* param1, int* param2, int N)
    {
        const std::thread::id tid = std::this_thread::get_id();
        //
        // get device assigned to thread
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);

        //
        // samplers
        const int numResources = 3;
        std::vector<vk::DescriptorSetLayoutBinding> bindings(numResources);        
        uint32_t iter = 0;
        for(auto& bind : bindings)
        {
            bind.binding = iter;
            bind.descriptorType = vk::DescriptorType::eStorageBuffer;
            bind.descriptorCount = 1;
            bind.stageFlags = vk::ShaderStageFlagBits::eCompute;
            bind.pImmutableSamplers = nullptr;
            ++iter;
        }

        //
        // specialized constants
        const uint32_t numSpecializedConstants = 1;
        
        std::vector<specialization> specials(numSpecializedConstants);
        size_t totalSize = 0;
        iter = 0;
        for(auto& entry : specials)
        {
            entry.m_mapEntry.constantID = iter;
            entry.m_mapEntry.offset = 0;
            entry.m_mapEntry.size = sizeof(uint32_t);
            entry.m_data = N;
            ++iter;
        }

        //
        // push constants (used for dynamic parameters)
        // ...

        //
        // create or retrieve shader program from logical device        
        const uint64_t kernel = tinfo.GetLogicalDevice()->CreateKernel(filename, entry, bindings, specials, blocks, threads);
        
        //
        // retrieve and list the buffers
        // [ create struct containing memadrr and buffer description ]        
        std::vector<void*> memadrr = { param0, param1, param2 };
        std::vector<vk::DescriptorBufferInfo> bufferdescs =
        {
            tinfo.GetLogicalDevice()->GetBufferDescriptor(param0),
            tinfo.GetLogicalDevice()->GetBufferDescriptor(param1),
            tinfo.GetLogicalDevice()->GetBufferDescriptor(param2)
        };
        
        //
        // add to the compute queue
        const uint32_t stream_id = stream; // queueNum on the queueFamily
        tinfo.GetLogicalDevice()->UpdateDescriptorAndCommandBuffer(tid, kernel, memadrr, bufferdescs, stream_id);        
    }

} //namespace vuda