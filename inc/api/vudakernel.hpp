#pragma once

namespace vuda
{

    //
    // template kernel launch function
    //    
    /*
        the user applied parameters must match with the spv kernel.

        filename and entry are used as identifiers for the kernel / shader module (from file for now)
        #blocks: it is required to specify the number of blocks to dispatch work
        #stream: the stream parameter could be optional in the future, but we keep it explicit for now
        #threads: note that the number of threads in the kernel can be passed as a scalar or multiple scalars
        #Ns: note that dynamic allocation of shared memory will likely follow the same pattern
        (i.e. they will be regarded as a shader specialization)

        NOTE: For now all parameters passed to kernelLaunch follow the scheme
        - all pointers are treated as being pointers returned by calling vuda::malloc
        - all scalars are treated as shader specialization
        - [ push constants could be another option under various conditions ]
        - everything else is neglected

        until the vuda pipeline is better developed taking into account the shader module layout
        - descriptor set layout
        - specialization constants
        - push constants
    */
    template <typename... Ts>
    inline void kernelLaunch(char const* filename, char const* entry, int blocks, int stream, Ts... args)
    {
        //
        // get thread id
        const std::thread::id tid = std::this_thread::get_id();

        //
        // get device assigned to thread
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);

        //
        // book-keeping                
        kernel_launch_info<Ts...> kl_info(tinfo.GetLogicalDevice(), args...);

        //
        // add to the compute queue
        const uint32_t stream_id = stream; // queue index on the queueFamily        
        
        //
        // [ dummy copy for now ]
        std::vector<vk::DescriptorSetLayoutBinding> bindings(kl_info.getBindings().size());
        std::copy_n(kl_info.getBindings().begin(), kl_info.getBindings().size(), bindings.begin());
        
        std::vector<vk::DescriptorBufferInfo> bufferdescs(kl_info.getBufferDesc().size());
        std::copy_n(kl_info.getBufferDesc().begin(), kl_info.getBufferDesc().size(), bufferdescs.begin());

        tinfo.GetLogicalDevice()->SubmitKernel(tid, filename, entry, bindings, kl_info.getSpecials(), bufferdescs, blocks, stream_id);
    }

} //namespace vuda