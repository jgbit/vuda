#pragma once

namespace vuda
{

    //
    // template kernel launch function
    //
    /*
        the user applied parameters must match with the spv kernel.

        filename and entry are used as identifiers for the kernel / shader module (from file for now)
        #gridDim: it is required to specify the number of blocks to dispatch work
        #stream: the stream parameter could be optional in the future, but we keep it explicit for now
        #threads: note that the number of threads in the kernel can be passed as an (optional) scalar or multiple scalars
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

    /*
        compute kernels are provided in the spir-v format.

        their binary can be included in two ways:
        1. either they are loaded from spv files.
        2. or they are embedded at compile-time.

        the binary is used to create a vulkan shader module
        1. either on first call to launchKernel (gives a warmup overhead)
        2. or it is pre-loaded by a call to loadKernel (this is an extention to the cuda api)
            vuda has a pre-load kernel functionality
            there will be no performance-wise gain in using this functionality
            other than the load can be diverted elsewhere (e.g. start of program)
            the pre-load can be either from a file or an embedded binary.
    */

    /*
    identifier
        (1) either specifies the filename of the spirv file containing the compute kernel binary, or
        (2) it provides the binary directly.
    */
    template <typename... Ts>
    inline void launchKernel(std::string const& identifier, char const* entry, int stream, int gridDim, Ts... args)
    {
        launchKernel<Ts...>(identifier, entry, stream, dim3(gridDim), args...);
    }

    /*
    identifier
        (1) either specifies the filename of the spirv file containing the compute kernel binary, or
        (2) it provides the binary directly.
    */
    template <typename... Ts>
    inline void launchKernel(std::string const& identifier, char const* entry, stream_t stream, dim3 gridDim, Ts... args)
    {
        //
        // get thread id
        const std::thread::id tid = std::this_thread::get_id();

        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);

        //
        // book-keeping
        const detail::kernel_launch_info<Ts...> kl_info(tinfo->GetLogicalDevice(), args...);

        //
        // add to the compute queue
        const stream_t stream_id = stream; // queue index on the queueFamily
        
        //
        // submit kernel
        tinfo->GetLogicalDevice()->SubmitKernel(tid, identifier, entry, kl_info.getBindings(), kl_info.getSpecials(), kl_info.getBufferDesc(), gridDim, stream_id);
    }

    /*__host__
    inline error_t launchKernel(const void* func, dim3 gridDim, dim3 blockDim, void** args, size_t sharedMem, stream_t stream)
    {
        // https://stackoverflow.com/questions/48552390/whats-the-difference-between-launching-with-an-api-call-vs-the-triple-chevron-s
        return vudaSuccess;
    }*/

    /*
    inline void loadKernel(std::string const& identifier, char const* entry)
    {
        //
        // get device assigned to thread
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        //
        // lookup or create a shader module
        tinfo->GetLogicalDevice()->CreateShaderModule(identifier, entry);
    }*/

} //namespace vuda