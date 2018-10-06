#pragma once

namespace vuda
{
    // __host__ ​ __device__ ​
    inline vudaError_t malloc(void** devPtr, size_t size)
    {
        std::ostringstream ostr;

        //
        // get device assigned to thread
        const thread_info tinfo = interface_thread_info::GetThreadInfo(std::this_thread::get_id());
        
        //
        // allocate mem on the device
        tinfo.GetLogicalDevice()->malloc(devPtr, size);

        return vudaSuccess;
    }

    //__host__ ​ __device__ 
    inline vudaError_t free(void* devPtr)
    {
        //
        // get device assigned to thread
        const thread_info tinfo = interface_thread_info::GetThreadInfo(std::this_thread::get_id());
        
        //
        // free the allocation on the device
        tinfo.GetLogicalDevice()->free(devPtr);

        return vudaSuccess;
    }

    //__host__ ​
    inline vudaError_t memcpy(void* dst, const void* src, const size_t count, const vudaMemcpyKind kind, const uint32_t stream=0)
    {
        std::thread::id tid = std::this_thread::get_id();
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);

        if(kind == vuda::memcpyHostToDevice)
        {
            //
            // copy the memory to the staging buffer which is allocated with host visible memory
            // (this is the infamous double copy)
            std::memcpy(dst, src, count);

            //
            // submit the copy command to command buffer (this is an async call)
            tinfo.GetLogicalDevice()->memcpyToDevice(tid, dst, count, stream);
        }
        else if(kind == vuda::memcpyDeviceToDevice)
        {
            //
            // submit the copy command to command buffer (this is an async call)
            tinfo.GetLogicalDevice()->memcpyDeviceToDevice(tid, dst, src, count, stream);

        }
        else if(kind == vuda::memcpyDeviceToHost)
        {            
            //
            // submit the copy command to command buffer (this is an async call)
            tinfo.GetLogicalDevice()->memcpyToHost(tid, src, count, stream);
            
            //            
            // execute kernels that have access to modify src
            // the adress src will only be associated with one logical device.
            // the logical device associated with the calling thread must be the same for all calling threads accessing src
            //            
            tinfo.GetLogicalDevice()->FlushQueue(tid, stream); // src

            //
            // copy the memory back to the staging buffer (host visible memory)            
            std::memcpy(dst, src, count);
        }

        return vudaSuccess;
    }

} //namespace vuda