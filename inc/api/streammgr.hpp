#pragma once

namespace vuda
{

    /*__host__*/
    inline vudaError_t streamSynchronize(uint32_t stream)
    {
        //            
        // execute and sync with queue/stream

        std::thread::id tid = std::this_thread::get_id();
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);

        tinfo.GetLogicalDevice()->FlushQueue(tid, stream);        
        
        return vudaSuccess;
    }

} //namespace vuda