#pragma once

namespace vuda
{

    // Create an asynchronous stream.
    /*__host__*/
    inline error_t streamCreate(stream_t* pStream)
    {

    }

    // Destroys and cleans up an asynchronous stream.
    /*__host__ __device__*/
    inline error_t streamDestroy(stream_t stream)
    {

    }        

    /*__host__*/
    inline error_t streamSynchronize(stream_t stream)
    {
        //            
        // execute and sync with queue/stream

        std::thread::id tid = std::this_thread::get_id();
        const thread_info tinfo = interface_thread_info::GetThreadInfo(tid);

        tinfo.GetLogicalDevice()->FlushQueue(tid, stream);        
        
        return vudaSuccess;
    }

} //namespace vuda