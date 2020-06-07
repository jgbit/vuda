#pragma once

namespace vuda
{

    // Create an asynchronous stream.
    /*__host__*/
    inline error_t streamCreate(stream_t* pStream)
    {
        return vudaSuccess;
    }

    // Destroys and cleans up an asynchronous stream.
    /*__host__ __device__*/
    inline error_t streamDestroy(stream_t stream)
    {
        return vudaSuccess;
    }

    /*__host__*/
    inline error_t streamSynchronize(stream_t stream)
    {
        //
        // execute and sync with queue/stream

        std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);

        tinfo->GetLogicalDevice()->FlushQueue(tid, stream);

        return vudaSuccess;
    }

} //namespace vuda