#pragma once

namespace vuda
{

    //
    // Creates an event object.
    /*__host__*/
    inline error_t eventCreate(event_t* event)
    {
        const std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);
        tinfo->GetLogicalDevice()->CreateEvent(event);
        return vudaSuccess;
    }

    //
    // Destroys an event object.
    /*__host__ __device__*/
    inline error_t eventDestroy(event_t event)
    {
        const std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);
        tinfo->GetLogicalDevice()->DestroyEvent(event);
        return vudaSuccess;
    }

    //
    // Computes the elapsed time between events.
    /*__host__*/
    inline error_t eventElapsedTime(float* ms, event_t start, event_t end)
    {
        const std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);
        *ms = tinfo->GetLogicalDevice()->GetElapsedTimeBetweenEvents(start, end);
        return vudaSuccess;
    }        

    //
    // Queries an event's status.
    /*__host__*/
    /*inline error_t eventQuery(event_t event)
    {
        const thread_info tinfo = interface_thread_info::GetThreadInfo(std::this_thread::get_id());
        vk::Result res = vk::Result::eEventSet;// tinfo.GetLogicalDevice()->GetEventStatus(event);

        if(res == vk::Result::eEventSet)
            return vudaErrorNotReady;
        else// if(res == vk::Result::eEventReset)
            return vudaSuccess;
    }*/

    //
    // Records an event.
    /*__host__ __device__*/
    inline error_t eventRecord(event_t event, stream_t stream = 0)
    {
        const std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);
        tinfo->GetLogicalDevice()->RecordEvent(tid, event, stream);
        return vudaSuccess;
    }

    //
    // Waits for an event to complete.
    //__host__
    inline error_t eventSynchronize(event_t event)
    {
        const std::thread::id tid = std::this_thread::get_id();
        const detail::thread_info* tinfo = detail::interface_thread_info::GetThreadInfo(tid);
        tinfo->GetLogicalDevice()->FlushEvent(tid, event);
        return vudaSuccess;
    }

} //namespace vuda