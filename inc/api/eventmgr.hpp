#pragma once

namespace vuda
{

    //__host__
    inline vudaError_t EventSynchronize()//cudaEvent_t event
    {
        const thread_info tinfo = interface_thread_info::GetThreadInfo(std::this_thread::get_id());

        tinfo.GetLogicalDevice()->WaitOn();

        return vudaSuccess;
    }

} //namespace vuda