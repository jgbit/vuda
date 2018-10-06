#pragma once

namespace vuda
{
    class thread_info
    {
    public:
        thread_info(const int deviceID, logical_device* deviceptr) : m_deviceID(deviceID), m_ldptr(deviceptr) {}

        /*int GetID(void) const
        {
            return m_deviceID;
        }*/

        logical_device* GetLogicalDevice(void) const
        {
            return m_ldptr;
        }

    private:
        int m_deviceID;
        logical_device* m_ldptr;
    };

    class interface_thread_info final
    {
    public:
        static void insert(std::thread::id tid, int deviceID, logical_device* device)
        {                        
            // only one thread/writer can insert/write.
            std::unique_lock<std::shared_mutex> lock(mtx());
            device->CreateCommandPool(tid);
            get().insert({ tid, thread_info(deviceID, device) });
        }

        static thread_info GetThreadInfo(std::thread::id tid)
        {
            // Multiple threads/readers can read at the same time.
            std::shared_lock<std::shared_mutex> lock(mtx());            
            return get().at(tid);
        }

    private:
        // singleton
        static std::unordered_map<std::thread::id, thread_info>& get(void)
        {
            static std::unordered_map<std::thread::id, thread_info> local_thread_infos;
            return local_thread_infos;
        }

        static std::shared_mutex& mtx(void)
        {
            // https://en.cppreference.com/w/cpp/thread/shared_mutex
            static std::shared_mutex local_mtx;
            return local_mtx;
        }

    private:
        interface_thread_info() = default;
        ~interface_thread_info() = default;

        // delete copy and move constructors and assign operators
        interface_thread_info(interface_thread_info const&) = delete;             // Copy construct
        interface_thread_info(interface_thread_info&&) = delete;                  // Move construct
        interface_thread_info& operator=(interface_thread_info const&) = delete;  // Copy assign
        interface_thread_info& operator=(interface_thread_info &&) = delete;      // Move assign

    };

} //namespace vuda