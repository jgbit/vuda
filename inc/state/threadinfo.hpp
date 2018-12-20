#pragma once

namespace vuda
{
    namespace detail
    {

        class thread_info
        {
        public:
            thread_info(const std::thread::id tid, const int deviceID, logical_device* deviceptr) : m_deviceID(deviceID), m_ldptr(deviceptr)
            {
                deviceptr->CreateCommandPool(tid);
            }

            int GetDeviceID(void) const
            {
                return m_deviceID;
            }

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
            static void insert(const std::thread::id tid, const int deviceID, logical_device* device)
            {                        
                // only one thread/writer can insert/write.
                std::unique_lock<std::shared_mutex> lock(mtx());            
                get().try_emplace(tid, thread_info(tid, deviceID, device));
            }

            static thread_info GetThreadInfo(std::thread::id tid)
            {
                // Multiple threads/readers can read at the same time.
                std::shared_lock<std::shared_mutex> lock(mtx());

            #ifdef VUDA_DEBUG_ENABLED
                auto search = get().find(tid);
                if(search == get().end())
                {
                    std::stringstream ostr; ostr << "The thread " << tid << " has not been assigned a device!";
                    throw std::runtime_error(ostr.str());
                }
            #endif

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

    } //namespace detail
} //namespace vuda