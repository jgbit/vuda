#pragma once

namespace vuda
{
    namespace debug
    {
        #ifdef VUDA_STD_LAYER_ENABLED
    
        //
        // functions

        /*
        std::vector<VkExtensionProperties> availableExtensions(GetAllExtensions());
        std::vector<VkLayerProperties> availableLayers(GetAllLayers());
        */

        inline std::vector<VkExtensionProperties> GetAllExtensions(void)
        {
            //
            // all available vulkan extensions

            uint32_t extensionCount = 0;
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, nullptr);
            std::vector<VkExtensionProperties> availableExtensions(extensionCount);
            vkEnumerateInstanceExtensionProperties(nullptr, &extensionCount, availableExtensions.data());

            return availableExtensions;
        }

        inline std::vector<VkLayerProperties> GetAllLayers(void)
        {
            //
            // all available vulkan layers

            uint32_t layerCount;
            vkEnumerateInstanceLayerProperties(&layerCount, nullptr);
            std::vector<VkLayerProperties> availableLayers(layerCount);
            vkEnumerateInstanceLayerProperties(&layerCount, availableLayers.data());

            return availableLayers;
        }

        inline void PrintAllExtentions(const std::vector<VkExtensionProperties>& availableExtensions)
        {
            std::ostringstream ostr;
            ostr << "Available extensions:" << std::endl;
            for(const auto& extension : availableExtensions)
                ostr << "\t" << extension.extensionName << std::endl;
            std::cout << ostr.str();
        }

        inline void PrintAllLayers(const std::vector<VkLayerProperties>& availableLayers)
        {
            std::ostringstream ostr;
            ostr << "Available validation layers:" << std::endl;
            for(const auto& layerProperties : availableLayers)
                ostr << "\t" << layerProperties.layerName << std::endl;
            std::cout << ostr.str();
        }

        /*inline VkResult CreateDebugUtilsMessengerEXT(VkInstance instance, const VkDebugUtilsMessengerCreateInfoEXT* pCreateInfo, const VkAllocationCallbacks* pAllocator, VkDebugUtilsMessengerEXT* pCallback)
        {
            auto func = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT");

            if(func != nullptr)
                return func(instance, pCreateInfo, pAllocator, pCallback);
            else
                return VK_ERROR_EXTENSION_NOT_PRESENT;
        }

        inline void DestroyDebugUtilsMessengerEXT(VkInstance instance, VkDebugUtilsMessengerEXT callback, const VkAllocationCallbacks* pAllocator)
        {
            auto func = (PFN_vkDestroyDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkDestroyDebugUtilsMessengerEXT");

            if(func != nullptr)
                func(instance, callback, pAllocator);
        }*/

        inline VKAPI_ATTR VkBool32 VKAPI_CALL DebugCallbackFunction(VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity, VkDebugUtilsMessageTypeFlagsEXT messageType, const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData, void* pUserData)
        {
            if(messageSeverity >= VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
            {
                // Message is important enough to show
            }

            std::string severity("");

            if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_VERBOSE_BIT_EXT)
                severity += "VERBOSE:";
            if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_INFO_BIT_EXT)
                severity += "INFO:";
            if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT)
                severity += "WARNING:";
            if(messageSeverity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT)
                severity += "ERROR:";


            std::string prefix("");

            if(messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT)
                prefix += "GENERAL:";
            if(messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT)
                prefix += "VALIDATION:";
            if(messageType & VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT)
                prefix += "PERFORMANCE:";

            std::ostringstream ostr;
            ostr << "Validation layer: " << severity << " " << prefix << " " << pCallbackData->pMessage << std::endl << std::endl;
            std::cerr << ostr.str();
            return VK_FALSE;
        }

        #endif //VUDA_STD_LAYER_ENABLED

        #ifdef VUDA_DEBUG_KERNEL
        
        // singleton
        class syncthreads final : public detail::singleton
        {
        private:

            static std::atomic<uint32_t>& get(void)
            {
                static std::atomic<uint32_t> syncthreads_atomic_counter = 0;
                return syncthreads_atomic_counter;
            }

        public:

            static uint32_t& num_threads(void)
            {
                static uint32_t num_threads_max = 32;
                return num_threads_max;
            }

            static void set_max(uint32_t max)
            {
                num_threads() = max;
            }

            //
            // counter

            static void reset(void)
            {
                get().store(0);
            }

            static void incr(void)
            {
                get()++;
            }

            static uint32_t read(void)
            {
                return get().load();
            }

        //public:
            //static std::mutex m_mtx;
            //static std::condition_variable m_cond;
        };

        template<typename T>
        class Property
        {
        public:
            Property(T& value) : m_value(value)
            {
            }

            void set_value(const T& val)
            {
                std::lock_guard<std::mutex> lck(m_mtx);
                m_value = val;
            }

            /*Property<T>& operator=(const T& val)
            {
                std::lock_guard<std::mutex> lck(m_mtx);
                m_value = val;
                return *this;
            };*/

            operator const T() const
            {
                std::lock_guard<std::mutex> lck(m_mtx);

                std::thread::id tid = std::this_thread::get_id();

                // check if thread has been assigned an unique identifier
                //m_thrd_id.try_emplace
                auto iter = m_thrd_id.find(tid);
                if(iter != m_thrd_id.end())
                {
                    return iter->second;
                }

                // assign index to calling thread
                T value = m_value; // .load();
                m_thrd_id[tid] = value;

                // increment thread identifier
                m_value++;

                return value;
            };

        private:
            mutable std::mutex m_mtx;
            mutable std::map<std::thread::id, T> m_thrd_id;
            T& m_value;
        };

        class threadIdx_def
        {
        public:

            threadIdx_def(uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) : x(x), y(y), z(z)
            {
            }

            void reset(void)
            {
                x.set_value(0);
                y.set_value(0);
                z.set_value(0);
            }

            Property<uint32_t> x;
            Property<uint32_t> y;
            Property<uint32_t> z;
        };

        #endif //VUDA_DEBUG_KERNEL

    } //namespace debug
} //namespace vuda

#ifdef VUDA_DEBUG_KERNEL

//
// cuda language extensions for c++ compilation
//

// typedefs
#define __device__
#define __global__
#define __host__

// memory
//#define __shared__ static
#define __constant__ const

// functions
inline void __syncthreads()
{
    vuda::debug::syncthreads::incr();

    //
    // notify lock
    while(vuda::debug::syncthreads::read() != vuda::debug::syncthreads::num_threads())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    /*
    {        
        // wait on notify
    }
    {
        //
        // reset sync
        vuda::debug::syncthreads::reset();

        //
        // notify threads

    }*/   
}

extern vuda::dim3 gridDim;
extern vuda::dim3 blockDim;
extern vuda::dim3 blockIdx;
extern vuda::debug::threadIdx_def threadIdx;
extern int warpSize;

#endif