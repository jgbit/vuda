#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // singleton vulkan instance class
        //
        class Instance final : public singleton
        {
        private:

            static vk::ApplicationInfo& getInstanceCreateInfo(void)
            {
                static vk::ApplicationInfo info("VUDA", 1, "vuda.hpp", 1, VK_API_VERSION_1_1);
                return info;
            }            

            static std::vector<vk::PhysicalDevice>& GetPhysicalDevices(void)
            {
                static std::vector<vk::PhysicalDevice> physicalDevices = Instance::get()->enumeratePhysicalDevices();
                return physicalDevices;
            }
        
    #ifndef VUDA_STD_LAYER_ENABLED

            static vk::UniqueInstance& get(void)
            {
                // the initialization of the function local static variable is thread-safe.
                static vk::UniqueInstance instance = vk::createInstanceUnique(vk::InstanceCreateInfo(vk::InstanceCreateFlags(), &getInstanceCreateInfo()));
                return instance;
            }

    #else
            static std::vector<const char*>& getExtensionList(void)
            {
                static std::vector<const char*> extensions = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };
                return extensions;
            }

            static vk::DispatchLoaderDynamic& getDispatchLoaderDynamic(const vk::UniqueInstance& instance)
            {
                // This dispatch class will fetch all function pointers through the passed instance
                static vk::DispatchLoaderDynamic local_dldy(instance.get(), vkGetInstanceProcAddr);
                return local_dldy;
            }

            /*static vk::DispatchLoaderStatic& getDispatchLoaderStatic(void)
            {
                static vk::DispatchLoaderStatic local;
                return local;
            }*/

            static vk::UniqueInstance& get(void)
            {
                //
                // the initialization of the function local static variable is thread-safe.            
                static vk::UniqueInstance local_instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &getInstanceCreateInfo(), 1, getValidationLayers().data(), 1, getExtensionList().data()));

                //
                // check whether we can skip debug initialization
                static std::atomic<bool> debug_initialized = false;
                static std::atomic_flag debug_lock = ATOMIC_FLAG_INIT;

                if(debug_lock.test_and_set(std::memory_order_acquire) == false)
                {
                    std::ostringstream ostr;
                    ostr << std::this_thread::get_id() << " - validation layer enabled" << std::endl;
                    std::cout << ostr.str();

                    SetupDebugCallback(local_instance);
                    debug_initialized.store(true);
                }

                // a simple spin lock make sure that debug layer has been initialized
                while(debug_initialized.load() == false)
                {
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }                    

                return local_instance;
            }
        
            /*void SetLayerFlags(const VkDebugUtilsMessageSeverityFlagsEXT messageSeverity, const VkDebugUtilsMessageTypeFlagsEXT messageType)
            {
                vk::debug::messageSeverity = messageSeverity;
                vk::debug::messageType = messageType;
            }*/

            static void SetupDebugCallback(const vk::UniqueInstance& instance)
            {
                //
                // default flags
                //vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
                vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
                vk::DebugUtilsMessageTypeFlagsEXT messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

                //
                // create info            
                const vk::DebugUtilsMessengerCreateInfoEXT debugInfo({}, messageSeverity, messageType, debug::DebugCallbackFunction, nullptr);

                {
                    //static vk::UniqueDebugUtilsMessengerEXT local_debugMsg = instance.get().createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, getDispatchLoaderDynamic(instance));
                    static auto local_debugMsg = instance.get().createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, getDispatchLoaderDynamic(instance));
                }
            }

        public:

            static std::vector<const char*>& getValidationLayers(void)
            {
                #if defined(__APPLE__)
                    static std::vector<const char*> validationLayers = { "VK_LAYER_KHRONOS_validation" };
                #else
                    static std::vector<const char*> validationLayers = { "VK_LAYER_LUNARG_standard_validation" };
                #endif
                return validationLayers;
            }
    #endif        

        public:

            static size_t GetPhysicalDeviceCount(void)
            {
                return GetPhysicalDevices().size();
            }

            static vk::PhysicalDevice GetPhysicalDevice(const int device)
            {
                assert(device >= 0 && device < (int)GetPhysicalDevices().size());
                return GetPhysicalDevices()[device];
            }
        
        };

    } //namespace detail
} //namespace vuda