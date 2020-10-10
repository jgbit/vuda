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
                static vk::ApplicationInfo info("VUDA", 1, "vuda.hpp", 1, VK_API_VERSION_1_2);
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
            static constexpr std::array<const char*, 1> vk_extensions = { VK_EXT_DEBUG_UTILS_EXTENSION_NAME };

            static vk::DispatchLoaderDynamic& getDispatchLoaderDynamic(const vk::UniqueInstance& instance)
            {
                // This dispatch class will fetch all function pointers through the passed instance
                static vk::DispatchLoaderDynamic local_dldy(instance.get(), vkGetInstanceProcAddr);
                return local_dldy;
            }

            static vk::UniqueInstance& get(void)
            {
                //
                // the initialization of the function local static variable is thread-safe.
                static vk::UniqueInstance local_instance = vk::createInstanceUnique(vk::InstanceCreateInfo({}, &getInstanceCreateInfo(), 1, vk_validationLayers.data(), 1, vk_extensions.data()));
                static std::once_flag debug_once;
                std::call_once(debug_once, SetupDebugCallback, local_instance);
                return local_instance;
            }

            static void SetupDebugCallback(const vk::UniqueInstance& instance)
            {
                //
                //
                std::ostringstream ostr;
                ostr << std::this_thread::get_id() << " - validation layer enabled" << std::endl;
                std::cout << ostr.str();

                //
                // default flags
                //vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
                const vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
                const vk::DebugUtilsMessageTypeFlagsEXT messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

                //
                // create info
                const vk::DebugUtilsMessengerCreateInfoEXT debugInfo({}, messageSeverity, messageType, debug::DebugCallbackFunction, nullptr);
                {
                    //static vk::UniqueDebugUtilsMessengerEXT local_debugMsg = instance.get().createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, getDispatchLoaderDynamic(instance));
                    static auto local_debugMsg = instance.get().createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, getDispatchLoaderDynamic(instance));
                }
            }

        public:

            static constexpr std::array<const char*, 1> vk_validationLayers = { "VK_LAYER_KHRONOS_validation" };

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