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

    } //namespace debug
} //namespace vuda