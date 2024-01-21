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
				static vk::ApplicationInfo info("VUDA", 1, "vuda.hpp", 1, VK_API_VERSION_1_3);
				return info;
			}

			static std::vector<vk::PhysicalDevice>& GetPhysicalDevices(void)
			{
				static std::vector<vk::PhysicalDevice> physicalDevices = Instance::get()->enumeratePhysicalDevices();
				return physicalDevices;
			}

			static vk::UniqueInstance& get(void)
			{
				// the initialization of the function local static variable is thread-safe.
				static vk::UniqueInstance instance = vk::createInstanceUnique(vk::InstanceCreateInfo(vk_createinfoflags, &getInstanceCreateInfo(), (uint32_t)vk_layernames.size(), vk_layernames.data(), (uint32_t)vk_instanceextensions.size(), vk_instanceextensions.data()));

				#ifdef VUDA_STD_LAYER_ENABLED
				static std::once_flag debug_once;
				std::call_once(debug_once, SetupDebugCallback, instance);
				#endif

				return instance;
			}

#ifdef VUDA_STD_LAYER_ENABLED

			static vk::DispatchLoaderDynamic& getDispatchLoaderDynamic(const vk::UniqueInstance& instance)
			{
				// This dispatch class will fetch all function pointers through the passed instance
				static vk::DispatchLoaderDynamic local_dldy(instance.get(), vkGetInstanceProcAddr);
				return local_dldy;
			}

			static void SetupDebugCallback(const vk::UniqueInstance& instance)
			{
				std::ostringstream ostr;
				ostr << std::this_thread::get_id() << " - validation layer enabled" << std::endl;
				std::cout << ostr.str();

				// default flags
				//vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
				const vk::DebugUtilsMessageSeverityFlagsEXT messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning | vk::DebugUtilsMessageSeverityFlagBitsEXT::eError;
				const vk::DebugUtilsMessageTypeFlagsEXT messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation | vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance;

				// create info
				const vk::DebugUtilsMessengerCreateInfoEXT debugInfo({}, messageSeverity, messageType, debug::DebugCallbackFunction, nullptr);
				{
					static auto local_debugMsg = instance.get().createDebugUtilsMessengerEXTUnique(debugInfo, nullptr, getDispatchLoaderDynamic(instance));
				}
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

		public:

			// instance creation flags
			#if (PLATFORM_NAME == VUDA_APPLE)
			static constexpr vk::InstanceCreateFlags vk_createinfoflags = vk::InstanceCreateFlags(VK_INSTANCE_CREATE_ENUMERATE_PORTABILITY_BIT_KHR);
			#else
			static constexpr vk::InstanceCreateFlags vk_createinfoflags = vk::InstanceCreateFlags();
			#endif

			// extensions
			#if (PLATFORM_NAME == VUDA_APPLE)
			#ifdef VUDA_STD_LAYER_ENABLED
			static constexpr uint32_t vk_extensionsNum = 3;
			#else
			static constexpr uint32_t vk_extensionsNum = 2;
			#endif
			static constexpr std::array<const char*, vk_extensionsNum> vk_instanceextensions = {
				VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME
				, VK_KHR_PORTABILITY_SUBSET_EXTENSION_NAME
				#ifdef VUDA_STD_LAYER_ENABLED
				, VK_EXT_DEBUG_UTILS_EXTENSION_NAME
				#endif
			};
			#else
			#ifdef VUDA_STD_LAYER_ENABLED
			static constexpr uint32_t vk_extensionsNum = 1;
			#else
			static constexpr uint32_t vk_extensionsNum = 0;
			#endif
			static constexpr std::array<const char*, vk_extensionsNum> vk_instanceextensions = {
				#ifdef VUDA_STD_LAYER_ENABLED
				VK_EXT_DEBUG_UTILS_EXTENSION_NAME
				#endif
			};
			#endif

			// layers
			#ifdef VUDA_STD_LAYER_ENABLED
			static constexpr uint32_t vk_layersNum = 1;
			static constexpr std::array<const char*, vk_layersNum> vk_layernames = { "VK_LAYER_KHRONOS_validation" };
			#else
			static constexpr uint32_t vk_layersNum = 0;
			static constexpr std::array<const char*, vk_layersNum> vk_layernames = { };
			#endif

		};

	} //namespace detail
} //namespace vuda