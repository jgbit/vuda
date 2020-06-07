
#include "setup/setup.hpp"

#pragma once

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <thread>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <vector>
#include <stack>
#include <queue>
#include <map>
#include <unordered_map>
#include <deque>
#include <array>
#include <tuple>
#include <chrono>

#include "setup/types.hpp"
#include "setup/debug.hpp"

#include "state/pool.hpp"
#include "state/event.hpp"
#include "state/specialization.hpp"
#include "state/vulkanfunc.hpp"
#include "state/kernelprogram.hpp"
#include "state/memoryallocator.hpp"
#include "state/binarysearchtree.hpp"
#include "state/virtualalloc.hpp"
#include "state/node_internal.hpp"
#include "state/node_storage.hpp"
#include "state/node_device.hpp"
#include "state/node_host.hpp"
#include "state/node_cached.hpp"
#include "state/thrdcmdpool.hpp"
#include "state/logicaldevice.hpp"
#include "state/kernellaunchinfo.hpp"

// state (singletons)
#include "state/singleton.hpp"
#include "state/instance.hpp"
#include "state/interfacelogicaldevices.hpp"
#include "state/threadinfo.hpp"

/*
client api
instantiation of singletons:
    setDevice is required in any thread before any API functionality can be called.
    most API functions start by calling GetThreadInfo(); this throws an exception if a device has not been assigned (provided VUDA_DEBUG_ENABLED is defined).
    exceptions: {getDeviceCount, getDeviceProperties} accesss the vuda instance directly.
*/
#include "api/devicemgr.hpp"
#include "api/memmgr.hpp"
#include "api/streammgr.hpp"
#include "api/eventmgr.hpp"
#include "api/vudakernel.hpp"