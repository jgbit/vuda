
#include "setup/setup.hpp"

#pragma once

#include <vulkan/vulkan.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <mutex>
#include <shared_mutex>
#include <atomic>
#include <unordered_map>
#include <unordered_set>
#include <deque>
#include <array>
#include <tuple>
#include <thread>

#include "setup/debug.hpp"
#include "setup/types.hpp"

#include "state/specialization.hpp"
#include "state/instance.hpp"
#include "state/vulkanfunc.hpp"
#include "state/kernelprogram.hpp"
#include "state/binarysearchtree.hpp"
#include "state/storagebuffer.hpp"
#include "state/thrdcmdpool.hpp"
#include "state/logicaldevice.hpp"
#include "state/interfacelogicaldevices.hpp"
#include "state/threadinfo.hpp"
#include "state/state.hpp"

#include "api/devicemgr.hpp"
#include "api/memmgr.hpp"
#include "api/streammgr.hpp"
#include "api/eventmgr.hpp"
#include "api/vudakernel.hpp"
