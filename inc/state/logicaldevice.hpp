#pragma once

namespace vuda
{
    
    class logical_device
    {
    public:

        logical_device(const vk::DeviceCreateInfo& deviceInfo, const vk::PhysicalDevice& physDevice);       
        //~logical_device();

        //
        // events
        void CreateEvent(event_t* event);        
        void RecordEvent(const std::thread::id tid, const event_t& event, const stream_t stream);        
        float GetElapsedTimeBetweenEvents(const event_t& start, const event_t& end);

        //
        // queries
        //void GetQueryID(const std::thread::id tid, event_t* event) const;        
        //void WriteTimeStamp(const std::thread::id tid, const event_t event, const stream_t stream) const;
        //float GetQueryPoolResults(const std::thread::id tid, const event_t event) const;

        //
        // memory
        void malloc(void** devPtr, size_t size);
        void mallocHost(void** ptr, size_t size);
        void hostAlloc(void** ptr, size_t size);
        void free(void* devPtr);

        //
        // 
        vk::DescriptorBufferInfo GetBufferDescriptor(void* devPtr) const;

        //
        // kernel creation
        template <typename... specialTypes>
        inline void SubmitKernel(   const std::thread::id tid, char const* filename, char const* entry,
                                    const std::vector<vk::DescriptorSetLayoutBinding>& bindings,
                                    specialization<specialTypes...>& specials,
                                    const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors,
                                    const uint32_t blocks,
                                    const uint32_t stream);

        /*template <typename... specialTypes>
        uint64_t CreateKernel(char const* filename, char const* entry, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, specialization<specialTypes...>& specials, int blocks);*/
        
        //
        // command pool functions        
        void CreateCommandPool(const std::thread::id tid);
        
        void memcpyHtH(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const;
        void memcpyToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const;
        void memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);
        void memcpyToHost(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);

        //void UpdateDescriptorAndCommandBuffer(const std::thread::id tid, const uint64_t kernelIndex, const std::vector<void*>& memaddr, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t stream);
        void FlushQueue(const std::thread::id tid, const uint32_t stream) const;
        void FlushEvent(const std::thread::id tid, const event_t event);

        //
        //
        void WaitOn(void) const;

    private:

        //std::vector<uint32_t> GetStreamIdentifiers(const void* src);

        host_cached_node_internal* get_cached_buffer(const vk::DeviceSize size);
        void push_mem_node(default_storage_node* node);

    private:

        //
        // device handles
        vk::PhysicalDevice m_physDevice;
        vk::UniqueDevice m_device;

        //
        // physical device properties
        float m_timestampPeriod;

        //
        // compute family indices
        uint32_t m_queueFamilyIndex;
        uint32_t m_queueComputeCount;

        //
        // transfer family indices

        //
        // thread command pools
        std::unique_ptr<std::shared_mutex> m_mtxCmdPools;
        std::unordered_map<std::thread::id, thrdcmdpool> m_thrdCommandPools;

        //
        // queue family and queues
        std::vector<std::unique_ptr<std::mutex>> m_mtxQueues;        
        std::unordered_map<uint32_t, std::vector<vk::Queue>> m_queues;

        //
        // compute kernels        
        std::unique_ptr<std::shared_mutex> m_mtxKernels;
        std::vector<std::shared_ptr<kernel_interface>> m_kernels;

        //
        // resources
        std::unique_ptr<std::shared_mutex> m_mtxResources;

        //
        // buffers
        /*storage_buffer_node* m_storageBST_root;
        std::vector<storage_buffer_node*> m_storageBST;
        bst<storage_buffer_node, void*> m_storage;*/
        default_storage_node* m_storageBST_root;
        std::vector<default_storage_node*> m_storageBST;
        bst<default_storage_node, void*> m_storage;

        //std::unordered_map<void*, vk::DeviceMemory> m_storage_memory;
        //std::unordered_map<void*, vk::Buffer> m_buffers;

        //
        // memory
        std::unique_ptr<std::mutex> m_mtxAllocatorDevice;
        std::unique_ptr<std::mutex> m_mtxAllocatorHost;
        std::unique_ptr<std::mutex> m_mtxAllocatorCached;
        std::unordered_map<VkFlags, uint32_t> m_memoryAllocatorTypes;
        memory_allocator m_allocatorDevice;
        memory_allocator m_allocatorHost;
        memory_allocator m_allocatorCached;

        // cached buffers
        std::unique_ptr<std::mutex> m_mtxCached_internal;
        std::vector<std::unique_ptr<host_cached_node_internal>> m_cachedBuffers;

        //
        // events
        std::unique_ptr<std::shared_mutex> m_mtxEvents;        
        std::map<vk::Event, event_tick> m_events;

        //
        // kernel to memory access
        //std::unique_ptr<std::shared_mutex> m_mtxResourceKernelAccess;
    };

} //namespace vuda

//
// inline function definitions
#include "logicaldevice.inl"