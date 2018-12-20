#pragma once

namespace vuda
{
    namespace detail
    {
    
        class logical_device
        {
        public:

            logical_device(const vk::DeviceCreateInfo& deviceInfo, const vk::PhysicalDevice& physDevice);       
            //~logical_device();

        #ifdef VUDA_DEBUG_ENABLED
            // external access to the device handle
            vk::Device GetDeviceHandle(void);
        #endif

            //
            // events
            void CreateEvent(event_t* event);
            void DestroyEvent(const event_t event);
            void RecordEvent(const std::thread::id tid, const event_t& event, const stream_t stream);        
            float GetElapsedTimeBetweenEvents(const event_t& start, const event_t& end);

            //
            // queries
            void GetQueryID(const std::thread::id tid, uint32_t* event) const;
            void WriteTimeStamp(const std::thread::id tid, const uint32_t event, const stream_t stream) const;
            float GetQueryPoolResults(const std::thread::id tid, const uint32_t startQuery, const uint32_t stopQuery) const;

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
                                        const dim3 blocks,
                                        const uint32_t stream);

            /*template <typename... specialTypes>
            uint64_t CreateKernel(char const* filename, char const* entry, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, specialization<specialTypes...>& specials, int blocks);*/
        
            //
            // command pool functions        
            void CreateCommandPool(const std::thread::id tid);
        
            void memcpyHtH(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const;
            void memcpyToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);
            void memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream) const;
            void memcpyToHost(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);

            //void UpdateDescriptorAndCommandBuffer(const std::thread::id tid, const uint64_t kernelIndex, const std::vector<void*>& memaddr, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t stream);
            void FlushQueue(const std::thread::id tid, const uint32_t stream);
            void FlushEvent(const std::thread::id tid, const event_t event);        

            //
            //
            void WaitOn(void) const;

        private:

            //std::vector<uint32_t> GetStreamIdentifiers(const void* src);
                
            void push_mem_node(default_storage_node* node);

        private:

            //
            // device handles
            //vk::PhysicalDevice m_physDevice;
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
            // ...

            //
            // thread command pools
            std::unique_ptr<std::shared_mutex> m_mtxCmdPools;
            std::unordered_map<std::thread::id, thrdcmdpool> m_thrdCommandPools;
            std::vector<std::unordered_map<internal_node*, std::vector<std::thread::id>>> m_internal_pinned_buffers_in_use;

            //
            // queue family and queues
            std::vector<std::unique_ptr<std::mutex>> m_mtxQueues;        
            //std::unordered_map<uint32_t, std::vector<vk::Queue>> m_queues;
            std::vector<vk::Queue> m_queues;

            //
            // compute kernels
            std::unique_ptr<std::atomic<bool>> m_kernel_creation_lock;
            std::unique_ptr<std::shared_mutex> m_mtxKernels;
            //std::vector<std::shared_ptr<kernel_interface>> m_kernels;
            std::vector<std::shared_ptr<kernel_interface>> m_kernels;

            //
            // resources
            std::unique_ptr<std::shared_mutex> m_mtxResources;

            //
            // buffers            
            default_storage_node* m_storageBST_root;
            std::vector<default_storage_node*> m_storageBST;
            bst<default_storage_node, void*> m_storage;

            //
            // memory
            memory_allocator m_allocator;

            //
            // internal buffers {cached, pinned}
            internal_buffers<host_cached_node_internal> m_cachedBuffers;        
            internal_buffers<host_pinned_node_internal> m_pinnedBuffers;

            //
            // events
            std::unique_ptr<std::shared_mutex> m_mtxEvents;        
            std::map<vk::Event, event_tick> m_events;
        };

    } //namespace detail
} //namespace vuda

//
// inline function definitions
#include "logicaldevice.inl"