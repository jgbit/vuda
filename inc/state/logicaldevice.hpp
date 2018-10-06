#pragma once

namespace vuda
{

    class logical_device
    {
    public:

        logical_device(const vk::DeviceCreateInfo& deviceInfo, const vk::PhysicalDevice& physDevice);       
        //~logical_device();

        //
        // memory
        void malloc(void** devPtr, size_t size);
        void free(void* devPtr);

        //
        // temp (buffer and memory could have own class)
        //vk::Buffer GetBuffer(void* devPtr);
        vk::DescriptorBufferInfo GetBufferDescriptor(void* devPtr);

        //
        // kernel creation
        uint64_t CreateKernel(char const* filename, char const* entry, const std::vector<vk::DescriptorSetLayoutBinding>& bindings, const std::vector<specialization>& specials, int blocks, int threads);
        
        //
        // command pool functions        
        void CreateCommandPool(const std::thread::id tid);
        
        void memcpyToDevice(const std::thread::id tid, void* dst, const size_t count, const uint32_t stream);
        void memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);
        void memcpyToHost(const std::thread::id tid, const void* src, const size_t count, const uint32_t stream);

        void UpdateDescriptorAndCommandBuffer(const std::thread::id tid, const uint64_t kernelIndex, const std::vector<void*>& memaddr, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t stream);
        void FlushQueue(const std::thread::id tid, const uint32_t stream);

        //
        //
        void WaitOn(void);

    private:

        std::vector<uint32_t> GetStreamIdentifiers(const void* src);

    private:
        vk::PhysicalDevice m_physDevice;
        vk::UniqueDevice m_device;

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

        /*//
        // command pool
        vk::UniqueCommandPool m_commandPool;

        //
        // command buffers (each compute queue gets a default commandbuffer)
        // [ should have their own class ]
        std::vector<std::unique_ptr<std::mutex>> m_mtxCommandBuffers;
        enum vudaCommandBufferState { cbReset = 0, cbRecording = 1, cbSubmitted = 2};
        std::vector<vudaCommandBufferState> m_commandBufferState;
        std::vector<vk::UniqueCommandBuffer> m_commandBuffers;        
        std::vector<vk::UniqueFence> m_ufences;*/

        //
        // compute kernels
        //std::unique_ptr<std::mutex> m_mtxKernels;
        std::unique_ptr<std::shared_mutex> m_mtxKernels;
        std::vector<kernelprogram> m_kernels;

        //
        // buffers
        std::unique_ptr<std::shared_mutex> m_mtxResources;

        storage_buffer_node* m_storageBST_root;
        std::vector<storage_buffer_node*> m_storageBST;
        bst<storage_buffer_node, void*> m_storage;
        //std::unordered_map<void*, vk::DeviceMemory> m_storage_memory;
        //std::unordered_map<void*, vk::Buffer> m_buffers;

        //
        // kernel to memory access
        //std::unique_ptr<std::shared_mutex> m_mtxResourceKernelAccess;        
    };

} //namespace vuda

//
// inline function definitions
#include "logicaldevice.inl"