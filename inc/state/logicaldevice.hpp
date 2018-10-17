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
        
        void memcpyToDevice(const std::thread::id tid, void* dst, const size_t count, const uint32_t stream);
        void memcpyDeviceToDevice(const std::thread::id tid, void* dst, const void* src, const size_t count, const uint32_t stream);
        void memcpyToHost(const std::thread::id tid, const void* src, const size_t count, const uint32_t stream);

        //void UpdateDescriptorAndCommandBuffer(const std::thread::id tid, const uint64_t kernelIndex, const std::vector<void*>& memaddr, const std::vector<vk::DescriptorBufferInfo>& bufferDescriptors, const uint32_t stream);
        void FlushQueue(const std::thread::id tid, const uint32_t stream);

        //
        //
        void WaitOn(void);

    private:

        std::vector<uint32_t> GetStreamIdentifiers(const void* src);

    private:

        //
        // device handles
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

        //
        // compute kernels        
        std::unique_ptr<std::shared_mutex> m_mtxKernels;
        std::vector<std::shared_ptr<kernel_interface>> m_kernels;

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