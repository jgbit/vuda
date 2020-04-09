#pragma once

namespace vuda
{
    namespace detail
    {

        /*
            default pool node
        */
        template<size_t capacity, typename T>
        class default_pool_allocator
        {
        public:

            default_pool_allocator() :
                m_next(nullptr)
            {
                init();
            }

            virtual ~default_pool_allocator()
            {
            }

            void set_next(default_pool_allocator* next)
            {
                m_next = next;
            }

            default_pool_allocator* get_next()
            {
                return m_next;
            }

            size_t return_element(void) const
            {
                return ++m_returned;
            }

            void init(void) const
            {
                m_returned.store(0);
            }

            //
            // functionality
            virtual T get(uint32_t index) const = 0;
            virtual void reset(void) = 0;

        private:

            // linked list
            default_pool_allocator* m_next;
            mutable std::atomic<size_t> m_returned;
        };

        /*
            descriptor set pool node
        */
        template<uint32_t capacity, typename T>
        class descriptor_pool_allocator : public default_pool_allocator<capacity, T>
        {
        public:

            descriptor_pool_allocator(vk::Device device, const uint32_t binding_size, vk::DescriptorSetLayout descriptorSetLayout) :
                m_device(device),
                //
                // allocate descriptor pool
                // [ for now, kernels can only take eStorageBuffer as input, the pool therefore only have one descriptor type ]
                m_descriptorPoolSize(vk::DescriptorPoolSize(vk::DescriptorType::eStorageBuffer, binding_size * capacity)),
                m_descriptorPool(device.createDescriptorPool(vk::DescriptorPoolCreateInfo(vk::DescriptorPoolCreateFlags(), capacity, 1, &m_descriptorPoolSize))),

                //
                // allocate descriptor sets
                m_descriptorSets(device.allocateDescriptorSets(vk::DescriptorSetAllocateInfo(m_descriptorPool, capacity, std::vector<vk::DescriptorSetLayout>(capacity, descriptorSetLayout).data())))
            {
                /*
                    - VkDescriptorSets are allocated from a "parent" VkDescriptorPool
                    - descriptors allocated in different threads must come from different pools
                    - But VkDescriptorSets from the same pool can be written to by different threads
                */

                //std::cout << "vuda:: descriptorPool capacity: " << capacity << std::endl;
            }

            ~descriptor_pool_allocator()
            {
                m_device.destroyDescriptorPool(m_descriptorPool);
            }

            T get(const uint32_t index) const
            {
                return m_descriptorSets[index];
            }
            
            void reset(void)
            {
                m_device.resetDescriptorPool(m_descriptorPool, vk::DescriptorPoolResetFlags());
            }

        private:
            vk::Device m_device;

            vk::DescriptorPoolSize m_descriptorPoolSize;
            vk::DescriptorPool m_descriptorPool;

            std::vector<vk::DescriptorSet> m_descriptorSets;            
        };

        /*
            command buffer pool node
        */
        template<size_t capacity, typename T>
        class commandbuffer_pool_allocator : public default_pool_allocator<capacity, T>
        {
        public:

            commandbuffer_pool_allocator(vk::Device device, uint32_t queueFamilyIndex) :
                m_device(device),
                // allocate pool
                m_commandPool(device.createCommandPool(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamilyIndex))),                
                // allocate buffers
                m_commandBuffers(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(m_commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)capacity)))
            {                
            }

            ~commandbuffer_pool_allocator()
            {
                m_device.destroyCommandPool(m_commandPool);
            }

            T get(const uint32_t index) const
            {
                return m_commandBuffers[index];
            }

            void reset(void)
            {
                m_device.resetCommandPool(m_commandPool, vk::CommandPoolResetFlags());
            }

        private:
            vk::Device m_device;

            vk::CommandPool m_commandPool;
            std::vector<vk::CommandBuffer> m_commandBuffers;
        };

        /*
            {command buffer, fence, descriptor usage} pool node
        */
        struct command_buffer_ext
        {
        public:
            command_buffer_ext(const vk::Device device, const vk::CommandPool cmdpool) :
                m_fence(device.createFence(vk::FenceCreateFlags()))                
            {                
            }

            void set_command_buffer(const vk::CommandBuffer& cb)
            {
                m_commandBuffer = cb;
            }

            // note there are no destructor as that would require to store device,
            // it is up to thrdcb_pool_allocator to destroy the fence allocation on cleanup
            void destroyfence(const vk::Device device)
            {
                device.destroyFence(m_fence);
            }

            void reset(const vk::Device device)
            {
                device.resetFences(1, &m_fence);

                //
                // return all descriptor sets to their respective pools
                for(size_t i = 0; i < m_desc_pools.size(); ++i)
                    m_desc_pools[i]->return_element();
                m_desc_pools.clear();
            }

            void addpoolid(descriptor_pool_allocator<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet>* id)
            {
                m_desc_pools.push_back(id);
            }

        private:

            vk::Fence m_fence;
            vk::CommandBuffer m_commandBuffer;
            std::vector<descriptor_pool_allocator<VUDA_NUM_KERNEL_DESCRIPTOR_SETS, vk::DescriptorSet>*> m_desc_pools;
        };

        template<size_t capacity, typename T>
        class thrdcb_pool_allocator : public default_pool_allocator<capacity, T>
        {
        public:

            thrdcb_pool_allocator(vk::Device device, uint32_t queueFamilyIndex) :
                m_device(device),
                // allocate pool
                m_commandPool(device.createCommandPool(vk::CommandPoolCreateInfo(vk::CommandPoolCreateFlags(), queueFamilyIndex))),
                // allocate buffers
                m_commandBuffers(device.allocateCommandBuffers(vk::CommandBufferAllocateInfo(m_commandPool, vk::CommandBufferLevel::ePrimary, (uint32_t)capacity))),
                // allocate command buffer ext
                m_cbext(capacity, command_buffer_ext(device, m_commandPool))
            {
                for(size_t i = 0; i < capacity; ++i)
                    m_cbext[i].set_command_buffer(m_commandBuffers[i]);
            }

            ~thrdcb_pool_allocator()
            {
                m_device.destroyCommandPool(m_commandPool);
                for(size_t i = 0; i < capacity; ++i)
                    m_cbext[i].destroyfence(m_device);
            }

            T get(const uint32_t index) const
            {
                return m_cbext[index];
            }

            void reset(void)
            {
                m_device.resetCommandPool(m_commandPool, vk::CommandPoolResetFlags());
                // reset cmd buf ext
                for(size_t i = 0; i < capacity; ++i)
                    m_cbext[i].reset(m_device);
            }

        private:
            vk::Device m_device;
            vk::CommandPool m_commandPool;
            std::vector<vk::CommandBuffer> m_commandBuffers;
            std::vector<command_buffer_ext> m_cbext;
        };

        /*
            (thread-safe) pool containing sets of finite elements.
            - a set is allocated when the current one has used up all its elements
            - when elements are returned they are not recycled until all elements in the set has been returned.
        */
        template<size_t capacity, typename T, class T_alloc, typename... Ts> //=default_pool_allocator
        class pool_finite_sets
        {
        private:

            // unpacking tuple
            template<int... > struct seq {};
            template<int N, int... S> struct gens : gens<N - 1, N - 1, S...> {};
            template<int... S> struct gens<0, S...> { typedef seq<S...> type; };

            template<int... S>
            inline T_alloc *alloc_call(seq<S...>)
            {
                return new T_alloc(std::get<S>(m_args)...);
            }

        public:

            explicit pool_finite_sets(Ts... args) :
                m_args(args...),
                m_first_set(args...),
                m_used(0),
                m_current_set(&m_first_set),
                m_last_set(&m_first_set)                
            {
                /*if(m_maxsets < 1)
                    throw std::invalid_argument("max set size must be at least 1.");*/

                //std::function<T_alloc()> func = [&](Ts... args) { return T_alloc(args...); };
            }
            
            ~pool_finite_sets()
            {
                // delete all sets
                T_alloc* set = static_cast<T_alloc*>(m_first_set.get_next());
                while(set)
                {
                    T_alloc* next_set = static_cast<T_alloc*>(set->get_next());
                    delete set;
                    set = next_set;
                }
            }

            //
            // thread-safe functions
            //

            void get_element(T* element, T_alloc** pool_id)
            {
                unsigned int index;

                while(true)
                {
                    // atomically increase the counter
                    index = m_used++;

                    // exit look when we have a valid (unique) index
                    if(index < capacity)
                        break;
                    
                    // take lock
                    if(m_creation_lock.test_and_set(std::memory_order_acquire) == false)
                    {
                        /*std::ostringstream ostr;
                        ostr << "thrd: " << std::this_thread::get_id() << ", took lock, index is at: " << index << std::endl;
                        std::cout << ostr.str();
                        ostr.str("");*/

                        if(m_deleted_sets.empty() == false)
                        {
                            // reuse previous set
                            std::lock_guard<std::mutex> lck(m_mtx_deletedsets);
                            m_current_set = m_deleted_sets.front();
                            m_deleted_sets.pop();

                            /*std::ostringstream ostr;
                            ostr << "allocating old set, current: " << m_current_set << std::endl;
                            std::cout << ostr.str();*/
                        }
                        else
                        {
                            // allocate new set
                            /*std::ostringstream ostr;
                            ostr << "allocating new set, current: " << m_current_set;*/

                            T_alloc* new_set = alloc_call(typename gens<sizeof...(Ts)>::type());

                            m_last_set->set_next(new_set);
                            m_last_set = new_set;
                            m_current_set = new_set;

                            /*ostr << ", new set: " << new_set << std::endl;
                            std::cout << ostr.str();*/
                        }

                        /*ostr << "thrd: " << std::this_thread::get_id() << ", releasing lock." << std::endl;
                        std::cout << ostr.str();*/

                        // reset counter (threads accessing the outerscope is allowed to request from the current set)
                        m_used = 0;
                        // release lock (let the stuck threads have a go)
                        m_creation_lock.clear(std::memory_order_release);
                    }
                }

                // it is safe to request the data at the unique index
                *element = m_current_set->get(index);
                *pool_id = m_current_set;
            }

            void return_element(T_alloc* pool)
            {
                size_t num = pool->return_element();

                assert(num <= capacity); // too many elements have been returned to the pool

                if(num == capacity)
                {
                    // all elements have been returned, reset pool for new use
                    pool->reset();
                    pool->init();
                    
                    // make the pool available again
                    std::lock_guard<std::mutex> lck(m_mtx_deletedsets);
                    m_deleted_sets.push(pool);

                    /*std::ostringstream ostr;
                    ostr << "thrd: " << std::this_thread::get_id() << ", set fully returned: " << pool << std::endl;
                    std::cout << ostr.str();*/
                }
            }

        private:

            std::tuple<Ts...> m_args;
            T_alloc m_first_set;

            std::atomic_flag m_creation_lock = ATOMIC_FLAG_INIT;
            std::atomic<uint32_t> m_used;
            
            T_alloc* m_current_set;
            T_alloc* m_last_set;

            std::mutex m_mtx_deletedsets;
            std::queue<T_alloc*> m_deleted_sets;
        };

    } //namespace detail
} // namespace vuda