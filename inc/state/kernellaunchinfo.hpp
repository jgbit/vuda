#pragma once

namespace vuda
{
    namespace detail
    {

        template <typename... Ts>
        class kernel_launch_info
        {
        private:

            //
            // recursive helpers

            /*template <typename head, typename... tail>
            static constexpr size_t memptrs()
            {
                // c++17
                if constexpr(sizeof...(tail) == 0)
                    return std::is_pointer<head>::value;
                else
                    return std::is_pointer<head>::value + memptrs<tail... >();
            }*/

            // recursive c++11
            template <typename h1>
            static constexpr size_t memptrs()
            {
                return std::is_pointer<h1>::value;
            }
            template <typename h1, typename h2, typename... tail>
            static constexpr size_t memptrs()
            {
                return std::is_pointer<h1>::value + memptrs<h2, tail... >();
            }

            //
            // type list
            template <typename...> struct type_list {};

            //
            // type list wrap of specialization class
            template <typename> struct tl_wrap_special;

            template <typename... Ts1>
            struct tl_wrap_special<type_list<Ts1...>>
            {
                specialization<Ts1...> sp;
            };

            //
            // type list append implementation
            template <typename, typename> struct list_append_impl;

            template <typename... Ts1, typename... Us1>
            struct list_append_impl<type_list<Ts1...>, type_list<Us1...>>
            {
                using type = type_list<Ts1..., Us1...>;
            };

            //
            // filter implementation
            template <template <typename> class, typename...> struct filter_impl;

            template <template <typename> class Predicate>
            struct filter_impl<Predicate>
            {
                using type = type_list<>;
            };

            template <template <typename> class Predicate, typename T, typename... Rest>
            struct filter_impl<Predicate, T, Rest...>
            {
                using type = typename list_append_impl<
                    typename std::conditional<Predicate<T>::value, type_list<T>, type_list<>>::type,
                    typename filter_impl<Predicate, Rest...>::type
                >::type;
            };

            //
            // convenience
            template <template <typename> class Predicate, typename... Ts1>
            using filter_t = typename filter_impl<Predicate, Ts1...>::type;                


            //
            // run-time fill structures
            template <size_t begin, size_t rsrc_index, size_t sp_index, typename head, typename... tail>
            void fill_memptr(logical_device* ldptr, head h, tail... t)
            {
                if constexpr(std::is_pointer<head>::value)
                {
                    //
                    // all pointers are assumed to be resource pointers
                    //m_memptrs[rsrc_index] = h;
    #ifdef VUDA_DEBUG_ENABLED                
                    // [ check whether this is a resource pointer in range ]
                    // ...
    #endif

                    //
                    // fill sampler binding using information about the resource pointers
                    // [ for now they are always assumed to be storagebuffers and we dont need to use the memptrs ]
                    m_bindings[rsrc_index] = vk::DescriptorSetLayoutBinding(rsrc_index, vk::DescriptorType::eStorageBuffer, 1, vk::ShaderStageFlagBits::eCompute, nullptr);

                    //
                    // retrieve the buffer associated with the resource pointer
                    m_bufferdescs[rsrc_index] = ldptr->GetBufferDescriptor(h);

                    //
                    if constexpr(sizeof...(tail) > 0)
                        fill_memptr<begin + 1, rsrc_index + 1, sp_index, tail...>(ldptr, t...);
                }
                else if constexpr(std::is_arithmetic<head>::value)
                {
                    //
                    // fill special
                    m_tl_special.sp.template set<sp_index>(h);

                    if constexpr(sizeof...(tail) > 0)
                        fill_memptr<begin + 1, rsrc_index, sp_index + 1, tail...>(ldptr, t...);
                }
                else
                {
                    assert(0);
                }
            }

        public:

            using specialTypes = filter_t<std::is_arithmetic, Ts...>;
            static constexpr size_t memptrCount = memptrs<Ts...>();
                
            //
            //

            kernel_launch_info(logical_device* ldptr, Ts... args)
            {
                fill_memptr<0, 0, 0>(ldptr, args...);
            }

            //
            // get

            const std::array<vk::DescriptorSetLayoutBinding, memptrCount>& getBindings(void) const
            {
                return m_bindings;
            }

            const std::array<vk::DescriptorBufferInfo, memptrCount>& getBufferDesc(void) const
            {
                return m_bufferdescs;
            }

            const auto& getSpecials(void) const
            {
                return m_tl_special.sp;
            }

        private:

            //
            // pointers to resources, bindings and buffer descriptors
            //std::array<void*, memptrCount> m_memptrs;
            std::array<vk::DescriptorSetLayoutBinding, memptrCount> m_bindings;
            std::array<vk::DescriptorBufferInfo, memptrCount> m_bufferdescs;

            //
            // specialization constants        
            tl_wrap_special<specialTypes> m_tl_special;
        };

    } //namespace detail
} //namespace vuda