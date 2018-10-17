#pragma once

namespace vuda
{

    template <typename... Ts>
    class specialization
    {
    private:

        //
        // recursive helpers

        template <typename head, typename... tail>
        static constexpr size_t byte_size()
        {
            static_assert(std::is_arithmetic<head>::value, "vuda::specialization: entries must be of arithmetic type");

            if constexpr(sizeof...(tail) > 0)
                return sizeof(head) + byte_size<tail...>();
            else
                return sizeof(head);
        }

        template <size_t begin, size_t end>
        static constexpr size_t offset()
        {
            if constexpr(begin == end)
                return 0;
            else
                return sizeof(type<begin>) + offset<begin + 1, end>();
        }

        template <size_t begin>
        void fill_entries(void)
        {
            if constexpr(begin == count)
                return;
            else
            {
                m_entries[begin] = vk::SpecializationMapEntry
                {
                    begin,
                    static_cast<uint32_t>(offset0<begin>),
                    sizeof(type<begin>)
                };
                fill_entries<begin+1>();
            }
        }

        //
        // individual helpers

        template <size_t index>
        using type = typename std::tuple_element<index, std::tuple<Ts...>>::type;

        template <size_t index>
        static constexpr size_t offset0 = offset<0, index>();

        void create_info(void)
        {
            m_info = vk::SpecializationInfo
            {
                static_cast<uint32_t>(count),            
                m_entries.data(),
                static_cast<vk::DeviceSize>(bytesize),
                reinterpret_cast<void*>(m_data.data())
            };
        }

    public:

        //
        // the number of constants and the bytesize can be public
        static constexpr size_t count = sizeof...(Ts);
        static constexpr size_t bytesize = byte_size<Ts...>();

        //
        // c++11 rule of five

        //
        // empty constructor

        specialization()
        {
            fill_entries<0>();
            create_info();
        }

        //~specialization()

        //
        // copy constructor and assignment operator

        specialization(const specialization& other) : m_entries{ other.m_entries }, m_data{ other.m_data }
        {
            create_info();
        }

        specialization& operator=(const specialization& other)
        {
            if(this != &other)
                m_data = other.m_data;        
            return *this;
        }

        //
        // move constructor and assignment operator

        specialization(specialization&& other) : m_entries{ std::move(other.m_entries) }, m_data{ std::move(other.m_data) }
        {
            create_info();
        }

        specialization& operator=(specialization&& other)
        {
            if(this != &other)
                m_data = std::move(other.m_data);
            return *this;
        }

        //
        // get specializationization info struct        
        vk::SpecializationInfo& info(void)
        {
            return m_info;
        }

        //
        // get copy of data
        std::array<uint8_t, bytesize> data(void)
        {
            return m_data;
        }

        //
        // get entry
        template <size_t index>
        auto get() const
        {
            type<index> value;
            std::memcpy(&value, m_data.data() + offset0<index>, sizeof(type<index>));
            return value;
        }

        //
        // set entry
        template <size_t index>
        void set(type<index> value)
        {
            std::memcpy(m_data.data() + offset0<index>, &value, sizeof(type<index>));
        }

    private:

        vk::SpecializationInfo m_info;
        std::array<vk::SpecializationMapEntry, count> m_entries;
        std::array<uint8_t, bytesize> m_data;
    };

} //namespace vuda