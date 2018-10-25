#pragma once

namespace vuda
{

    class event_tick
    {
    public:
        using time_point = std::chrono::high_resolution_clock::time_point;

        //
        //

        void tick(void)
        {
            m_tick = std::chrono::high_resolution_clock::now();
        }

        //
        // set

        void setStream(const stream_t& stream)
        {
            m_stream = stream;
        }

        //
        // get

        float toc_diff(const time_point& start) const
        {
            return std::chrono::duration_cast<std::chrono::duration<float>>(m_tick - start).count();
        }

        time_point getTick(void) const
        {
            return m_tick;
        }

        stream_t getStream(void) const
        {
            return m_stream;
        }

    private:
        time_point m_tick;
        stream_t m_stream;
    };

} //namespace vuda