#pragma once

namespace vuda
{
    namespace detail
    {
        //
        // singleton base class
        //
        class singleton
        {
        protected:

            singleton() = default;
            ~singleton() = default;

        private:

            // delete copy and move constructors and assign operators
            singleton(singleton const&) = delete;             // Copy construct
            singleton(singleton&&) = delete;                  // Move construct
            singleton& operator=(singleton const&) = delete;  // Copy assign
            singleton& operator=(singleton &&) = delete;      // Move assign
        };

    } //namespace detail
} //namespace vuda