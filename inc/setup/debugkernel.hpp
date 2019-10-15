#pragma once

namespace vuda
{
    namespace debug
    {
        #ifdef VUDA_DEBUG_KERNEL

        // singleton
        class syncthreads final : public detail::singleton
        {
        private:

            static std::atomic<uint32_t>& get(void)
            {
                static std::atomic<uint32_t> syncthreads_atomic_counter = 0;
                return syncthreads_atomic_counter;
            }

        public:

            static uint32_t& num_threads(void)
            {
                static uint32_t num_threads_max = 32;
                return num_threads_max;
            }

            static void set_max(uint32_t max)
            {
                num_threads() = max;
            }

            //
            // counter

            static void reset(void)
            {
                get().store(0);
            }

            static void incr(void)
            {
                get()++;
            }

            static uint32_t read(void)
            {
                return get().load();
            }

            //public:
                //static std::mutex m_mtx;
                //static std::condition_variable m_cond;
        };

        template<typename T>
        class Property
        {
        public:
            Property(T& value) : m_value(value)
            {
                m_mtx = std::make_unique<std::mutex>();
            }

            void set_value(const T& val)
            {
                std::lock_guard<std::mutex> lck(*m_mtx);
                m_value = val;
            }

            /*Property<T>& operator=(const T& val)
            {
                std::lock_guard<std::mutex> lck(m_mtx);
                m_value = val;
                return *this;
            };*/

            operator const T() const
            {
                std::lock_guard<std::mutex> lck(*m_mtx);

                std::thread::id tid = std::this_thread::get_id();

                // check if thread has been assigned an unique identifier
                //m_thrd_id.try_emplace
                auto iter = m_thrd_id.find(tid);
                if(iter != m_thrd_id.end())
                {
                    return iter->second;
                }

                // assign index to calling thread
                T value = m_value; // .load();
                m_thrd_id[tid] = value;

                // increment thread identifier
                m_value++;

                return value;
            };

        private:
            std::unique_ptr<std::mutex> m_mtx;
            mutable std::map<std::thread::id, T> m_thrd_id;
            T& m_value;
        };

        class threadIdx_def
        {
        public:

            threadIdx_def(uint32_t x = 0, uint32_t y = 0, uint32_t z = 0) : x(x), y(y), z(z)
            {
            }

            void reset(void)
            {
                x.set_value(0);
                y.set_value(0);
                z.set_value(0);
            }

            Property<uint32_t> x;
            Property<uint32_t> y;
            Property<uint32_t> z;
        };

        #endif //VUDA_DEBUG_KERNEL

    } //namespace debug    

} //namespace vuda

#ifdef VUDA_DEBUG_KERNEL

//
// cuda language extensions for c++ compilation
//

// typedefs
#define __device__
#define __global__
#define __host__

// memory
//#define __shared__ static
#define __constant__ const

// functions
inline void __syncthreads()
{
    vuda::debug::syncthreads::incr();

    //
    // notify lock
    while(vuda::debug::syncthreads::read() != vuda::debug::syncthreads::num_threads())
    {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    /*
    {
        // wait on notify
    }
    {
        //
        // reset sync
        vuda::debug::syncthreads::reset();

        //
        // notify threads

    }*/
}

extern vuda::dim3 gridDim;
extern vuda::dim3 blockDim;
extern vuda::dim3 blockIdx;
extern vuda::debug::threadIdx_def threadIdx;
extern int warpSize;

#endif

namespace vuda
{    
    #ifdef VUDA_DEBUG_KERNEL

    template <typename... Ts>
    inline error_t hostKernel(void(*func)(Ts...), dim3 gridDims, dim3 blockDims, Ts... args)//, void** args, size_t sharedMem, stream_t stream=0)
    {
        //
        // create a callable functor with the arguments given
        std::function<void()> functor = std::bind(func, args...);

        //
        // all threads in a block is executed in parallel, but each block is executed sequential
        unsigned int num_threads = blockDims.x * blockDims.y * blockDims.z;
        std::thread* t = new std::thread[num_threads];

        // setup __syncthreads();
        debug::syncthreads::set_max(num_threads);

        //
        // set grid and block dimensions
        blockDim = blockDims;
        gridDim = gridDims;

        // for each block in the grid
        //for(unsigned int block=0; block <num_blocks; ++block)
        for(unsigned int bx = 0; bx < gridDims.x; ++bx)
        {
            for(unsigned int by = 0; by < gridDims.y; ++by)
            {
                for(unsigned int bz = 0; bz < gridDims.z; ++bz)
                {
                    //
                    // set block index
                    blockIdx = vuda::dim3(bx, by, bz);

                    // reset threadIdx
                    threadIdx.reset();

                    // reset __syncthreads();
                    debug::syncthreads::reset();

                    // launch all threads
                    for(unsigned int i = 0; i < num_threads; ++i)
                        t[i] = std::thread(functor);
                    for(unsigned int i = 0; i < num_threads; ++i)
                        t[i].join();
                }
            }
        }

        delete[] t;
        return vudaSuccess;
    }

    #endif //VUDA_DEBUG_KERNEL

} //namespace vuda