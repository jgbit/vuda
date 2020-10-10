
#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include <cstdlib>
#include <random>

int blocks_and_threads(const std::thread::id tid)
{
    //
    // settings
    const int deviceID = 0;
    vuda::setDevice(deviceID);

    vuda::deviceProp prop;
    vuda::getDeviceProperties(&prop, deviceID);
                
    const int maxblocks = 256;
    const int maxthreads = prop.maxThreadsPerBlock;
    const int N = maxblocks * maxthreads;
    int* c = new int [N];

    const int stream_id = 0;
    int *dev_c;

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<uint32_t> dist(1, 100);

    //
    // allocate memory on the device
    vuda::malloc((void**)&dev_c, N * sizeof(int));

    //
    // run kernel for different settings of the number of threads
    for(int blocks = 64; blocks <= maxblocks; blocks += 64)
    {
        for(int threads = 64; threads <= maxthreads; threads += 64)
        {
            int current_size = blocks * threads;

            //
            // reset all
            memset(c, 0, N * sizeof(int));

            //
            // generate a random unsigned integer
            uint32_t rndnum  = dist(mt);

            //
            // launch kernel with threads
            vuda::launchKernel("threadid.spv", "main", stream_id, blocks,
                                threads, rndnum, dev_c);

            //
            // copy result to host
            vuda::memcpy(c, dev_c, current_size * sizeof(int), vuda::memcpyDeviceToHost);

            //
            // display result
            bool check = true;
            std::cout << "#blocks: " << blocks << " #threads: " << threads;
            for(int i = 0; i < current_size; ++i)
            {
                if(c[i] != i * (int)rndnum)
                {
                    std::cout << ", wrong result at index " << i << std::endl;
                    check = false;
                    break;
                }
            }
            if(check)
                std::cout << ", ok" << std::endl;

        }
    }

    //
    // free memory
    vuda::free(dev_c);
    delete[] c;    

    return EXIT_SUCCESS;
}

int main()
{
    try
    {
        //
        // call kernel with different num of kernel threads
        blocks_and_threads(std::this_thread::get_id());

        //
        // call kernel from different host threads with different number of kernel threads
        // ...
    }
    catch(vk::SystemError& err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << "vuda::Error: " << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cout << "unknown error" << std::endl;
        return EXIT_FAILURE;
    }

    /*std::cout << "done" << std::endl;
    std::cin.get();*/

#ifndef NDEBUG
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return EXIT_SUCCESS;
}