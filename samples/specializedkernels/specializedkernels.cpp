
#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include <cstdlib>

int preloadkernel(void)
{
    /*
        in vuda a pre-load shader functionality is provided
        as the kernels usually are stored in spv files

        there will be no performance-wise gain in using this functionality
        except the load can be diverted elsewhere (e.g. start of program)
    */

    return 0;
}

int blocks_and_threads(const std::thread::id tid)
{
    try
    {
        //
        // settings
        const int deviceID = 0;
        vuda::SetDevice(deviceID);

        vuda::vudaDeviceProp prop;
        vuda::GetDeviceProperties(&prop, deviceID);
                
        const int maxblocks = 256;
        const int maxthreads = prop.maxThreadsPerBlock;
        const int N = maxblocks * maxthreads;
        int* c = new int [N];

        const int stream_id = 0;
        int *dev_a = 0, *dev_b, *dev_c;

        //
        // allocate memory on the device        
        vuda::malloc((void**)&dev_a, N * sizeof(int));
        vuda::malloc((void**)&dev_b, N * sizeof(int));
        vuda::malloc((void**)&dev_c, N * sizeof(int));

        //
        // copy the arrays a and b to the device
        //vuda::memcpy(dev_a, a, N * sizeof(int), vuda::memcpyHostToDevice);
        //vuda::memcpy(dev_b, b, N * sizeof(int), vuda::memcpyHostToDevice);

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
                // launch kernel with threads
                vuda::kernelLaunch("threadid.spv", "main", blocks, stream_id, 
                                   threads, 1, dev_a, dev_b, dev_c);

                //
                // copy result to host
                vuda::memcpy(c, dev_c, current_size * sizeof(int), vuda::memcpyDeviceToHost);

                //
                // display result
                bool check = true;
                std::cout << "#blocks: " << blocks << " #threads: " << threads;
                for(int i = 0; i < current_size; ++i)
                {
                    if(c[i] != i)
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
        // free memory on device        
        vuda::free(dev_a);
        vuda::free(dev_b);
        vuda::free(dev_c);
        delete[] c;
    }
    catch(vk::SystemError err)
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

    return EXIT_SUCCESS;
}

int main()
{
    //
    // call kernel with different num of kernel threads
    blocks_and_threads(std::this_thread::get_id());

    //
    // call kernel from different host threads with different number of kernel threads
    // ...

    std::cout << "done" << std::endl;
    std::cin.get();
}