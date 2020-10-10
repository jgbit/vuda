#ifndef _DEBUG
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>

int main(int argc, char *argv[])
{
    try
    {
        vuda::setDevice(0);

        const int n = 64;
        int a[n], r[n], d[n];

        for(int i = 0; i < n; i++)
        {
            a[i] = i;
            r[i] = n - i - 1;
            d[i] = 0;
        }

        int *d_d;
        vuda::malloc((void**)&d_d, n * sizeof(int));
        
        //
        // run version with static shared memory
        vuda::memcpy(d_d, a, n * sizeof(int), vuda::memcpyHostToDevice);
        vuda::launchKernel("glslStaticReverse.spv", "main", 0, 1, n, d_d);
        vuda::memcpy(d, d_d, n * sizeof(int), vuda::memcpyDeviceToHost);

        for(int i = 0; i < n; i++)
            if(d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

        //
        // run version with dynamic shared memory (glsl)
        vuda::memcpy(d_d, a, n * sizeof(int), vuda::memcpyHostToDevice);
        vuda::launchKernel("glslDynamicReverse.spv", "main", 0, 1, n, d_d, n);
        vuda::memcpy(d, d_d, n * sizeof(int), vuda::memcpyDeviceToHost);

        for(int i = 0; i < n; i++)
            if(d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

        // clean up
        vuda::free(d_d);
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

#ifndef NDEBUG
    std::cout << "done." << std::endl;
    std::cin.get();
#endif
    return 0;
}