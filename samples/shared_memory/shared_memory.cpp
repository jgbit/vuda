#ifndef _DEBUG
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>

//void VudaDebugKernel(const int n, int* a, int* r);

int main(void)
{
    try
    {
        //
        // Setup    

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
        // KERNEL DEBUG
        //VudaDebugKernel(n, a, r);

        //
        // VUDA

        //
        // function arguments
        void* args[] = { (void*)&d_d, (void*)&n };    

        //
        // run version with dynamic shared memory (glsl)
        vuda::memcpy(d_d, a, n * sizeof(int), vuda::memcpyHostToDevice);
        vuda::launchKernel("glslStaticReverse.spv", "main", 0, 1, n, d_d, n);
        vuda::memcpy(d, d_d, n * sizeof(int), vuda::memcpyDeviceToHost);

        for(int i = 0; i < n; i++)
            if(d[i] != r[i]) printf("Error: d[%d]!=r[%d] (%d, %d)\n", i, i, d[i], r[i]);

        // clean up
        vuda::free(d_d);
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

    system("pause");
    return 0;
}