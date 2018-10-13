#include <vuda.hpp>

void testfunc(void)
{
    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);
    std::cout << "number of devices: " << deviceCount << std::endl;

    vuda::vudaDeviceProp prop;
    for(int i = 0; i < deviceCount; ++i)
    {
        vuda::GetDeviceProperties(&prop, i);
        std::cout << "(" << i << "): " << prop.name << std::endl;
    }
}