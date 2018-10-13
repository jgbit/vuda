#include <vuda.hpp>

void testfunc(void);

int main()
{
    vuda::SetDevice(0);
    testfunc();
}