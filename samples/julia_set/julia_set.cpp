
#include <cstdlib>
#include <cstdio>
#include <complex>
#include "miniz.h"

#include <vuda.hpp>

#if defined(_WIN64)
#pragma warning(disable : 4996)
#endif

//
// image resolution
#define DIM 4096

/*
see https://en.wikipedia.org/wiki/Julia_set
*/
int julia(int x, int y)
{
    const float scale = 1.5f;
    float jx = scale * (float)(DIM / 2 - x) / (DIM / 2);
    float jy = scale * (float)(DIM / 2 - y) / (DIM / 2);

    std::complex<float> c(-0.8f, -0.156f);
    std::complex<float> a(jx, jy);
 
    for(int i = 0; i < 200; ++i)
    {
        a = a * a + c;

        if(std::norm(a) > 1000)
            return 0;
    }
    return 1;
}

//
// host implementation
//
void host_kernel(uint8_t* ptr)
{    
    for(int y = 0; y < DIM; y++)
    {
        for(int x = 0; x < DIM; x++)
        {
            int offset = x + y * DIM;

            int juliaValue = julia(x, y);

            ptr[offset * 4 + 0] = 255 * juliaValue;
            ptr[offset * 4 + 1] = 0;
            ptr[offset * 4 + 2] = 0;
            ptr[offset * 4 + 3] = 255;
        }
    }
}

//
// device implementation
//
void device_kernel(uint8_t* ptr, const unsigned int image_size)
{
    //
    // specify device
    vuda::setDevice(0);

    //
    // allocate device mem (we use 1x uint32_t to store 4x uint8_t)
    uint32_t* dev_ptr;
    size_t image_bytesize = image_size * sizeof(uint8_t);
    vuda::malloc((void**)&dev_ptr, image_bytesize);

    //
    // launch kernel
    vuda::dim3 grid(DIM, DIM);
    vuda::launchKernel("julia_kernel.spv", "main", 0, grid, DIM, dev_ptr);

    //
    // copy result to host
    vuda::memcpy(ptr, dev_ptr, image_bytesize, vuda::memcpyDeviceToHost);

    //
    // free device memory
    vuda::free(dev_ptr);
}

int main()
{
    //
    // output dimensions
    const int iXmax = DIM;
    const int iYmax = DIM;

    //
    // output filename
    static const char *pFilename = "julia_set.png";
    uint8_t* ptr_image = (uint8_t*) std::malloc(iXmax * 4 * iYmax);

    //
    // host implementation
    //host_kernel(ptr_image);

    //
    // device implementation
    device_kernel(ptr_image, iXmax * 4 * iYmax);

    //
    // write the PNG image to file
    // SEE: https://github.com/richgel999/miniz
    //
    {
        size_t png_data_size = 0;
        void *pPNG_data = tdefl_write_image_to_png_file_in_memory_ex(ptr_image, iXmax, iYmax, 4, &png_data_size, 6, MZ_FALSE);
        if(!pPNG_data)
            fprintf(stderr, "tdefl_write_image_to_png_file_in_memory_ex() failed!\n");
        else
        {
            FILE *pFile = fopen(pFilename, "wb");
            fwrite(pPNG_data, 1, png_data_size, pFile);
            fclose(pFile);
            printf("Wrote %s\n", pFilename);
        }

        mz_free(pPNG_data);
    }

    free(ptr_image);

    return EXIT_SUCCESS;
}