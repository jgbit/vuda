#pragma once

#include <sstream>
#include <stdexcept>

inline void checkError(cudaError_t code, char const * func, const char *file, const int line, bool abort = true)
{
    if(code != cudaSuccess)
    {
        const char* errorMessage = cudaGetErrorString(code);

        std::ostringstream throwMessage;
#if defined(__NVCC__)
        throwMessage << "cuda: ";
#else
        throwMessage << "vuda: ";
#endif
        throwMessage << "error returned from " << func << " at " << file << ":" << line << ", Error code: " << code << " (" << errorMessage << ")" << std::endl;
        if(abort)
        {
            cudaDeviceReset();            
            throw std::runtime_error(throwMessage.str());
        }
    }
}

#define SafeCall(val) { checkError((val), #val, __FILE__, __LINE__); }