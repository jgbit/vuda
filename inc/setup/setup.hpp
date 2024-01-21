#pragma once

///////////////////////////////////////////////////////////////////////////////////
// platform

#define VUDA_WINDOWS    0u
#define VUDA_UNIX       1u
#define VUDA_APPLE      2u

//
// https://stackoverflow.com/questions/142508/how-do-i-check-os-with-a-preprocessor-directive/8249232

#if defined(__linux__) || defined(__unix__)
#define PLATFORM_NAME VUDA_UNIX
#elif defined(__APPLE__)
#define PLATFORM_NAME VUDA_APPLE
#elif defined(_WIN64)
#define PLATFORM_NAME VUDA_WINDOWS
#endif

///////////////////////////////////////////////////////////////////////////////////
// values

#define VUDA_NUM_KERNEL_DESCRIPTOR_SETS 1024u
#define VUDA_MAX_QUERY_COUNT 32u