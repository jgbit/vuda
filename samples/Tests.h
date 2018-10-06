#pragma once

#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#endif

#include <vuda.hpp>

//#define THREAD_VERBOSE_OUTPUT

class Test
{
public:

    //
    // check functions
    typedef void(*funcptr)(const int);

    static void Check(funcptr ptr);
    static void Test_InstanceCreation(int tid);
    static void Test_LogicalDeviceCreation(int tid);
    static void Test_DeviceProperties(int tid);
    static void Test_MallocAndFree(int tid);
    static void Test_MallocAndMemcpyAndFree(int tid);
    static void Test_CopyComputeCopy(int tid);

    //
    // Tests
    typedef int(*funcptr2)(const int, const int, const unsigned int, int*, int*, int*);

    static int SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c);
    static int SingleThreadMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);
    static int MultipleThreadsMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);

    static void CheckResult(const unsigned int N, int* a, int* b, int* c);
    static void Launch(std::string name, const int num_threads, const unsigned int N, funcptr2 fptr);

    //
    // bst
    static void binarytreecheck(void);
};