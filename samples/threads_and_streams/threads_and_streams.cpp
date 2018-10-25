
#ifndef _DEBUG 
#define NDEBUG
#endif
#ifndef NDEBUG
#define VUDA_STD_LAYER_ENABLED
#define VUDA_DEBUG_ENABLED
#endif

#include <vuda.hpp>
#include <random>
#include "../tools/timer.hpp"

class Test
{
public:
    //
    // Tests
    typedef int(*funcptr2)(const int, const int, const unsigned int, int*, int*, int*);

    static int SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c);
    static int SingleThreadMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);
    static int MultipleThreadsMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c);

    static void CheckResult(const unsigned int N, int* a, int* b, int* c);
    static void Launch(std::string name, const int num_threads, const unsigned int N, funcptr2 fptr);
};

int Test::SingleThreadSingleStreamExample(const int tid, const int nthreads, const unsigned int N, int* a, int* b, int* c)
{
    try
    {
        //
        // settings
        const int deviceID = 0;
        vuda::SetDevice(deviceID);

        const int stream_id = 0;
        const int blocks = 128;
        const int threads = 128;

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        int *dev_a = 0, *dev_b, *dev_c;

        //
        // allocate memory on the device        
        vuda::malloc((void**)&dev_a, N * sizeof(int));
        vuda::malloc((void**)&dev_b, N * sizeof(int));
        vuda::malloc((void**)&dev_c, N * sizeof(int));

        //
        // copy the arrays a and b to the device
        vuda::memcpy(dev_a, a, N * sizeof(int), vuda::memcpyHostToDevice);
        vuda::memcpy(dev_b, b, N * sizeof(int), vuda::memcpyHostToDevice);

        //
        // run kernel                
        vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);

        //
        // copy result to host
        vuda::memcpy(c, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost);

        //
        // display results
#ifdef THREAD_VERBOSE_OUTPUT
        ostr << "thread id: " << tid << ", device id: " << deviceID << ", stream id: " << stream_id << ": ";
        std::cout << ostr.str();
#endif

        //
        // free memory on device
        vuda::free(dev_a);
        vuda::free(dev_b);
        vuda::free(dev_c);
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
        std::cout << "unknown error\n";
        return EXIT_FAILURE;
    }

    return 0;
}

/*
    Multiple CUDA streams example taken from the book CUDA BY EXAMPLE
*/
int Test::SingleThreadMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c)
{
    try
    {
        //
        // hardcore device 0
        const int deviceID = 0;
        vuda::SetDevice(deviceID);

        //
        // kernel params
        const int nstreams = 2;
        const int stream0_id = 0;
        const int stream1_id = 1;
        const int blocks = 128;
        const int threads = 128;

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        if(FULL_DATA_SIZE % 2 != 0)
            return EXIT_FAILURE;

        const int N = FULL_DATA_SIZE / 2;
        int *dev_a0, *dev_b0, *dev_c0;
        int *dev_a1, *dev_b1, *dev_c1;

        //
        // allocate memory on the device        
        vuda::malloc((void**)&dev_a0, N * sizeof(int));
        vuda::malloc((void**)&dev_b0, N * sizeof(int));
        vuda::malloc((void**)&dev_c0, N * sizeof(int));
        vuda::malloc((void**)&dev_a1, N * sizeof(int));
        vuda::malloc((void**)&dev_b1, N * sizeof(int));
        vuda::malloc((void**)&dev_c1, N * sizeof(int));

        //
        // hardcode stream submission

        for(unsigned int i = 0; i < FULL_DATA_SIZE; i += nstreams * N)
        {
            //
            // copy the arrays a and b to the device
            vuda::memcpy(dev_a0, a + i, N * sizeof(int), vuda::memcpyHostToDevice, stream0_id);
            vuda::memcpy(dev_a1, a + i + N, N * sizeof(int), vuda::memcpyHostToDevice, stream1_id);

            vuda::memcpy(dev_b0, b + i, N * sizeof(int), vuda::memcpyHostToDevice, stream0_id);
            vuda::memcpy(dev_b1, b + i + N, N * sizeof(int), vuda::memcpyHostToDevice, stream1_id);

            //
            // run kernel            
            vuda::launchKernel("add.spv", "main", stream0_id, blocks, threads, dev_a0, dev_b0, dev_c0, N);
            vuda::launchKernel("add.spv", "main", stream1_id, blocks, threads, dev_a1, dev_b1, dev_c1, N);

            //
            // copy result to host
            vuda::memcpy(c + i, dev_c0, N * sizeof(int), vuda::memcpyDeviceToHost, stream0_id);
            vuda::memcpy(c + i + N, dev_c1, N * sizeof(int), vuda::memcpyDeviceToHost, stream1_id);
        }

        //
        // display results
#ifdef THREAD_VERBOSE_OUTPUT
        ostr << "thread id: " << tid << ", device id: " << deviceID << ", stream id: " << stream0_id << ", " << stream1_id << ": ";
        std::cout << ostr.str();
#endif  

        //
        // free memory on device
        vuda::free(dev_a0);
        vuda::free(dev_b0);
        vuda::free(dev_c0);
        vuda::free(dev_a1);
        vuda::free(dev_b1);
        vuda::free(dev_c1);
    }
    catch(vk::SystemError err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cout << "unknown error\n";
        return EXIT_FAILURE;
    }

    return 0;
}

int Test::MultipleThreadsMultipleStreamsExample(const int tid, const int nthreads, const unsigned int FULL_DATA_SIZE, int* a, int* b, int* c)
{
    try
    {
        //
        // hardcode device 0
        const int deviceID = 0;
        vuda::SetDevice(deviceID);

        //
        // kernel params        
        const int stream_id = tid;
        const int blocks = 128;
        const int threads = 128;

#ifdef THREAD_VERBOSE_OUTPUT
        std::ostringstream ostr;
        std::thread::id stdtid = std::this_thread::get_id();
        ostr << "thread id: (" << tid << "," << stdtid << "), device id: " << deviceID << ", stream: " << stream_id << std::endl;
        std::cout << ostr.str();
        ostr.str("");
#endif

        if(FULL_DATA_SIZE % nthreads != 0)
        {
            std::cout << "FULL_DATA_SIZE should be divisible by " << nthreads << std::endl;
            return EXIT_FAILURE;
        }

        const int N = FULL_DATA_SIZE / nthreads;
        int *dev_a, *dev_b, *dev_c;

        //
        // allocate memory on the device
        vuda::malloc((void**)&dev_a, N * sizeof(int));
        vuda::malloc((void**)&dev_b, N * sizeof(int));
        vuda::malloc((void**)&dev_c, N * sizeof(int));

        // copy the arrays a and b to the device
        vuda::memcpy(dev_a, a + tid * N, N * sizeof(int), vuda::memcpyHostToDevice, stream_id);        
        vuda::memcpy(dev_b, b + tid * N, N * sizeof(int), vuda::memcpyHostToDevice, stream_id);
        
        //
        // run kernel
        vuda::launchKernel("add.spv", "main", stream_id, blocks, threads, dev_a, dev_b, dev_c, N);

        //
        // [until we have async memcpy, we must sync the device between threads ]
        //vuda::streamSynchronize(stream_id);

        //
        // copy result to host
        vuda::memcpy(c + tid * N, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost, stream_id);

        //
        // free memory on device
        vuda::free(dev_a);
        vuda::free(dev_b);
        vuda::free(dev_c);
    }
    catch(vk::SystemError err)
    {
        std::cout << "vk::SystemError: " << err.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    catch(...)
    {
        std::cout << "unknown error\n";
        return EXIT_FAILURE;
    }

    return 0;
}

void Test::CheckResult(const unsigned int N, int* a, int* b, int* c)
{
    std::ostringstream ostr;
    bool resultok = true;
    for(unsigned int i = 0; i < N; ++i)
    {
        if(a[i] + b[i] != c[i])
        {
            ostr << "ERROR: kernel execution did not provide the right result at index: " << i << ", ";
            ostr << a[i] << " + " << b[i] << " != " << c[i] << std::endl;
            resultok = false;
            break;
        }
    }
    if(resultok == true)
        ostr << "kernel result verified!" << std::endl;

    std::cout << ostr.str();
}

void Test::Launch(std::string name, const int num_threads, const unsigned int N, funcptr2 fptr)
{
    //
    // fill arrays on the Host
    std::cout << "Generating random host data arrays of size " << N << " (int)" << std::endl;

    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, 100);
    for(unsigned int i = 0; i < N; ++i)
    {
        a[i] = dist(mt); // -i + tid;
        b[i] = dist(mt); // i*i;
    }
    std::cout << "done." << std::endl << std::endl;

    //
    // threads    
    Timer timer;
    std::thread* t = new std::thread[num_threads];
    const int totalRuns = 12;
    double totalElapsedTime;

    std::cout << name << std::endl;
    std::cout << std::string(name.length(), '=') << std::endl;

    // 
    // WARM UP AND POTENTIAL HACK TO AVOID RACE CONDITION IN ACCESSING PHYSICAL DEVICES
    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);
    for(int deviceID = 0; deviceID < deviceCount; ++deviceID)
        vuda::SetDevice(deviceID);
    vuda::SetDevice(0);

    totalElapsedTime = 0.0;
    for(int run = 0; run < totalRuns; ++run)
    {
        // reset test
        for(unsigned int i = 0; i < N; ++i)
            c[i] = 0;

        timer.tic();
        {
            for(int i = 0; i < num_threads; ++i)
                t[i] = std::thread(fptr, i, num_threads, N, a, b, c);
            for(int i = 0; i < num_threads; ++i)
                t[i].join();
        }
        double elapsed = timer.toc();
        std::cout << "run: " << run << ", runtime: " << elapsed << "s" << std::endl;

        if(run > 0)
            totalElapsedTime += elapsed;

        // check result
        CheckResult(N, a, b, c);
    }
    if(totalRuns > 1)
        std::cout << "avg. runtime: " << totalElapsedTime / (totalRuns - 1) << "s" << std::endl;

    //
    // clean up
    delete[] t;
    delete[] a;
    delete[] b;
    delete[] c;
}

int main(int argc, char *argv[])
{
    //
    // arguments
    std::vector<const char*> args;
    for(size_t i = 0; i < argc; i++)
        args.push_back(argv[i]);

    //
    // default parameters
    // {problem size, run all tests }

    unsigned int N = 1000000;
    unsigned int test = 0;
    std::vector<bool> test_runs(3, true);
    /*if(HandleArguments(N, test) == EXIT_SUCCESS)
        return EXIT_SUCCESS;*/

    //
    // Examples

    std::vector<std::string> test_names = {
    "Single device, single thread, single stream",
    "Single device, single thread, multiple streams",
    //"Single device, multiple threads, single stream",
    "Single device, multiple threads, multiple streams"
    };

    int testid;
    //
    // single device, single thread, single stream
    testid = 0;
    if(test_runs[testid])
        Test::Launch(test_names[testid], 1, N, &Test::SingleThreadSingleStreamExample);
    
    //
    // single device, single thread, multiple streams
    /*testid = 1;
    if(test_runs[testid])
        Test::Launch(test_names[testid], 1, N, &Test::SingleThreadMultipleStreamsExample);
    
    //
    // single device, multiple threads, multiple streams
    testid = 2;
    if(test_runs[testid])
        Test::Launch(test_names[testid], 8, N, &Test::MultipleThreadsMultipleStreamsExample);*/
    
    
    /*std::cout << "done" << std::endl;
    std::cin.get();*/
    return EXIT_SUCCESS;
}