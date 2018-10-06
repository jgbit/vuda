

#include "Tests.h"

#include "timer.h"
#include <random>
#include <thread>

void Test::Test_InstanceCreation(int tid)
{
    std::ostringstream ostr;
    Timer timer;
    timer.tic();
    
    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);

    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), #devices: " << deviceCount << ", elapsed: " << timer.toc() << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Test_LogicalDeviceCreation(int tid)
{
    std::ostringstream ostr;
    Timer timer;
    timer.tic();

    const int deviceID = tid % 2;
    vuda::SetDevice(deviceID);

    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << ", elapsed: " << timer.toc() << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Test_DeviceProperties(int tid)
{
    std::ostringstream ostr;
    Timer timer;
    timer.tic();

    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);

    vuda::vudaDeviceProp prop;
    for(int i = 0; i < deviceCount; ++i)
    {
        vuda::GetDeviceProperties(&prop, i);
        ostr << "(" << i << "): " << prop.name << ", ";
    }

    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), elapsed: " << timer.toc() << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Test_MallocAndFree(int tid)
{
    const int deviceID = 0;// tid % 2;

    std::ostringstream ostr;
    //ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << std::endl;
    //std::cout << ostr.str();
    //ostr.str("");

    Timer timer;
    timer.tic();
    {   
        vuda::SetDevice(deviceID);

        int *dev_a;
        vuda::malloc((void**)&dev_a, 10000 * sizeof(int));

        vuda::free(dev_a);
    }
    double elapsed = timer.toc();
    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << ", elapsed: " << elapsed << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Test_MallocAndMemcpyAndFree(int tid)
{
    const int deviceID = 0;

    /*
        if stream_id = 0,    all threads will use the same queue/stream (sequential submission), but the commandbuffers will be generated simultaneous
        if stream_id = tid,  each thread will use a seperate queue/stream and the commandbuffers will be generated simultaneous
    */
    const int stream_id = 0;
    //const int stream_id = tid;

    std::ostringstream ostr;
    //ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << std::endl;
    //std::cout << ostr.str();
    //ostr.str("");
    
    const int N = 1000000;
    int* a = new int [N];
    int* b = new int [N];

    for(int i = 0; i < N; ++i)
    {
        a[i] = i;
        b[i] = 0;
    }

    vuda::SetDevice(deviceID);

    int *dev_a, *dev_b;
    vuda::malloc((void**)&dev_a, N * sizeof(int));
    vuda::malloc((void**)&dev_b, N * sizeof(int));

    Timer timer;
    timer.tic();
    {
        // host -> device
        vuda::memcpy(dev_a, a, N * sizeof(int), vuda::memcpyHostToDevice, stream_id);

        // device -> device
        vuda::memcpy(dev_b, dev_a, N * sizeof(int), vuda::memcpyDeviceToDevice, stream_id);

        // device -> host
        vuda::memcpy(b, dev_b, N * sizeof(int), vuda::memcpyDeviceToHost, stream_id);
    }
    double elapsed = timer.toc();

    // check result
    for(int i = 0; i < N; ++i)
    {
        if(a[i] != b[i])
        {
            ostr << "memcpy failed at index: " << i << ", ";
            break;
        }
    }

    vuda::free(dev_a);
    vuda::free(dev_b);
    delete[] a; 
    delete[] b;

    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << ", elapsed: " << elapsed << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Test_CopyComputeCopy(int tid)
{
    const int deviceID = 0;

    /*
        if stream_id = 0,    all threads will use the same queue/stream (sequential submission), but the commandbuffers will be generated simultaneous
        if stream_id = tid,  each thread will use a seperate queue/stream and the commandbuffers will be generated simultaneous
    */
    const int stream_id = 0;
    //const int stream_id = tid;

    const int blocks = 128;
    const int threads = 128;

    std::ostringstream ostr;
    //ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << std::endl;
    //std::cout << ostr.str();
    //ostr.str("");

    const int N = 1000000;
    int* a = new int[N];
    int* b = new int[N];
    int* c = new int[N];

    for(int i = 0; i < N; ++i)
    {
        a[i] = tid*i;
        b[i] = i*i;
        c[i] = 0;
    }

    vuda::SetDevice(deviceID);

    int *dev_a, *dev_b, *dev_c;
    vuda::malloc((void**)&dev_a, N * sizeof(int));
    vuda::malloc((void**)&dev_b, N * sizeof(int));
    vuda::malloc((void**)&dev_c, N * sizeof(int));

    Timer timer;
    timer.tic();
    {
        // host -> device
        vuda::memcpy(dev_a, a, N * sizeof(int), vuda::memcpyHostToDevice, stream_id);        
        vuda::memcpy(dev_b, b, N * sizeof(int), vuda::memcpyHostToDevice, stream_id);

        //
        // run kernel        
        vuda::kernelLaunch("add.spv", "main", blocks, threads, stream_id, dev_a, dev_b, dev_c, N);

        // device -> host
        vuda::memcpy(c, dev_c, N * sizeof(int), vuda::memcpyDeviceToHost, stream_id);
    }
    double elapsed = timer.toc();

    // check result
    for(int i = 0; i < N; ++i)
    {
        if(c[i] != a[i] + b[i])
        {
            ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), memcpy failed at index: " << i << ", ";
            break;
        }
    }

    vuda::free(dev_a);
    vuda::free(dev_b);
    vuda::free(dev_c);
    delete[] a;
    delete[] b;
    delete[] c;

    ostr << "thrd: (" << tid << ", " << std::this_thread::get_id() << "), device: " << deviceID << ", elapsed: " << elapsed << "s" << std::endl;
    std::cout << ostr.str();
}

void Test::Check(funcptr ptr)
{
    std::cout << std::string(40, '=') << std::endl;

    // 
    // WARM UP AND POTENTIAL HACK TO AVOID RACE CONDITION IN ACCESSING PHYSICAL DEVICES
    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);
    for(int deviceID=0; deviceID <deviceCount; ++deviceID)    
        vuda::SetDevice(deviceID);
    
    //
    // RUN THREADS
    const int num_threads = 16;
    std::thread t[num_threads];

    Timer timer;
    timer.tic();
    {
        for(int i = 0; i < num_threads; ++i)
            t[i] = std::thread(ptr, i);
        for(int i = 0; i < num_threads; ++i)
            t[i].join();
    }
    double elapsed = timer.toc();
    std::cout << "total time elapsed: " << elapsed << "s" << std::endl;
}

//
//
//

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
        vuda::kernelLaunch("add.spv", "main", blocks, threads, stream_id, dev_a, dev_b, dev_c, N);

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
        int *dev_a0 = 0, *dev_b0, *dev_c0;
        int *dev_a1 = 0, *dev_b1, *dev_c1;

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
            vuda::kernelLaunch("add.spv", "main", blocks, threads, stream0_id, dev_a0, dev_b0, dev_c0, N);
            vuda::kernelLaunch("add.spv", "main", blocks, threads, stream1_id, dev_a1, dev_b1, dev_c1, N);

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

        // run kernel
        vuda::kernelLaunch("add.spv", "main", 128, 128, stream_id, dev_a, dev_b, dev_c, N);

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
    const int totalRuns = 50;
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

//
//
//

void Test::binarytreecheck(void)
{
    vuda::bst<vuda::bst_default_node, void*> storage;

    const int num = 52;
    vuda::bst_default_node bst[num];
    vuda::bst_default_node* root = nullptr;

    int rnd[num];

    //    
    // create tree

    for(int i = 0; i<num; ++i)
    {
        bst[i].set_key(&bst[i]);
        rnd[i] = i;
    }

    // BS size
    std::ptrdiff_t diffsize = abs(static_cast<char*>(bst[1].key()) - static_cast<char*>(bst[0].key()));

    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_int_distribution<int> dist(0, num - 1);
    std::uniform_int_distribution<int> dist2(1, num - 1);

    // shuffle
    for(int i = 0; i<num; ++i)
    {
        int v1 = dist(mt);
        int v2 = dist(mt);
        std::swap(rnd[v1], rnd[v2]);
    }

    for(int i = 0; i<num; ++i)
    {
        std::cout << rnd[i] << " ";
        storage.insert_node(root, &bst[rnd[i]]);
    }
    std::cout << std::endl;

    //
    // print tree
    std::cout << "tree: " << std::endl;
    //storage.walk(root);
    storage.walk_depth(root);

    //
    // delete a node
    storage.delete_node(root, &bst[0]);

    //
    // print tree
    std::cout << "tree after deletion: " << std::endl;
    //storage.walk(root);
    storage.walk_depth(root);

    //
    // operations    

    const vuda::bst_default_node* searchnode = nullptr;
    const int ref = dist2(mt);
    void* value0 = bst[ref].key();

    const int addptr = 6;
    //void* value1 = static_cast<char*>(value0) + addptr*diffsize - 17;
    //void* value1 = static_cast<char*>(value0) + addptr*diffsize + 13;
    //void* value1 = static_cast<char*>(value0) - addptr*diffsize - 13;
    void* value1 = static_cast<char*>(value0) - addptr*diffsize + 13;

    searchnode = storage.search(root, value0);
    std::cout << "search (" << value0 << "): ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.search(root, value1);
    std::cout << "search (" << value1 << "): ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    //
    // FUN WITH POINTERS
    /*int size = sizeof(void*);
    void* ptr1 = nullptr;
    //void* ptr2 = static_cast<char*>(ptr1) + 10;
    void* ptr2 = static_cast<char*>(ptr1) + (std::numeric_limits<uint64_t>::max)() / 2 + 1;

    // std::ptrdiff_t is the signed integer type of the result of subtracting two pointers.
    std::ptrdiff_t diff12 = static_cast<char*>(ptr1) - static_cast<char*>(ptr2);
    std::ptrdiff_t diff21 = static_cast<char*>(ptr2) - static_cast<char*>(ptr1);
    auto min = PTRDIFF_MIN;
    auto max = PTRDIFF_MAX;
    uint64_t ad2 = 2 * (PTRDIFF_MAX)+1;
    auto ad = (uint64_t)ptr2;*/

    searchnode = storage.search_range(root, value1);
    std::cout << "search range (" << value1 << "): ";
    if(searchnode)
    {
        searchnode->print();
        std::cout << "distance between elements: " << (static_cast<char*>(searchnode->key()) - static_cast<char*>(value0)) / diffsize << std::endl;
    }
    else
        std::cout << "NO NODE WAS RETURNED!" << std::endl;

    searchnode = storage.minimum(root);
    std::cout << "min: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.maximum(root);
    std::cout << "max: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.predecessor(root);
    std::cout << "predecessor: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;

    searchnode = storage.successor(root);
    std::cout << "successor: ";
    if(searchnode)
        searchnode->print();
    else
        std::cout << std::endl;
}