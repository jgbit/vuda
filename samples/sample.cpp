
#include "Tests.h"

static std::vector<const char*> args;
static std::vector<std::string> test_names = {
    "Single device, single thread, single stream",
    "Single device, single thread, multiple streams",
    //"Single device, multiple threads, single stream",
    "Single device, multiple threads, multiple streams"
};
static std::vector<bool> test_runs;

int HandleArguments(unsigned int& N, unsigned int& test)
{
    int deviceCount;
    vuda::GetDeviceCount(&deviceCount);

    if(deviceCount == 0)
    {
        std::cout << "no vulkan devices found exiting!" << std::endl;
        return EXIT_SUCCESS;
    }

    //
    // as default we run all tests
    test = (unsigned int)test_names.size();
    for(unsigned int i = 0; i < test_names.size(); ++i)
        test_runs.push_back(true);

    for(size_t i = 0; i < args.size(); i++)
    {
        //
        // print help
        if(args[i] == std::string("-help"))
        {
            std::cout << "Usage:" << std::endl;
            std::cout << "  -N <value>: specifies the problem size." << std::endl;
            std::cout << "  -test <value>: specifies the example that should run. if parameter is excluded all examples are run." << std::endl;
            return EXIT_SUCCESS;
        }

        //
        // select device
        if(args[i] == std::string("-device"))
        {            
        }

        //
        // num threads
        /*if(args[i] == std::string("-nthreads"))
        {
            char* endptr;
            int nthreads = strtol(args[i + 1], &endptr, 10);
        }*/

        //
        // problem size
        if(args[i] == std::string("-N"))
        {
            char* endptr;
            N = strtol(args[i + 1], &endptr, 10);
        }

        //
        // tests to be performed
        if(args[i] == std::string("-test"))
        {
            //
            // problem size
            char* endptr;
            test = strtol(args[i + 1], &endptr, 10);

            if(test < test_names.size())
            {
                for(unsigned int i = 0; i < test_names.size(); ++i)
                    test_runs[i] = false;
                test_runs[test] = true;
            }
            else
                test = (unsigned int)test_names.size();
        }
    }

    //
    // write out the settings
    std::ostringstream ostr;
    ostr << "Settings:" << std::endl;
    ostr << " - Problem size set to " << N << std::endl;

    for(unsigned int i = 0; i < test_names.size(); ++i)
    {
        if(test_runs[i])
            ostr << " - [x]" << test_names[i] << std::endl;
        else
            ostr << " - [ ]" << test_names[i] << std::endl;        
    }

    std::cout << ostr.str();

    return 1;
}

int main(int argc, char *argv[])
{
    //
    // small tests
    
    /*Test::Check(&Test::Test_InstanceCreation);
    Test::Check(&Test::Test_LogicalDeviceCreation);
    Test::Check(&Test::Test_DeviceProperties);
    Test::Check(&Test::Test_MallocAndFree);
    Test::Check(&Test::Test_MallocAndMemcpyAndFree);
    Test::Check(&Test::Test_CopyComputeCopy);*/
    
    //
    // arguments
    for(size_t i = 0; i < argc; i++)
        args.push_back(argv[i]);

    //
    // default parameters
    // {problem size, run all tests }

    unsigned int N = 1000000;
    unsigned int test = 0;
    if(HandleArguments(N, test) == EXIT_SUCCESS)
        return EXIT_SUCCESS;

    //
    // Examples

    //
    // single device, single thread, single stream
    int testid = 0;
    if(test_runs[testid])
        Test::Launch(test_names[testid], 1, N, &Test::SingleThreadSingleStreamExample);

    //
    // single device, single thread, multiple streams
    testid = 1;    
    if(test_runs[testid])
        Test::Launch(test_names[testid], 1, N, &Test::SingleThreadMultipleStreamsExample);
    
    //
    // single device, multiple threads, multiple streams
    testid = 2;
    if(test_runs[testid])
        Test::Launch(test_names[testid], 2, N, &Test::MultipleThreadsMultipleStreamsExample);
   
    system("pause");
    return EXIT_SUCCESS;
}