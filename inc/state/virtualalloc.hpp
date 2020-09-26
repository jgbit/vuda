#pragma once

//
// https://stackoverflow.com/questions/142508/how-do-i-check-os-with-a-preprocessor-directive/8249232

#if defined(__linux__) || defined(__unix__) || defined(__APPLE__)
#define PLATFORM_NAME VUDA_UNIX
#include <unistd.h>
#include <string.h>
#include <sys/mman.h>
#elif defined(_WIN64)
#define PLATFORM_NAME VUDA_WINDOWS
#include <Windows.h>
#endif

namespace vuda
{
    namespace detail
    {

        inline size_t reshape(size_t size, size_t alignment)
        {
            size_t nAllocatedSize = size;
            size_t mul = nAllocatedSize / alignment;

            if(nAllocatedSize % alignment != 0)
                nAllocatedSize = (mul + 1) * alignment;

            return nAllocatedSize;
        }

#if(PLATFORM_NAME == VUDA_WINDOWS)

        inline std::string GetLastErrorAsString()
        {
            // get the error message, if any.
            DWORD errorMessageID = ::GetLastError();
            if(errorMessageID == 0)
                return std::string(); // no error message has been recorded

            LPSTR messageBuffer = nullptr;
            size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                NULL, errorMessageID, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&messageBuffer, 0, NULL);

            std::string message(messageBuffer, size);

            // free the buffer.
            LocalFree(messageBuffer);

            return message;
        }

        inline void* VirtAlloc(size_t size, size_t& allocSize)
        {
            //
            // to determine the size of a page and the allocation granularity on the host computer, use the GetSystemInfo function.
            SYSTEM_INFO info;
            GetSystemInfo(&info);
            size_t nPageSize = info.dwPageSize;

            //
            // adapt size to be multiple of alignment
            allocSize = reshape(size, nPageSize);

            //
            // reserve some adress space
            LPVOID lp = VirtualAlloc(NULL, allocSize, MEM_RESERVE, PAGE_NOACCESS);

            //
            // check if reservation failed
            if(lp == NULL)
            {
                std::ostringstream ostr;
                ostr << "vuda: virtual memory reservation failed with: " << GetLastErrorAsString().c_str() << std::endl;
                throw std::runtime_error(ostr.str());
            }

            return lp;
        }

        inline void VirtFree(void *addr, const size_t length)
        {
            // If the dwFreeType parameter is MEM_RELEASE, this parameter must be 0 (zero). 
            // The function frees the entire region that is reserved in the initial allocation call to VirtualAlloc.
            BOOL ret = VirtualFree(addr, 0, MEM_RELEASE);

            if(ret == 0)
            {
                std::ostringstream ostr;
                ostr << "vuda: failed to free virtual memory reservation at " << addr << " of length " << length << "!";
                throw std::runtime_error(ostr.str());
            }
        }

#elif(PLATFORM_NAME == VUDA_UNIX)

        /*
            https://linux.die.net/man/2/mmap
            http://man7.org/linux/man-pages/man2/mmap.2.html
        */
        
        /*inline std::string get_errno(void)
        {
            const size_t buflen = 1024;
            char buf[buflen];
            
            // replacement for strerror_r
            // http://www.club.cc.cmu.edu/~cmccabe/blog_strerror.html
            // char * errorMessage = strerror_r(errno, buffer, buflen); // get string message from errno

            if(errno < sys_nerr)
                snprintf(buf, buflen, "%s", sys_errlist[errno]);
            else
                snprintf(buf, buflen, "Unknown error %d", errno);

            return buf;
        }*/

        inline void* VirtAlloc(size_t size, size_t& allocSize)
        {
            long nPageSize = sysconf(_SC_PAGE_SIZE);
            allocSize = reshape(size, nPageSize);

            //std::cout << "size requested: " << size << ", page size: " << nPageSize << ", resized: " << allocSize << std::endl;

            // If addr is NULL, then the kernel chooses the address at which to create the mapping
            void* ptr = mmap(NULL, allocSize, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);

            //
            // check if reservation failed
            if(ptr == NULL)
            {
                std::ostringstream ostr;
                ostr << "vuda: virtual memory reservation failed" << /*"with: " << get_errno() <<*/ std::endl;
                throw std::runtime_error(ostr.str());
            }

            return ptr;
        }

        inline void VirtFree(void *addr, const size_t length)
        {
            int ret = munmap(addr, length);

            if(ret == -1)
            {
                std::ostringstream ostr;
                ostr << "vuda: failed to free virtual memory reservation at " << addr << " of length " << length << /*"with: " << get_errno() <<*/ std::endl;
                throw std::runtime_error(ostr.str());
            }
        }

#endif

    } //namespace detail
} //namespace vuda